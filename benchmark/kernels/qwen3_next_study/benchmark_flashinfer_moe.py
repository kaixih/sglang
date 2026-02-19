"""
Benchmark comparing BF16 vs FP8 vs NVFP4 FlashInfer TRTLLM MoE for Qwen3-Next on SM100 (Blackwell).

All three data types use the flashinfer_trtllm MoE backend on SM100.
This benchmark compares their performance characteristics:
- BF16: Full precision baseline using trtllm_bf16_moe
- FP8: Block-scaled FP8 quantization using trtllm_fp8_block_scale_moe
- NVFP4: Block-scaled FP4 quantization using trtllm_fp4_block_scale_moe

Qwen3-Next MoE configuration:
- num_experts = 512
- topk = 10
- intermediate_size = 1024
- hidden_size = 4096
- routing_method_type = RenormalizeNaive (Softmax -> TopK -> Renormalize)

Note: batch_size = number of tokens, matching typical chunked prefill sizes:
- 1K-2K: Low-end GPUs (T4, 4080, A10, 4090)
- 4K: Mid-range GPUs (A100 40GB, L40)
- 8K: High-end GPUs (H100, A100 80GB, H200)
- 16K: Top-tier GPUs (B200, MI300)

Usage:
    python benchmark_flashinfer_moe.py                      # Default: 1K-16K tokens
    python benchmark_flashinfer_moe.py --ep_size 2          # With expert parallelism
    python benchmark_flashinfer_moe.py --dtype bf16         # BF16 only
    python benchmark_flashinfer_moe.py --dtype fp8          # FP8 only
    python benchmark_flashinfer_moe.py --dtype nvfp4        # NVFP4 only
    python benchmark_flashinfer_moe.py --run_correctness    # Run smoke tests
    python benchmark_flashinfer_moe.py --show_distribution  # Show token distribution
    python benchmark_flashinfer_moe.py --batch_sizes 1024 8192 16384  # Custom token counts
"""

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import triton

from sglang.srt.layers.quantization.fp8_kernel import per_token_group_quant_fp8
from sglang.srt.utils import is_flashinfer_available, next_power_of_2

if is_flashinfer_available():
    from flashinfer import fp4_quantize, nvfp4_block_scale_interleave
    from flashinfer.fused_moe import (
        convert_to_block_layout,
        trtllm_bf16_moe,
        trtllm_fp4_block_scale_moe,
        trtllm_fp8_block_scale_moe,
    )
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )


# Qwen3-Next MoE config
TOTAL_NUM_EXPERTS = 512
TOPK = 10
INTERMEDIATE_SIZE = 1024
HIDDEN_SIZE = 4096
WEIGHT_BLOCK_K = 128  # Block size for FP8 block quantization

# These will be set based on --ep_size
NUM_LOCAL_EXPERTS = TOTAL_NUM_EXPERTS
LOCAL_EXPERT_OFFSET = 0

# Routing config for Qwen3-Next (RenormalizeNaive = 4)
# Note: For benchmarking we use Default (0) since we don't have grouped routing
ROUTING_METHOD_TYPE = 4  # RenormalizeNaive

# Dtype config
DTYPE_STR = "fp8"


def set_ep_config(ep_size: int):
    """Set global expert counts based on EP size."""
    global NUM_LOCAL_EXPERTS, LOCAL_EXPERT_OFFSET
    NUM_LOCAL_EXPERTS = TOTAL_NUM_EXPERTS // ep_size
    LOCAL_EXPERT_OFFSET = 0  # For benchmarking, assume rank 0
    print(f"EP={ep_size}: Total experts={TOTAL_NUM_EXPERTS}, Local experts={NUM_LOCAL_EXPERTS}")


def analyze_token_distribution(router_logits: torch.Tensor, num_experts: int, topk: int, ep_size: int = 1):
    """Analyze and print token distribution across experts and GPUs.
    
    Args:
        router_logits: [batch_size, num_experts] routing scores
        num_experts: Total number of experts
        topk: Number of experts selected per token
        ep_size: Expert parallelism size (number of GPUs)
    
    Returns:
        Dictionary with distribution statistics
    """
    batch_size = router_logits.shape[0]
    
    # Compute softmax and get topk expert indices
    scores = torch.softmax(router_logits.float(), dim=-1)
    _, topk_ids = torch.topk(scores, k=topk, dim=-1)  # [batch_size, topk]
    
    # Count tokens per expert
    expert_counts = torch.zeros(num_experts, dtype=torch.int64, device=router_logits.device)
    for k in range(topk):
        expert_ids = topk_ids[:, k]
        expert_counts.scatter_add_(0, expert_ids, torch.ones_like(expert_ids, dtype=torch.int64))
    
    # Compute statistics
    total_assignments = batch_size * topk
    ideal_per_expert = total_assignments / num_experts
    
    counts_cpu = expert_counts.cpu().numpy()
    max_count = counts_cpu.max()
    min_count = counts_cpu.min()
    avg_count = counts_cpu.mean()
    std_count = counts_cpu.std()
    
    # Balance ratio: max / avg (1.0 = perfect balance)
    balance_ratio = max_count / avg_count if avg_count > 0 else float('inf')
    
    # How many experts are used (received at least 1 token)
    experts_used = (expert_counts > 0).sum().item()
    
    print(f"\n  Token Distribution Analysis:")
    print(f"    Batch size: {batch_size}, TopK: {topk}")
    print(f"    Total assignments: {total_assignments}")
    print(f"    Ideal per expert: {ideal_per_expert:.2f}")
    print(f"    Actual: min={min_count}, max={max_count}, avg={avg_count:.2f}, std={std_count:.2f}")
    print(f"    Experts used: {experts_used}/{num_experts} ({100*experts_used/num_experts:.1f}%)")
    print(f"    Balance ratio: {balance_ratio:.3f} (1.0 = perfect)")
    
    # GPU-level balance analysis (for EP > 1)
    gpu_stats = None
    if ep_size > 1:
        experts_per_gpu = num_experts // ep_size
        gpu_counts = torch.zeros(ep_size, dtype=torch.int64, device=router_logits.device)
        
        # Sum tokens for each GPU's local experts
        for gpu_id in range(ep_size):
            start_expert = gpu_id * experts_per_gpu
            end_expert = start_expert + experts_per_gpu
            gpu_counts[gpu_id] = expert_counts[start_expert:end_expert].sum()
        
        gpu_counts_cpu = gpu_counts.cpu().numpy()
        ideal_per_gpu = total_assignments / ep_size
        gpu_max = gpu_counts_cpu.max()
        gpu_min = gpu_counts_cpu.min()
        gpu_avg = gpu_counts_cpu.mean()
        gpu_std = gpu_counts_cpu.std()
        gpu_balance_ratio = gpu_max / gpu_avg if gpu_avg > 0 else float('inf')
        
        print(f"\n  GPU Balance Analysis (EP={ep_size}):")
        print(f"    Experts per GPU: {experts_per_gpu}")
        print(f"    Ideal tokens per GPU: {ideal_per_gpu:.2f}")
        print(f"    Actual: min={gpu_min}, max={gpu_max}, avg={gpu_avg:.2f}, std={gpu_std:.2f}")
        print(f"    GPU balance ratio: {gpu_balance_ratio:.3f} (1.0 = perfect)")
        print(f"    Per-GPU breakdown: {gpu_counts_cpu.tolist()}")
        
        gpu_stats = {
            "experts_per_gpu": experts_per_gpu,
            "ideal_per_gpu": ideal_per_gpu,
            "gpu_min": int(gpu_min),
            "gpu_max": int(gpu_max),
            "gpu_avg": float(gpu_avg),
            "gpu_std": float(gpu_std),
            "gpu_balance_ratio": float(gpu_balance_ratio),
            "gpu_counts": gpu_counts_cpu.tolist(),
        }
    
    return {
        "batch_size": batch_size,
        "topk": topk,
        "total_assignments": total_assignments,
        "ideal_per_expert": ideal_per_expert,
        "min_count": min_count,
        "max_count": max_count,
        "avg_count": avg_count,
        "std_count": std_count,
        "experts_used": experts_used,
        "balance_ratio": balance_ratio,
        "gpu_stats": gpu_stats,
    }


# ============================================================================
# FP8 MoE
# ============================================================================


def create_fp8_moe_inputs(batch_size: int):
    """Create inputs for FP8 block-scale MoE."""
    # Hidden states in bfloat16
    hidden_states = torch.randn(
        batch_size, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
    )

    # Router logits
    router_logits = torch.randn(
        batch_size, TOTAL_NUM_EXPERTS, dtype=torch.bfloat16, device="cuda"
    )

    # FP8 weights with block scales
    # w13: gate_proj and up_proj concatenated [num_experts, 2*intermediate, hidden]
    # w2: down_proj [num_experts, hidden, intermediate]
    w13_weight = torch.randn(
        NUM_LOCAL_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE,
        dtype=torch.bfloat16, device="cuda"
    ).to(torch.float8_e4m3fn)

    w2_weight = torch.randn(
        NUM_LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE,
        dtype=torch.bfloat16, device="cuda"
    ).to(torch.float8_e4m3fn)

    # Block scales for weights [num_experts, M/128, K/128]
    # Use random scales in realistic range [0.01, 0.11] to avoid overly optimistic results
    w13_weight_scale = torch.rand(
        NUM_LOCAL_EXPERTS,
        (2 * INTERMEDIATE_SIZE + 127) // 128,
        (HIDDEN_SIZE + 127) // 128,
        dtype=torch.float32, device="cuda"
    ) * 0.1 + 0.01

    w2_weight_scale = torch.rand(
        NUM_LOCAL_EXPERTS,
        (HIDDEN_SIZE + 127) // 128,
        (INTERMEDIATE_SIZE + 127) // 128,
        dtype=torch.float32, device="cuda"
    ) * 0.1 + 0.01

    return hidden_states, router_logits, w13_weight, w2_weight, w13_weight_scale, w2_weight_scale


def run_fp8_moe(hidden_states, router_logits, w13_weight, w2_weight,
                w13_weight_scale, w2_weight_scale):
    """Run FP8 block-scale MoE."""
    # Quantize hidden states to FP8 with block scales
    a_q, a_sf = per_token_group_quant_fp8(hidden_states, WEIGHT_BLOCK_K)
    a_sf_t = a_sf.t().contiguous()

    output = trtllm_fp8_block_scale_moe(
        routing_logits=router_logits,
        routing_bias=None,
        hidden_states=a_q,
        hidden_states_scale=a_sf_t,
        gemm1_weights=w13_weight,
        gemm1_weights_scale=w13_weight_scale,
        gemm2_weights=w2_weight,
        gemm2_weights_scale=w2_weight_scale,
        num_experts=TOTAL_NUM_EXPERTS,
        top_k=TOPK,
        n_group=0,
        topk_group=0,
        intermediate_size=INTERMEDIATE_SIZE,
        local_expert_offset=LOCAL_EXPERT_OFFSET,
        local_num_experts=NUM_LOCAL_EXPERTS,
        routed_scaling_factor=1.0,
        routing_method_type=ROUTING_METHOD_TYPE,
        use_shuffled_weight=False,
        tune_max_num_tokens=next_power_of_2(hidden_states.shape[0]),
    )
    return output


# ============================================================================
# BF16 MoE
# ============================================================================


def prepare_bf16_weights(num_experts: int, intermediate_size: int, hidden_size: int):
    """Prepare shuffled BF16 weights for TRTLLM MoE.
    
    The trtllm_bf16_moe kernel requires weights to be preprocessed with:
    1. Permute indices for w3_w1 reordering (gated activation)
    2. Convert to block layout with block_k=128
    """
    epilogue_tile_m = 128
    block_k = 128
    _cache_permute_indices = {}

    # Create raw BF16 weights
    # w13: gate_proj and up_proj concatenated [num_experts, 2*intermediate, hidden]
    # w2: down_proj [num_experts, hidden, intermediate]
    w13_weight_raw = torch.randn(
        num_experts, 2 * intermediate_size, hidden_size,
        dtype=torch.bfloat16, device="cuda"
    )
    w2_weight_raw = torch.randn(
        num_experts, hidden_size, intermediate_size,
        dtype=torch.bfloat16, device="cuda"
    )

    # Apply permutation and block layout per expert
    w13_shuffled_list = []
    w2_shuffled_list = []

    for i in range(num_experts):
        # Get permute indices for w13 (gated activation reordering)
        permute_indices = _maybe_get_cached_w3_w1_permute_indices(
            _cache_permute_indices,
            w13_weight_raw[i].view(torch.uint8),
            epilogue_tile_m,
        )

        # Apply permutation to w13
        tmp_w13 = (
            w13_weight_raw[i]
            .clone()
            .view(torch.uint8)[permute_indices.to("cuda")]
            .contiguous()
        )

        # Get permute indices for w2
        w2_permute_indices = get_w2_permute_indices_with_cache(
            _cache_permute_indices,
            w2_weight_raw[i].view(torch.uint8),
            epilogue_tile_m,
        )

        # Apply permutation to w2
        tmp_w2 = (
            w2_weight_raw[i]
            .clone()
            .view(torch.uint8)[w2_permute_indices.to("cuda")]
            .contiguous()
        )

        # Convert to block layout
        tmp_w13 = convert_to_block_layout(tmp_w13.view(torch.uint8), block_k)
        tmp_w2 = convert_to_block_layout(tmp_w2.view(torch.uint8), block_k)

        w13_shuffled_list.append(tmp_w13.view(torch.bfloat16))
        w2_shuffled_list.append(tmp_w2.view(torch.bfloat16))

    # Stack all experts
    w13_weight = torch.stack(w13_shuffled_list)
    w2_weight = torch.stack(w2_shuffled_list)

    return w13_weight, w2_weight


def create_bf16_moe_inputs(batch_size: int):
    """Create inputs for BF16 MoE (full precision baseline)."""
    # Hidden states in bfloat16
    hidden_states = torch.randn(
        batch_size, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
    )

    # Router logits
    router_logits = torch.randn(
        batch_size, TOTAL_NUM_EXPERTS, dtype=torch.bfloat16, device="cuda"
    )

    # Prepare shuffled BF16 weights (with block layout)
    w13_weight, w2_weight = prepare_bf16_weights(
        NUM_LOCAL_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE
    )

    return hidden_states, router_logits, w13_weight, w2_weight


def run_bf16_moe(hidden_states, router_logits, w13_weight, w2_weight):
    """Run BF16 MoE (full precision)."""
    output = trtllm_bf16_moe(
        routing_logits=router_logits,
        routing_bias=None,
        hidden_states=hidden_states,
        gemm1_weights=w13_weight,
        gemm2_weights=w2_weight,
        num_experts=TOTAL_NUM_EXPERTS,
        top_k=TOPK,
        n_group=0,
        topk_group=0,
        intermediate_size=INTERMEDIATE_SIZE,
        local_expert_offset=LOCAL_EXPERT_OFFSET,
        local_num_experts=NUM_LOCAL_EXPERTS,
        routed_scaling_factor=1.0,
        routing_method_type=ROUTING_METHOD_TYPE,
        tune_max_num_tokens=next_power_of_2(hidden_states.shape[0]),
    )
    return output


# ============================================================================
# NVFP4 MoE
# ============================================================================


def prepare_fp4_weights(num_experts: int, intermediate_size: int, hidden_size: int):
    """Prepare shuffled FP4 weights for TRTLLM MoE."""
    epilogue_tile_m = 128
    _cache_permute_indices = {}

    # Create random FP4-like weights (packed as uint8)
    # w13: [num_experts, 2*intermediate, hidden/2] (packed)
    # w2: [num_experts, hidden, intermediate/2] (packed)
    gemm1_weights_fp4 = torch.randint(
        0, 256,
        (num_experts, 2 * intermediate_size, hidden_size // 2),
        dtype=torch.uint8, device="cuda"
    ).view(torch.float8_e4m3fn)

    gemm2_weights_fp4 = torch.randint(
        0, 256,
        (num_experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8, device="cuda"
    ).view(torch.float8_e4m3fn)

    # Scales [num_experts, M, K/16]
    # Use random scales in realistic range to avoid overly optimistic results
    gemm1_scales = (torch.rand(
        num_experts, 2 * intermediate_size, hidden_size // 16,
        dtype=torch.float32, device="cuda"
    ) * 0.1 + 0.01).to(torch.float8_e4m3fn)

    gemm2_scales = (torch.rand(
        num_experts, hidden_size, intermediate_size // 16,
        dtype=torch.float32, device="cuda"
    ) * 0.1 + 0.01).to(torch.float8_e4m3fn)

    # Shuffle for MMA (per-expert)
    gemm1_weights_shuffled_list = []
    gemm1_scales_shuffled_list = []
    gemm2_weights_shuffled_list = []
    gemm2_scales_shuffled_list = []

    for i in range(num_experts):
        # Get permute indices for w13 (gated activation reordering + MMA shuffle)
        # API: _maybe_get_cached_w3_w1_permute_indices(cache, weight, epilogue_tile_m)
        permute_indices = _maybe_get_cached_w3_w1_permute_indices(
            _cache_permute_indices,
            gemm1_weights_fp4[i].view(torch.uint8),
            epilogue_tile_m,
        )

        # Get permute indices for scales
        permute_sf_indices = _maybe_get_cached_w3_w1_permute_indices(
            _cache_permute_indices,
            gemm1_scales[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
        )

        # Get w2 permute indices
        w2_permute_indices = get_w2_permute_indices_with_cache(
            _cache_permute_indices,
            gemm2_weights_fp4[i].view(torch.uint8),
            epilogue_tile_m,
        )

        w2_scale_permute_indices = get_w2_permute_indices_with_cache(
            _cache_permute_indices,
            gemm2_scales[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
        )

        # Apply permutations
        gemm1_weights_shuffled_list.append(
            gemm1_weights_fp4[i].view(torch.uint8)[permute_indices.to("cuda")].contiguous()
        )
        gemm1_scales_shuffled_list.append(
            nvfp4_block_scale_interleave(
                gemm1_scales[i].view(torch.uint8)[permute_sf_indices.to("cuda")].contiguous()
            )
        )
        gemm2_weights_shuffled_list.append(
            gemm2_weights_fp4[i].view(torch.uint8)[w2_permute_indices.to("cuda")].contiguous()
        )
        gemm2_scales_shuffled_list.append(
            nvfp4_block_scale_interleave(
                gemm2_scales[i].view(torch.uint8)[w2_scale_permute_indices.to("cuda")].contiguous()
            )
        )

    # Keep weights as uint8 (FP4 packed), scales as uint8 (will be viewed as fp8 in kernel call)
    gemm1_weights_shuffled = torch.stack(gemm1_weights_shuffled_list)  # uint8
    gemm1_scales_shuffled = torch.stack(gemm1_scales_shuffled_list)    # uint8
    gemm2_weights_shuffled = torch.stack(gemm2_weights_shuffled_list)  # uint8
    gemm2_scales_shuffled = torch.stack(gemm2_scales_shuffled_list)    # uint8

    return (gemm1_weights_shuffled, gemm1_scales_shuffled,
            gemm2_weights_shuffled, gemm2_scales_shuffled)


def create_nvfp4_moe_inputs(batch_size: int):
    """Create inputs for NVFP4 block-scale MoE."""
    # Hidden states in bfloat16
    hidden_states = torch.randn(
        batch_size, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
    )

    # Router logits
    router_logits = torch.randn(
        batch_size, TOTAL_NUM_EXPERTS, dtype=torch.bfloat16, device="cuda"
    )

    # Prepare shuffled FP4 weights
    (gemm1_weights_shuffled, gemm1_scales_shuffled,
     gemm2_weights_shuffled, gemm2_scales_shuffled) = prepare_fp4_weights(
        NUM_LOCAL_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE
    )

    # Scaling factors (per-expert) - use random values to avoid overly optimistic results
    g1_scale_c = torch.rand(NUM_LOCAL_EXPERTS, dtype=torch.float32, device="cuda") * 0.1 + 0.01
    g1_alphas = torch.rand(NUM_LOCAL_EXPERTS, dtype=torch.float32, device="cuda") * 0.1 + 0.01
    g2_alphas = torch.rand(NUM_LOCAL_EXPERTS, dtype=torch.float32, device="cuda") * 0.1 + 0.01
    w13_input_scale_quant = torch.rand(1, dtype=torch.float32, device="cuda") * 0.1 + 0.01

    return (hidden_states, router_logits,
            gemm1_weights_shuffled, gemm1_scales_shuffled,
            gemm2_weights_shuffled, gemm2_scales_shuffled,
            g1_scale_c, g1_alphas, g2_alphas, w13_input_scale_quant)


def run_nvfp4_moe(hidden_states, router_logits,
                  gemm1_weights_shuffled, gemm1_scales_shuffled,
                  gemm2_weights_shuffled, gemm2_scales_shuffled,
                  g1_scale_c, g1_alphas, g2_alphas, w13_input_scale_quant):
    """Run NVFP4 block-scale MoE."""
    # Quantize hidden states to FP4
    hs_fp4_bytes, hs_sf_bytes = fp4_quantize(
        hidden_states,
        w13_input_scale_quant,
        16,  # sf_vec_size
        False,  # use_ue8m0
        False,  # is_sf_swizzled_layout
    )

    seq_len, hidden_size = hidden_states.shape
    hs_fp4 = hs_fp4_bytes.reshape(seq_len, hidden_size // 2)
    hs_sf = hs_sf_bytes.view(torch.float8_e4m3fn).reshape(seq_len, hidden_size // 16)

    # Output buffer
    output = torch.empty(
        seq_len, hidden_size, dtype=torch.bfloat16, device="cuda"
    )

    result = trtllm_fp4_block_scale_moe(
        routing_logits=router_logits,
        routing_bias=None,
        hidden_states=hs_fp4,
        hidden_states_scale=hs_sf.view(torch.float8_e4m3fn).flatten(),
        gemm1_weights=gemm1_weights_shuffled,  # uint8 (FP4 packed)
        gemm1_weights_scale=gemm1_scales_shuffled.view(torch.float8_e4m3fn),
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=gemm2_weights_shuffled,  # uint8 (FP4 packed)
        gemm2_weights_scale=gemm2_scales_shuffled.view(torch.float8_e4m3fn),
        gemm2_bias=None,
        output1_scale_scalar=g1_scale_c,
        output1_scale_gate_scalar=g1_alphas,
        output2_scale_scalar=g2_alphas,
        num_experts=TOTAL_NUM_EXPERTS,
        top_k=TOPK,
        n_group=0,
        topk_group=0,
        intermediate_size=INTERMEDIATE_SIZE,
        local_expert_offset=LOCAL_EXPERT_OFFSET,
        local_num_experts=NUM_LOCAL_EXPERTS,
        routed_scaling_factor=1.0,
        routing_method_type=ROUTING_METHOD_TYPE,
        do_finalize=True,
        tune_max_num_tokens=next_power_of_2(seq_len),
        output=output,
    )[0]

    return result


# ============================================================================
# TFLOPS Calculation
# ============================================================================


def compute_moe_flops(batch_size: int) -> int:
    """
    Compute FLOPs for one MoE forward pass on a single GPU with EP.

    With Expert Parallelism (EP), each GPU only computes for its local experts.
    With uniform routing, the expected local experts per token is:
        local_experts_per_token = TOPK * (NUM_LOCAL_EXPERTS / TOTAL_NUM_EXPERTS)

    Each expert has two GEMMs:
    - GEMM1: [1, hidden] x [hidden, 2*intermediate] = 2 * hidden * 2*intermediate
    - GEMM2: [1, intermediate] x [intermediate, hidden] = 2 * intermediate * hidden

    Total per token = local_experts_per_token * 6 * intermediate * hidden
    """
    # Expected number of local experts per token (with uniform routing)
    local_experts_per_token = TOPK * NUM_LOCAL_EXPERTS / TOTAL_NUM_EXPERTS
    flops_per_token = local_experts_per_token * 6 * INTERMEDIATE_SIZE * HIDDEN_SIZE
    return int(batch_size * flops_per_token)


def compute_tflops(batch_size: int, time_ms: float) -> float:
    """Compute TFLOPS from batch size and time in ms."""
    flops = compute_moe_flops(batch_size)
    return flops / (time_ms * 1e-3) / 1e12


# ============================================================================
# Benchmark Functions
# ============================================================================


def create_shared_inputs(batch_size: int):
    """Create shared hidden_states and router_logits for fair comparison."""
    hidden_states = torch.randn(
        batch_size, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    router_logits = torch.randn(
        batch_size, TOTAL_NUM_EXPERTS, dtype=torch.bfloat16, device="cuda"
    )
    return hidden_states, router_logits


def benchmark_bf16_moe(batch_size: int, shared_hidden_states=None, shared_router_logits=None):
    """Benchmark BF16 MoE kernel using triton.testing.do_bench."""
    # Use shared inputs if provided, otherwise create new ones
    if shared_hidden_states is not None and shared_router_logits is not None:
        hidden_states = shared_hidden_states
        router_logits = shared_router_logits
        # Create preprocessed BF16 weights (with block layout)
        w13_weight, w2_weight = prepare_bf16_weights(
            NUM_LOCAL_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE
        )
    else:
        inputs = create_bf16_moe_inputs(batch_size)
        hidden_states, router_logits, w13_weight, w2_weight = inputs

    def kernel_fn():
        return trtllm_bf16_moe(
            routing_logits=router_logits,
            routing_bias=None,
            hidden_states=hidden_states,
            gemm1_weights=w13_weight,
            gemm2_weights=w2_weight,
            num_experts=TOTAL_NUM_EXPERTS,
            top_k=TOPK,
            n_group=0,
            topk_group=0,
            intermediate_size=INTERMEDIATE_SIZE,
            local_expert_offset=LOCAL_EXPERT_OFFSET,
            local_num_experts=NUM_LOCAL_EXPERTS,
            routed_scaling_factor=1.0,
            routing_method_type=ROUTING_METHOD_TYPE,
            tune_max_num_tokens=next_power_of_2(batch_size),
        )

    # Use triton's do_bench for proper warmup and timing
    time_ms = triton.testing.do_bench(kernel_fn, warmup=25, rep=100)
    return time_ms


def benchmark_fp8_moe(batch_size: int, shared_hidden_states=None, shared_router_logits=None):
    """Benchmark FP8 MoE kernel using triton.testing.do_bench."""
    # Use shared inputs if provided, otherwise create new ones
    if shared_hidden_states is not None and shared_router_logits is not None:
        hidden_states = shared_hidden_states
        router_logits = shared_router_logits
        # Create FP8 weights
        w13_weight = torch.randn(
            NUM_LOCAL_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE,
            dtype=torch.bfloat16, device="cuda"
        ).to(torch.float8_e4m3fn)
        w2_weight = torch.randn(
            NUM_LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE,
            dtype=torch.bfloat16, device="cuda"
        ).to(torch.float8_e4m3fn)
        w13_weight_scale = torch.rand(
            NUM_LOCAL_EXPERTS,
            (2 * INTERMEDIATE_SIZE + 127) // 128,
            (HIDDEN_SIZE + 127) // 128,
            dtype=torch.float32, device="cuda"
        ) * 0.1 + 0.01
        w2_weight_scale = torch.rand(
            NUM_LOCAL_EXPERTS,
            (HIDDEN_SIZE + 127) // 128,
            (INTERMEDIATE_SIZE + 127) // 128,
            dtype=torch.float32, device="cuda"
        ) * 0.1 + 0.01
    else:
        inputs = create_fp8_moe_inputs(batch_size)
        hidden_states, router_logits, w13_weight, w2_weight, w13_weight_scale, w2_weight_scale = inputs

    # Pre-quantize hidden states
    a_q, a_sf = per_token_group_quant_fp8(hidden_states, WEIGHT_BLOCK_K)
    a_sf_t = a_sf.t().contiguous()

    def kernel_fn():
        return trtllm_fp8_block_scale_moe(
            routing_logits=router_logits,
            routing_bias=None,
            hidden_states=a_q,
            hidden_states_scale=a_sf_t,
            gemm1_weights=w13_weight,
            gemm1_weights_scale=w13_weight_scale,
            gemm2_weights=w2_weight,
            gemm2_weights_scale=w2_weight_scale,
            num_experts=TOTAL_NUM_EXPERTS,
            top_k=TOPK,
            n_group=0,
            topk_group=0,
            intermediate_size=INTERMEDIATE_SIZE,
            local_expert_offset=LOCAL_EXPERT_OFFSET,
            local_num_experts=NUM_LOCAL_EXPERTS,
            routed_scaling_factor=1.0,
            routing_method_type=ROUTING_METHOD_TYPE,
            use_shuffled_weight=False,
            tune_max_num_tokens=next_power_of_2(batch_size),
        )

    # Use triton's do_bench for proper warmup and timing
    time_ms = triton.testing.do_bench(kernel_fn, warmup=25, rep=100)
    return time_ms


def benchmark_nvfp4_moe(batch_size: int, shared_hidden_states=None, shared_router_logits=None):
    """Benchmark NVFP4 MoE kernel using triton.testing.do_bench."""
    # Use shared inputs if provided, otherwise create new ones
    if shared_hidden_states is not None and shared_router_logits is not None:
        hidden_states = shared_hidden_states
        router_logits = shared_router_logits
        # Create NVFP4 weights
        (gemm1_weights_shuffled, gemm1_scales_shuffled,
         gemm2_weights_shuffled, gemm2_scales_shuffled) = prepare_fp4_weights(
            NUM_LOCAL_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE
        )
        g1_scale_c = torch.rand(1, dtype=torch.float32, device="cuda") * 0.1 + 0.01
        g1_alphas = torch.rand(NUM_LOCAL_EXPERTS, dtype=torch.float32, device="cuda") * 0.1 + 0.01
        g2_alphas = torch.rand(NUM_LOCAL_EXPERTS, dtype=torch.float32, device="cuda") * 0.1 + 0.01
        w13_input_scale_quant = torch.rand(1, dtype=torch.float32, device="cuda") * 0.1 + 0.01
    else:
        inputs = create_nvfp4_moe_inputs(batch_size)
        (hidden_states, router_logits,
         gemm1_weights_shuffled, gemm1_scales_shuffled,
         gemm2_weights_shuffled, gemm2_scales_shuffled,
         g1_scale_c, g1_alphas, g2_alphas, w13_input_scale_quant) = inputs

    # Pre-quantize hidden states to FP4
    hs_fp4_bytes, hs_sf_bytes = fp4_quantize(
        hidden_states,
        w13_input_scale_quant,
        16,
        False,
        False,
    )
    seq_len, hidden_size = hidden_states.shape
    hs_fp4 = hs_fp4_bytes.reshape(seq_len, hidden_size // 2)
    hs_sf = hs_sf_bytes.view(torch.float8_e4m3fn).reshape(seq_len, hidden_size // 16)

    # Output buffer
    output = torch.empty(seq_len, hidden_size, dtype=torch.bfloat16, device="cuda")

    def kernel_fn():
        return trtllm_fp4_block_scale_moe(
            routing_logits=router_logits,
            routing_bias=None,
            hidden_states=hs_fp4,
            hidden_states_scale=hs_sf.view(torch.float8_e4m3fn).flatten(),
            gemm1_weights=gemm1_weights_shuffled,  # uint8 (FP4 packed)
            gemm1_weights_scale=gemm1_scales_shuffled.view(torch.float8_e4m3fn),
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=gemm2_weights_shuffled,  # uint8 (FP4 packed)
            gemm2_weights_scale=gemm2_scales_shuffled.view(torch.float8_e4m3fn),
            gemm2_bias=None,
            output1_scale_scalar=g1_scale_c,
            output1_scale_gate_scalar=g1_alphas,
            output2_scale_scalar=g2_alphas,
            num_experts=TOTAL_NUM_EXPERTS,
            top_k=TOPK,
            n_group=0,
            topk_group=0,
            intermediate_size=INTERMEDIATE_SIZE,
            local_expert_offset=LOCAL_EXPERT_OFFSET,
            local_num_experts=NUM_LOCAL_EXPERTS,
            routed_scaling_factor=1.0,
            routing_method_type=ROUTING_METHOD_TYPE,
            do_finalize=True,
            tune_max_num_tokens=next_power_of_2(seq_len),
            output=output,
        )[0]

    # Use triton's do_bench for proper warmup and timing
    time_ms = triton.testing.do_bench(kernel_fn, warmup=25, rep=100)
    return time_ms


def run_benchmarks(batch_sizes: list, ep_sizes: list, dtypes: list, show_distribution: bool = False):
    """Run benchmarks and print results."""
    print("\n" + "=" * 100)
    print("FlashInfer TRTLLM MoE Benchmark: BF16 vs FP8 vs NVFP4 for Qwen3-Next")
    print("=" * 100)
    print(f"Config: num_experts={TOTAL_NUM_EXPERTS}, topk={TOPK}, "
          f"intermediate={INTERMEDIATE_SIZE}, hidden={HIDDEN_SIZE}")
    print("=" * 100)

    results = []

    for ep_size in ep_sizes:
        set_ep_config(ep_size)

        for batch_size in batch_sizes:
            # Create shared inputs for fair comparison across all dtypes
            shared_hidden_states, shared_router_logits = create_shared_inputs(batch_size)

            # Show token distribution if requested (uses same router_logits as benchmark)
            if show_distribution:
                print(f"\n[EP={ep_size}, Tokens={batch_size}]")
                analyze_token_distribution(shared_router_logits, TOTAL_NUM_EXPERTS, TOPK, ep_size)

            for dtype in dtypes:
                try:
                    if dtype == "bf16":
                        time_ms = benchmark_bf16_moe(batch_size, shared_hidden_states, shared_router_logits)
                    elif dtype == "fp8":
                        time_ms = benchmark_fp8_moe(batch_size, shared_hidden_states, shared_router_logits)
                    else:  # nvfp4
                        time_ms = benchmark_nvfp4_moe(batch_size, shared_hidden_states, shared_router_logits)

                    tflops = compute_tflops(batch_size, time_ms)
                    results.append({
                        "ep_size": ep_size,
                        "batch_size": batch_size,
                        "dtype": dtype,
                        "time_ms": time_ms,
                        "tflops": tflops,
                    })
                except Exception as e:
                    print(f"Error with ep={ep_size}, batch={batch_size}, dtype={dtype}: {e}")
                    results.append({
                        "ep_size": ep_size,
                        "batch_size": batch_size,
                        "dtype": dtype,
                        "time_ms": float("nan"),
                        "tflops": float("nan"),
                    })

    # Print comparison table with TFLOPS
    print("\n" + "=" * 160)
    print("BF16 vs FP8 vs NVFP4 Comparison")
    print("=" * 160)
    print(f"{'EP':>4} | {'Tokens':>10} | {'BF16 (ms)':>10} | {'BF16 TFLOPS':>12} | {'FP8 (ms)':>10} | {'FP8 TFLOPS':>12} | {'NVFP4 (ms)':>12} | {'NVFP4 TFLOPS':>14} | {'Best':>10}")
    print("-" * 160)

    for ep_size in ep_sizes:
        for batch_size in batch_sizes:
            bf16_results = [r for r in results if r['ep_size'] == ep_size
                          and r['batch_size'] == batch_size and r['dtype'] == 'bf16']
            fp8_results = [r for r in results if r['ep_size'] == ep_size
                          and r['batch_size'] == batch_size and r['dtype'] == 'fp8']
            nvfp4_results = [r for r in results if r['ep_size'] == ep_size
                            and r['batch_size'] == batch_size and r['dtype'] == 'nvfp4']

            bf16_time = bf16_results[0]['time_ms'] if bf16_results else float('nan')
            bf16_tflops = bf16_results[0]['tflops'] if bf16_results else float('nan')
            fp8_time = fp8_results[0]['time_ms'] if fp8_results else float('nan')
            fp8_tflops = fp8_results[0]['tflops'] if fp8_results else float('nan')
            nvfp4_time = nvfp4_results[0]['time_ms'] if nvfp4_results else float('nan')
            nvfp4_tflops = nvfp4_results[0]['tflops'] if nvfp4_results else float('nan')

            bf16_time_str = f"{bf16_time:.3f}" if bf16_time == bf16_time else "N/A"
            bf16_tflops_str = f"{bf16_tflops:.2f}" if bf16_tflops == bf16_tflops else "N/A"
            fp8_time_str = f"{fp8_time:.3f}" if fp8_time == fp8_time else "N/A"
            fp8_tflops_str = f"{fp8_tflops:.2f}" if fp8_tflops == fp8_tflops else "N/A"
            nvfp4_time_str = f"{nvfp4_time:.3f}" if nvfp4_time == nvfp4_time else "N/A"
            nvfp4_tflops_str = f"{nvfp4_tflops:.2f}" if nvfp4_tflops == nvfp4_tflops else "N/A"

            # Find the best (lowest time)
            times = {'BF16': bf16_time, 'FP8': fp8_time, 'NVFP4': nvfp4_time}
            valid_times = {k: v for k, v in times.items() if v == v}  # Filter out NaN
            if valid_times:
                best = min(valid_times, key=valid_times.get)
                best_time = valid_times[best]
                # Calculate speedup vs BF16 baseline
                if bf16_time == bf16_time and best_time < bf16_time:
                    speedup = bf16_time / best_time
                    best_str = f"{best} {speedup:.2f}x"
                else:
                    best_str = best
            else:
                best_str = "N/A"

            print(f"{ep_size:>4} | {batch_size:>10} | {bf16_time_str:>10} | {bf16_tflops_str:>12} | {fp8_time_str:>10} | {fp8_tflops_str:>12} | {nvfp4_time_str:>12} | {nvfp4_tflops_str:>14} | {best_str:>10}")

    print("-" * 160)


def run_smoke_tests():
    """Run smoke tests to verify kernels run without errors (NaN/Inf)."""
    print("\n" + "=" * 80)
    print("Smoke Tests: Verify kernels run without errors")
    print("=" * 80)

    batch_size = 16

    # Test BF16
    print(f"\n[Test] BF16 MoE (batch={batch_size}):")
    try:
        bf16_inputs = create_bf16_moe_inputs(batch_size)
        hidden_states, router_logits = bf16_inputs[0], bf16_inputs[1]
        
        # Analyze token distribution
        analyze_token_distribution(router_logits, TOTAL_NUM_EXPERTS, TOPK)
        
        bf16_output = run_bf16_moe(*bf16_inputs)
        print(f"  Output shape: {bf16_output.shape}")
        print(f"  Output dtype: {bf16_output.dtype}")
        print(f"  Output range: [{bf16_output.float().min().item():.4f}, {bf16_output.float().max().item():.4f}]")
        has_nan = torch.isnan(bf16_output.float()).any().item()
        has_inf = torch.isinf(bf16_output.float()).any().item()
        print(f"  Has NaN: {has_nan}, Has Inf: {has_inf}")

        if not has_nan and not has_inf:
            print("  [PASS] BF16 MoE kernel runs correctly")
        else:
            print("  [FAIL] BF16 MoE has invalid values")

    except Exception as e:
        print(f"  [FAIL] BF16 MoE failed: {e}")

    # Test FP8
    print(f"\n[Test] FP8 MoE (batch={batch_size}):")
    try:
        fp8_inputs = create_fp8_moe_inputs(batch_size)
        hidden_states, router_logits = fp8_inputs[0], fp8_inputs[1]
        
        # Analyze token distribution
        analyze_token_distribution(router_logits, TOTAL_NUM_EXPERTS, TOPK)
        
        fp8_output = run_fp8_moe(*fp8_inputs)
        print(f"  Output shape: {fp8_output.shape}")
        print(f"  Output dtype: {fp8_output.dtype}")
        print(f"  Output range: [{fp8_output.float().min().item():.4f}, {fp8_output.float().max().item():.4f}]")
        has_nan = torch.isnan(fp8_output.float()).any().item()
        has_inf = torch.isinf(fp8_output.float()).any().item()
        print(f"  Has NaN: {has_nan}, Has Inf: {has_inf}")

        if not has_nan and not has_inf:
            print("  [PASS] FP8 MoE kernel runs correctly")
        else:
            print("  [FAIL] FP8 MoE has invalid values")

    except Exception as e:
        print(f"  [FAIL] FP8 MoE failed: {e}")

    # Test NVFP4
    print(f"\n[Test] NVFP4 MoE (batch={batch_size}):")
    try:
        nvfp4_inputs = create_nvfp4_moe_inputs(batch_size)
        nvfp4_output = run_nvfp4_moe(*nvfp4_inputs)
        print(f"  Output shape: {nvfp4_output.shape}")
        print(f"  Output dtype: {nvfp4_output.dtype}")
        print(f"  Output range: [{nvfp4_output.float().min().item():.4f}, {nvfp4_output.float().max().item():.4f}]")
        has_nan = torch.isnan(nvfp4_output.float()).any().item()
        has_inf = torch.isinf(nvfp4_output.float()).any().item()
        print(f"  Has NaN: {has_nan}, Has Inf: {has_inf}")

        if not has_nan and not has_inf:
            print("  [PASS] NVFP4 MoE kernel runs correctly")
        else:
            print("  [FAIL] NVFP4 MoE has invalid values")

    except Exception as e:
        print(f"  [FAIL] NVFP4 MoE failed: {e}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark FP8 vs NVFP4 MoE")
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192, 16384],
        help="Number of tokens to benchmark (default: 1K-16K matching chunked prefill sizes)"
    )
    parser.add_argument(
        "--ep_size",
        type=int,
        default=None,
        help="Expert parallelism size (default: auto-sweep [1, 2, 4])"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["bf16", "fp8", "nvfp4"],
        help="Data type (default: auto-sweep all three)"
    )
    parser.add_argument(
        "--run_correctness",
        action="store_true",
        help="Run smoke tests (verify kernels run without NaN/Inf)"
    )
    parser.add_argument(
        "--show_distribution",
        action="store_true",
        help="Show token distribution across experts for each batch size"
    )

    args = parser.parse_args()

    # Determine EP sizes
    if args.ep_size is not None:
        ep_sizes = [args.ep_size]
    else:
        ep_sizes = [1, 2, 4]

    # Determine dtypes
    if args.dtype is not None:
        dtypes = [args.dtype]
    else:
        dtypes = ["bf16", "fp8", "nvfp4"]

    if args.run_correctness:
        set_ep_config(1)
        run_smoke_tests()

    run_benchmarks(args.batch_sizes, ep_sizes, dtypes, show_distribution=args.show_distribution)
