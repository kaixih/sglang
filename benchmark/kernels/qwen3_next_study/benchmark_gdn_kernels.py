"""Benchmark GDN kernels for decode and prefill phases.

This benchmark tests the Gated Delta Net (GDN) kernel implementations used in Qwen3-Next models.

DECODE MODE (--decode):
- Tests single-token generation kernels (T=1)
- CuTe DSL (GB200 only) vs Triton (all GPUs)
- Kernels: cutedsl_fused_sigmoid_gating_delta_rule_update vs fused_sigmoid_gating_delta_rule_update

PREFILL MODE (default):
- Tests multi-token prefill kernels (T≥128)
- Chunk-based Triton implementation
- Kernel: chunk_gated_delta_rule (Triton, no CuTe DSL version)

Target shapes (Qwen3-Next):
- QK num_heads (H): 16
- V num_heads (HV): 64
- Head dimension (K, V): 128

NOTE: Batch size labels (SmallBatch/LargeBatch) are for demonstration only and do not
reflect SGLang's actual kernel selection. SGLang currently uses a static choice based on
SGLANG_USE_CUTEDSL_GDN_DECODE environment variable (default: Triton for all batch sizes).
For optimal performance, dynamic kernel selection based on batch size would be beneficial.
"""

import argparse
import sys
from typing import Dict, Tuple

import torch
import triton

try:
    import cuda.bindings.driver as cuda_driver
    import cutlass  # noqa: F401
    from cutlass.cute.runtime import from_dlpack

    from sglang.jit_kernel import cutedsl_gdn

    CUTEDSL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: CuTe DSL not available: {e}")
    CUTEDSL_AVAILABLE = False
    cutedsl_gdn = None

try:
    from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
    from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

    TRITON_AVAILABLE = True
except ImportError as e:
    print(f"Error: Triton kernels not available: {e}")

try:
    from flashinfer.cute_dsl.gated_delta_rule import gated_delta_rule as flashinfer_gated_delta_rule

    FLASHINFER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: FlashInfer CuTe DSL not available: {e}")
    FLASHINFER_AVAILABLE = False
    flashinfer_gated_delta_rule = None
    TRITON_AVAILABLE = False


def run_triton_decode_kernel(A_log, dt_bias, q, k, v, a, b, initial_state, indices, scale):
    """Run Triton decode kernel (recurrent, T=1)."""
    return fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=initial_state,
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=None,
    )


def run_triton_prefill_kernel(A_log, dt_bias, q, k, v, a, b, initial_state, indices, scale, cu_seqlens):
    """Run Triton prefill kernel (chunk-based, T≥128)."""
    # fused_gdn_gating expects 2D: [seq_len, num_heads]
    # But we have 3D varlen: [1, total_seq_len, num_heads]
    # Squeeze to get [total_seq_len, num_heads]
    a_2d = a.squeeze(0)  # [total_seq_len, num_heads]
    b_2d = b.squeeze(0)  # [total_seq_len, num_heads]

    # Compute g and beta using fused_gdn_gating
    g, beta = fused_gdn_gating(A_log, a_2d, b_2d, dt_bias)

    # Call chunk_gated_delta_rule
    core_attn_out, last_recurrent_state, h = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        initial_state_indices=indices,
        cu_seqlens=cu_seqlens,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    return core_attn_out


def run_flashinfer_decode_kernel(A_log, dt_bias, q, k, v, a, b, initial_state, indices, scale, debug=False):
    """Run FlashInfer CuTe DSL decode kernel (T=1,2,3,4).

    NOTE: This is an isolated wrapper for FlashInfer's API which is still in review
    and may change. Key differences:
    - State layout: SGLang [B, HV, K, V] (V-fast) vs FlashInfer [B, HV, V, K] (K-fast)
    - State dtype: FlashInfer expects bfloat16, SGLang uses float32

    IMPORTANT: For benchmarking, initial_state should already be in FlashInfer format
    ([B, HV, V, K], bfloat16) to avoid conversion overhead.
    """
    if debug:
        B, T = q.shape[:2]
        print(f"[DEBUG] FlashInfer input: B={B}, T={T}, state_shape={initial_state.shape}, state_dtype={initial_state.dtype}")

    # Call FlashInfer's gated_delta_rule
    # Assumes initial_state is already in correct layout [B, HV, V, K] and dtype (bfloat16)
    output = flashinfer_gated_delta_rule(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=initial_state,
        initial_state_indices=indices,  # Unused by FlashInfer but kept for compatibility
        use_qk_l2norm_in_kernel=True,
        scale=scale,
    )

    return output


def gdn_flops(
    total_seq_len: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    num_seqs: int,
) -> int:
    """Calculate FLOPs for Gated Delta Rule (GDN) attention."""
    num_o_heads = max(num_q_heads, num_v_heads)
    # k @ v^T (outer product): 2 * d^2 per token per head
    outer_product_flops = 2 * total_seq_len * num_o_heads * head_size * head_size
    # q @ state: 2 * d^2 per token per head
    output_flops = 2 * total_seq_len * num_o_heads * head_size * head_size
    return outer_product_flops + output_flops


def gdn_bytes(
    total_seq_len: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    num_seqs: int,
    dtype: torch.dtype,
) -> int:
    """Calculate memory traffic for GDN attention (HBM reads + writes)."""
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads
    elem_size = dtype.itemsize

    # Input reads from HBM
    q_bytes = total_seq_len * num_q_heads * head_size * elem_size
    k_bytes = total_seq_len * num_k_heads * head_size * elem_size
    v_bytes = total_seq_len * num_v_heads * head_size * elem_size
    alpha_bytes = total_seq_len * num_sab_heads * elem_size
    beta_bytes = total_seq_len * num_sab_heads * elem_size

    # State: Read + Write (RMW pattern)
    state_bytes = num_seqs * num_sab_heads * head_size * head_size * 4
    state_read = state_bytes
    state_write = state_bytes

    # Output write to HBM
    o_bytes = total_seq_len * num_o_heads * head_size * elem_size

    # Total HBM traffic
    total_bytes = (
        q_bytes
        + k_bytes
        + v_bytes
        + alpha_bytes
        + beta_bytes
        + state_read
        + state_write
        + o_bytes
    )
    return total_bytes


def test_correctness_decode(
    batch_size: int,
    seq_len: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    atol: float = 0.15,
    fail_rate_threshold: float = 10.0,
    verbose: bool = True,
) -> Tuple[bool, Dict[str, float]]:
    """Test correctness of decode kernels (T=1,2,3,4 for MTP)."""
    torch.manual_seed(2025)
    N = batch_size
    T = seq_len
    scale = K**-0.5

    # Create input tensors for decode mode
    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
    a = torch.randn(N, T, HV, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, T, HV, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(N, T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(N, T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(N, T, HV, V, dtype=torch.bfloat16, device="cuda")
    indices = torch.arange(N, dtype=torch.int32, device="cuda")
    state_cutedsl = torch.randn(N, HV, K, V, dtype=torch.float32, device="cuda")
    state_triton = state_cutedsl.clone().reshape(-1).contiguous()

    # Warmup
    _ = cutedsl_gdn.cutedsl_fused_sigmoid_gating_delta_rule_update(
        A_log, dt_bias, q, k, v, a, b, state_cutedsl.clone(), indices, scale=scale
    )
    torch.cuda.synchronize()

    # Fresh state for actual test
    state_cutedsl = torch.randn(N, HV, K, V, dtype=torch.float32, device="cuda")
    state_triton = state_cutedsl.clone().reshape(-1).contiguous()
    # FlashInfer needs state in [B, HV, V, K] layout with bfloat16 dtype
    state_flashinfer = state_cutedsl.transpose(-2, -1).contiguous().to(torch.bfloat16) if FLASHINFER_AVAILABLE else None

    # Run kernels
    out_cutedsl = cutedsl_gdn.cutedsl_fused_sigmoid_gating_delta_rule_update(
        A_log, dt_bias, q, k, v, a, b, state_cutedsl, indices, scale=scale
    )
    out_triton = run_triton_decode_kernel(
        A_log, dt_bias, q, k, v, a, b, state_triton, indices, scale
    )
    if FLASHINFER_AVAILABLE:
        out_flashinfer = run_flashinfer_decode_kernel(
            A_log, dt_bias, q, k, v, a, b, state_flashinfer, indices, scale
        )

    # Check precision (compare all against CuTe DSL reference)
    abs_diff = (out_triton.float() - out_cutedsl.float()).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    fail_rate = (abs_diff > atol).float().mean().item() * 100
    has_nan = torch.isnan(out_cutedsl).any() or torch.isinf(out_cutedsl).any()

    # Check FlashInfer if available
    flashinfer_max_diff = flashinfer_mean_diff = flashinfer_fail_rate = 0.0
    if FLASHINFER_AVAILABLE:
        abs_diff_fi = (out_flashinfer.float() - out_cutedsl.float()).abs()
        flashinfer_max_diff = abs_diff_fi.max().item()
        flashinfer_mean_diff = abs_diff_fi.mean().item()
        flashinfer_fail_rate = (abs_diff_fi > atol).float().mean().item() * 100

    metrics = {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "fail_rate": fail_rate,
        "has_nan": has_nan,
        "atol": atol,
        "flashinfer_max_diff": flashinfer_max_diff,
        "flashinfer_mean_diff": flashinfer_mean_diff,
        "flashinfer_fail_rate": flashinfer_fail_rate,
    }

    passed = not has_nan and fail_rate < fail_rate_threshold
    if FLASHINFER_AVAILABLE:
        passed = passed and flashinfer_fail_rate < fail_rate_threshold

    if verbose:
        kernel_type = "SmallBatch" if N <= 64 else "LargeBatch"  # Display label only
        status = "✓ PASS" if passed else "✗ FAIL"
        msg = (
            f"  {status} | B={N:4d}, T={T:4d} ({kernel_type:10s}): "
            f"Triton: max={max_diff:.2e}, mean={mean_diff:.2e}, fail={fail_rate:.2f}%"
        )
        if FLASHINFER_AVAILABLE:
            msg += f" | FlashInfer: max={flashinfer_max_diff:.2e}, mean={flashinfer_mean_diff:.2e}, fail={flashinfer_fail_rate:.2f}%"
        print(msg)

    return passed, metrics


def test_correctness_prefill(
    batch_size: int,
    seq_len: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    atol: float = 0.15,
    fail_rate_threshold: float = 10.0,
    verbose: bool = True,
) -> Tuple[bool, Dict[str, float]]:
    """Test correctness of prefill kernel (T≥128).

    Note: For prefill, we only have Triton chunk_gated_delta_rule implementation.
    There is no CuTe DSL version, so we just verify it runs without errors.
    """
    torch.manual_seed(2025)
    N = batch_size
    T = seq_len
    scale = K**-0.5

    # Create input tensors for prefill mode
    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
    a = torch.randn(1, N * T, HV, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(1, N * T, HV, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(1, N * T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, N * T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, N * T, HV, V, dtype=torch.bfloat16, device="cuda")
    indices = torch.arange(N, dtype=torch.int32, device="cuda")
    state = torch.randn(N, HV, K, V, dtype=torch.float32, device="cuda")
    cu_seqlens = torch.arange(0, N * T + 1, T, dtype=torch.int32, device="cuda")

    # Run kernel
    try:
        out = run_triton_prefill_kernel(
            A_log, dt_bias, q, k, v, a, b, state, indices, scale, cu_seqlens
        )
        has_nan = torch.isnan(out).any() or torch.isinf(out).any()
        passed = not has_nan
    except Exception as e:
        print(f"    Error running prefill kernel: {e}")
        passed = False
        has_nan = True

    metrics = {
        "has_nan": has_nan,
    }

    if verbose:
        kernel_type = "SmallBatch" if N <= 64 else "LargeBatch"  # Display label only
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} | B={N:4d}, T={T:4d} ({kernel_type:10s}): Triton chunk_gated_delta_rule")

    return passed, metrics


def benchmark_performance_decode(
    batch_size: int,
    seq_len: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    warmup: int = 25,
    verbose: bool = True,
) -> Dict[str, float]:
    """Benchmark decode kernels (T=1,2,3,4 for MTP): CuTe DSL vs Triton."""
    torch.manual_seed(2025)
    N = batch_size
    T = seq_len
    scale = K**-0.5

    # Create input tensors
    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(N, T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(N, T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(N, T, HV, V, dtype=torch.bfloat16, device="cuda")
    a = torch.randn(N, T, HV, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, T, HV, dtype=torch.bfloat16, device="cuda")
    indices = torch.arange(N, dtype=torch.int32, device="cuda")
    state_cutedsl = torch.randn(N, HV, K, V, dtype=torch.float32, device="cuda")
    state_triton = state_cutedsl.reshape(-1).contiguous()

    # Compile CuTe DSL kernel first
    _ = cutedsl_gdn.cutedsl_fused_sigmoid_gating_delta_rule_update(
        A_log, dt_bias, q, k, v, a, b, state_cutedsl.clone(), indices, scale=scale
    )
    torch.cuda.synchronize()

    # Reset states
    state_cutedsl = torch.randn(N, HV, K, V, dtype=torch.float32, device="cuda")
    state_triton = state_cutedsl.reshape(-1).contiguous()
    # FlashInfer needs state in [B, HV, V, K] layout with bfloat16 dtype - prepare once
    state_flashinfer = state_cutedsl.transpose(-2, -1).contiguous().to(torch.bfloat16) if FLASHINFER_AVAILABLE else None

    # Warmup FlashInfer if available
    if FLASHINFER_AVAILABLE:
        _ = run_flashinfer_decode_kernel(
            A_log, dt_bias, q, k, v, a, b, state_flashinfer.clone(), indices, scale
        )
        torch.cuda.synchronize()
        # Recreate state in FlashInfer format for benchmark
        state_flashinfer = state_cutedsl.transpose(-2, -1).contiguous().to(torch.bfloat16)

    # Define benchmark functions (reuse state to avoid clone overhead in timing loop)
    def bench_cutedsl():
        return cutedsl_gdn.cutedsl_fused_sigmoid_gating_delta_rule_update(
            A_log, dt_bias, q, k, v, a, b, state_cutedsl, indices, scale=scale
        )

    def bench_triton():
        return run_triton_decode_kernel(
            A_log, dt_bias, q, k, v, a, b, state_triton, indices, scale
        )

    def bench_flashinfer():
        return run_flashinfer_decode_kernel(
            A_log, dt_bias, q, k, v, a, b, state_flashinfer, indices, scale
        )

    # Benchmark
    cutedsl_time_ms = triton.testing.do_bench(
        bench_cutedsl, warmup=warmup, return_mode="median"
    )
    triton_time_ms = triton.testing.do_bench(
        bench_triton, warmup=warmup, return_mode="median"
    )
    if FLASHINFER_AVAILABLE:
        flashinfer_time_ms = triton.testing.do_bench(
            bench_flashinfer, warmup=warmup, return_mode="median"
        )

    # Convert to microseconds
    cutedsl_time_us = cutedsl_time_ms * 1000
    triton_time_us = triton_time_ms * 1000
    flashinfer_time_us = flashinfer_time_ms * 1000 if FLASHINFER_AVAILABLE else 0
    speedup = triton_time_us / cutedsl_time_us
    speedup_flashinfer = triton_time_us / flashinfer_time_us if FLASHINFER_AVAILABLE else 0

    # Calculate TFLOPS and bandwidth (disabled for now)
    # total_seq_len = N * T
    # total_flops = gdn_flops(total_seq_len, H, H, HV, K, N)
    # total_bytes = gdn_bytes(total_seq_len, H, H, HV, K, N, torch.bfloat16)
    # cutedsl_tflops = total_flops / cutedsl_time_ms / 1e9
    # triton_tflops = total_flops / triton_time_ms / 1e9
    # cutedsl_bandwidth = total_bytes / cutedsl_time_ms / 1e9
    # triton_bandwidth = total_bytes / triton_time_ms / 1e9

    if verbose:
        kernel_type = "SmallBatch" if N <= 64 else "LargeBatch"  # Display label only
        msg = (
            f"  B={N:4d}, T={T:4d} ({kernel_type:10s}): "
            f"Triton={triton_time_us:8.2f}μs, "
            f"CuTeDSL={cutedsl_time_us:8.2f}μs, "
            f"speedup={speedup:.2f}x"
        )
        if FLASHINFER_AVAILABLE:
            msg += f", FlashInfer={flashinfer_time_us:8.2f}μs, speedup={speedup_flashinfer:.2f}x"
        print(msg)

    result = {
        "triton_time_us": triton_time_us,
        "cutedsl_time_us": cutedsl_time_us,
        "speedup": speedup,
    }
    if FLASHINFER_AVAILABLE:
        result["flashinfer_time_us"] = flashinfer_time_us
        result["speedup_flashinfer"] = speedup_flashinfer

    return result


def benchmark_performance_prefill(
    batch_size: int,
    seq_len: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    warmup: int = 25,
    verbose: bool = True,
) -> Dict[str, float]:
    """Benchmark prefill kernel (T≥128): Triton chunk_gated_delta_rule only."""
    torch.manual_seed(2025)
    N = batch_size
    T = seq_len
    scale = K**-0.5

    # Create input tensors - varlen format for chunk_gated_delta_rule
    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(1, N * T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, N * T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, N * T, HV, V, dtype=torch.bfloat16, device="cuda")
    a = torch.randn(1, N * T, HV, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(1, N * T, HV, dtype=torch.bfloat16, device="cuda")
    indices = torch.arange(N, dtype=torch.int32, device="cuda")
    state = torch.randn(N, HV, K, V, dtype=torch.float32, device="cuda")
    cu_seqlens = torch.arange(0, N * T + 1, T, dtype=torch.int32, device="cuda")

    # Warmup
    _ = run_triton_prefill_kernel(
        A_log, dt_bias, q, k, v, a, b, state.clone(), indices, scale, cu_seqlens
    )
    torch.cuda.synchronize()

    # Benchmark
    def bench_triton():
        s = state.clone()
        return run_triton_prefill_kernel(
            A_log, dt_bias, q, k, v, a, b, s, indices, scale, cu_seqlens
        )

    triton_time_ms = triton.testing.do_bench(
        bench_triton, warmup=warmup, return_mode="median"
    )

    triton_time_us = triton_time_ms * 1000

    # Calculate TFLOPS and bandwidth (disabled for now)
    # total_seq_len = N * T
    # total_flops = gdn_flops(total_seq_len, H, H, HV, K, N)
    # total_bytes = gdn_bytes(total_seq_len, H, H, HV, K, N, torch.bfloat16)
    # triton_tflops = total_flops / triton_time_ms / 1e9
    # triton_bandwidth = total_bytes / triton_time_ms / 1e9

    if verbose:
        kernel_type = "SmallBatch" if N <= 64 else "LargeBatch"  # Display label only
        print(
            f"  B={N:4d}, T={T:4d} ({kernel_type:10s}): "
            f"Triton={triton_time_us:8.2f}μs"
        )

    return {
        "triton_time_us": triton_time_us,
        # "triton_tflops": triton_tflops,
        # "triton_bandwidth_gbs": triton_bandwidth,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GDN kernels for decode and prefill"
    )
    parser.add_argument(
        "--decode",
        action="store_true",
        help="Benchmark decode kernels (T=1,2,3,4 for MTP, CuTe DSL vs Triton). Default: prefill mode.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,32,64,128,256,512",
        help="Comma-separated list of batch sizes",
    )
    parser.add_argument(
        "--decode-seq-lengths",
        type=str,
        default="1,2,3,4",
        help="Comma-separated list of sequence lengths for decode mode (for MTP testing)",
    )
    parser.add_argument(
        "--seq-lengths",
        type=str,
        default="128,1024",
        help="Comma-separated list of sequence lengths (prefill only, ignored for decode)",
    )
    parser.add_argument(
        "--qk-heads", type=int, default=16, help="Number of query/key heads (H)"
    )
    parser.add_argument(
        "--v-heads", type=int, default=64, help="Number of value heads (HV)"
    )
    parser.add_argument(
        "--head-dim", type=int, default=128, help="Head dimension (K and V)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="Number of warmup iterations (default: 25)",
    )
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Skip correctness testing",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip performance benchmarking",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.15,
        help="Absolute tolerance for correctness (default: 0.15)",
    )
    parser.add_argument(
        "--fail-rate-threshold",
        type=float,
        default=10.0,
        help="Max fail rate percentage for correctness (default: 10.0)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output CSV file for results"
    )

    args = parser.parse_args()

    if not TRITON_AVAILABLE:
        print("Error: Triton kernels are required")
        sys.exit(1)

    if args.decode and not CUTEDSL_AVAILABLE:
        print("Error: CuTe DSL not available. Decode mode requires both CuTe DSL and Triton.")
        sys.exit(1)

    if args.decode and FLASHINFER_AVAILABLE:
        print("Info: FlashInfer CuTe DSL available - will compare 3 kernels (Triton, CuTe DSL SGLang, FlashInfer)")
    elif args.decode:
        print("Info: FlashInfer CuTe DSL not available - comparing 2 kernels (Triton, CuTe DSL SGLang)")

    # Parse input ranges
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lengths = [int(x) for x in args.seq_lengths.split(",")]
    decode_seq_lengths = [int(x) for x in args.decode_seq_lengths.split(",")]

    # For prefill mode, filter out B >= 512 to avoid OOM
    if not args.decode:
        batch_sizes = [b for b in batch_sizes if b < 512]

    H = args.qk_heads
    HV = args.v_heads
    K = args.head_dim
    V = args.head_dim

    # Print configuration
    print("=" * 80)
    if args.decode:
        print("GDN Decode Kernel Benchmark (T=1,2,3,4 for MTP)")
        print("  Kernels: CuTe DSL (GB200) vs Triton (all GPUs)")
    else:
        print("GDN Prefill Kernel Benchmark (T≥128)")
        print("  Kernel: Triton chunk_gated_delta_rule")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  QK Heads (H): {H}")
    print(f"  V Heads (HV): {HV}")
    print(f"  Head Dim (K, V): {K}")
    print(f"  Batch Sizes (N): {batch_sizes}")
    if not args.decode:
        print(f"  Seq Lengths (T): {seq_lengths}")
    else:
        print(f"  Seq Lengths (T): {decode_seq_lengths} (decode/MTP)")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    if not args.skip_correctness:
        print(f"  Correctness: atol={args.atol}, fail_rate_threshold={args.fail_rate_threshold}%")
    print()

    results = []

    # Correctness testing
    if not args.skip_correctness:
        print("-" * 80)
        print("CORRECTNESS TESTING")
        print("-" * 80)
        all_passed = True

        if args.decode:
            # Decode mode: T=1,2,3,4 for MTP
            for T in decode_seq_lengths:
                for N in batch_sizes:
                    passed, metrics = test_correctness_decode(
                        N, T, H, HV, K, V,
                        atol=args.atol,
                        fail_rate_threshold=args.fail_rate_threshold,
                        verbose=True,
                    )
                    all_passed = all_passed and passed
                    results.append({
                        "batch_size": N,
                        "seq_len": T,
                        "mode": "decode",
                        "test": "correctness",
                        **metrics,
                    })
        else:
            # Prefill mode: T≥128
            for T in seq_lengths:
                for N in batch_sizes:
                    passed, metrics = test_correctness_prefill(
                        N, T, H, HV, K, V,
                        atol=args.atol,
                        fail_rate_threshold=args.fail_rate_threshold,
                        verbose=True,
                    )
                    all_passed = all_passed and passed
                    results.append({
                        "batch_size": N,
                        "seq_len": T,
                        "mode": "prefill",
                        "test": "correctness",
                        **metrics,
                    })

        print()
        if all_passed:
            print("✓ All correctness tests PASSED")
        else:
            print("✗ Some correctness tests FAILED")
            sys.exit(1)
        print()

    # Performance benchmarking
    if not args.skip_benchmark:
        print("-" * 80)
        print("PERFORMANCE BENCHMARKING")
        print("-" * 80)

        if args.decode:
            # Decode mode: T=1,2,3,4 for MTP
            for T in decode_seq_lengths:
                for N in batch_sizes:
                    try:
                        perf_metrics = benchmark_performance_decode(
                            N, T, H, HV, K, V,
                            warmup=args.warmup,
                            verbose=True,
                        )
                        results.append({
                            "batch_size": N,
                            "seq_len": T,
                            "mode": "decode",
                            "test": "performance",
                            **perf_metrics,
                        })
                    except Exception as e:
                        print(f"  Error benchmarking B={N}, T={T}: {e}")
        else:
            # Prefill mode: T≥128
            for T in seq_lengths:
                for N in batch_sizes:
                    try:
                        perf_metrics = benchmark_performance_prefill(
                            N, T, H, HV, K, V,
                            warmup=args.warmup,
                            verbose=True,
                        )
                        results.append({
                            "batch_size": N,
                            "seq_len": T,
                            "mode": "prefill",
                            "test": "performance",
                            **perf_metrics,
                        })
                    except Exception as e:
                        print(f"  Error benchmarking B={N}, T={T}: {e}")
        print()

    # Save results to CSV if requested
    if args.output:
        import csv

        print(f"Saving results to {args.output}")
        with open(args.output, "w", newline="") as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        print(f"Results saved to {args.output}")

    print("=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
