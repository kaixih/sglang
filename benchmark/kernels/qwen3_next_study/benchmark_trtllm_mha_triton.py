"""
Benchmark comparing TRTLLM MHA vs Triton attention backends for Qwen3-Next on SM100 (Blackwell).

The default attention backend for Qwen3-Next on SM100 is "triton".
This benchmark compares it against "trtllm_mha" which can be enabled via --attention-backend.

Shapes from Qwen3-Next (GQA - Grouped Query Attention):
- Total Q heads: 32
- Total K/V heads: 2
- Head dim: 256

Tensor Parallelism splits heads across GPUs:
- TP=1: Q=32, KV=2, ratio=16:1
- TP=2: Q=16, KV=1, ratio=16:1
- TP=4: Q=8,  KV=1 (replicated), ratio=8:1
- TP=8: Q=4,  KV=1 (replicated), ratio=4:1

Usage:
    python benchmark_trtllm_mha_triton.py --tp_size 1  # Default, no TP
    python benchmark_trtllm_mha_triton.py --tp_size 4  # 8:1 GQA ratio
"""

import argparse
import itertools
from typing import Tuple

import torch
import triton

from sglang.srt.layers.attention.triton_ops.decode_attention import decode_attention_fwd
from sglang.srt.layers.attention.triton_ops.extend_attention import extend_attention_fwd
from sglang.srt.utils import is_flashinfer_available

if is_flashinfer_available():
    import flashinfer


# Qwen3-Next GQA config (total, before TP split)
TOTAL_Q_HEADS = 32
TOTAL_KV_HEADS = 2
HEAD_DIM = 256

# These will be set based on --tp_size
NUM_Q_HEADS = TOTAL_Q_HEADS
NUM_KV_HEADS = TOTAL_KV_HEADS

# Dtype will be set based on --dtype
DTYPE = torch.bfloat16
DTYPE_STR = "bf16"


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch dtype."""
    if dtype_str == "bf16":
        return torch.bfloat16
    elif dtype_str == "fp8":
        return torch.float8_e4m3fn
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")


def set_dtype_config(dtype_str: str):
    """Set global dtype based on string."""
    global DTYPE, DTYPE_STR
    DTYPE = get_dtype(dtype_str)
    DTYPE_STR = dtype_str
    print(f"Dtype: {DTYPE_STR} ({DTYPE})")


def rand_tensor(*shape, dtype, device="cuda"):
    """Create random tensor, handling FP8 which doesn't support torch.randn directly."""
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # Generate in bfloat16 then convert to FP8 (uses less memory than float32)
        return torch.randn(*shape, dtype=torch.bfloat16, device=device).to(dtype)
    return torch.randn(*shape, dtype=dtype, device=device)


def zeros_tensor(*shape, dtype, device="cuda"):
    """Create zeros tensor, handling FP8."""
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return torch.zeros(*shape, dtype=torch.bfloat16, device=device).to(dtype)
    return torch.zeros(*shape, dtype=dtype, device=device)


def empty_tensor(*shape, dtype, device="cuda"):
    """Create empty tensor, handling FP8."""
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return torch.empty(*shape, dtype=torch.bfloat16, device=device).to(dtype)
    return torch.empty(*shape, dtype=dtype, device=device)


def get_heads_per_gpu(tp_size: int):
    """
    Get number of Q and KV heads per GPU for a given TP size.

    For GQA with TP:
    - Q heads are always divided by TP
    - KV heads are divided if TP <= num_kv_heads, otherwise replicated (min 1)

    Qwen3-Next (Q=32, KV=2):
    - TP=1: Q=32, KV=2, ratio=16:1
    - TP=2: Q=16, KV=1, ratio=16:1
    - TP=4: Q=8,  KV=1 (replicated), ratio=8:1
    - TP=8: Q=4,  KV=1 (replicated), ratio=4:1
    """
    num_q_heads = TOTAL_Q_HEADS // tp_size
    num_kv_heads = max(1, TOTAL_KV_HEADS // tp_size)
    return num_q_heads, num_kv_heads


def set_tp_config(tp_size: int):
    """Set global head counts based on TP size."""
    global NUM_Q_HEADS, NUM_KV_HEADS
    NUM_Q_HEADS, NUM_KV_HEADS = get_heads_per_gpu(tp_size)
    gqa_ratio = NUM_Q_HEADS // NUM_KV_HEADS
    print(f"TP={tp_size}: Q={NUM_Q_HEADS}, KV={NUM_KV_HEADS}, GQA ratio={gqa_ratio}:1")


# ============================================================================
# Prefill Attention Kernels
# ============================================================================


def create_prefill_triton_inputs(batch_size: int, seq_len: int, dtype: torch.dtype):
    """Create all inputs for Triton prefill attention."""
    total_tokens = batch_size * seq_len
    q = rand_tensor(total_tokens, NUM_Q_HEADS, HEAD_DIM, dtype=dtype)
    k = rand_tensor(total_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=dtype)
    v = rand_tensor(total_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=dtype)
    o = empty_tensor(total_tokens, NUM_Q_HEADS, HEAD_DIM, dtype=dtype)
    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * seq_len
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
    kv_indices = torch.empty(0, dtype=torch.int64, device="cuda")
    sm_scale = 1.0 / (HEAD_DIM**0.5)
    return q, k, v, o, qo_indptr, kv_indptr, kv_indices, sm_scale


def prefill_triton(q, k, v, o, qo_indptr, kv_indptr, kv_indices, sm_scale, seq_len):
    """Triton prefill attention (kernel call only)."""
    extend_attention_fwd(q, k, v, o, k, v, qo_indptr, kv_indptr, kv_indices,
                         None, True, None, seq_len, sm_scale)
    return o


def create_prefill_trtllm_inputs(batch_size: int, seq_len: int, dtype: torch.dtype, page_size: int = 64):
    """Create all inputs for TRTLLM prefill attention."""
    total_tokens = batch_size * seq_len
    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    # Total pages must match page_table indices (batch_size * num_pages_per_seq)
    num_pages = batch_size * num_pages_per_seq

    q = rand_tensor(total_tokens, NUM_Q_HEADS, HEAD_DIM, dtype=dtype)
    k_cache = rand_tensor(num_pages, NUM_KV_HEADS, page_size, HEAD_DIM, dtype=dtype)
    v_cache = rand_tensor(num_pages, NUM_KV_HEADS, page_size, HEAD_DIM, dtype=dtype)
    workspace = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * seq_len
    page_table = torch.arange(0, batch_size * num_pages_per_seq, dtype=torch.int32, device="cuda").view(batch_size, -1)
    bmm1_scale = 1.0 / (HEAD_DIM**0.5)
    return q, k_cache, v_cache, workspace, seq_lens, cu_seqlens, page_table, bmm1_scale


def prefill_trtllm(q, k_cache, v_cache, workspace, seq_lens, cu_seqlens, page_table, bmm1_scale, batch_size, seq_len):
    """TRTLLM MHA prefill attention (kernel call only)."""
    o = flashinfer.prefill.trtllm_batch_context_with_kv_cache(
        query=q, kv_cache=(k_cache, v_cache), workspace_buffer=workspace,
        block_tables=page_table, seq_lens=seq_lens, max_q_len=seq_len, max_kv_len=seq_len,
        bmm1_scale=bmm1_scale, bmm2_scale=1.0, batch_size=batch_size,
        cum_seq_lens_q=cu_seqlens, cum_seq_lens_kv=cu_seqlens,
        window_left=-1, sinks=None, out_dtype=q.dtype,
    )
    return o


# ============================================================================
# Decode Attention Kernels
# ============================================================================


def create_decode_triton_inputs(batch_size: int, kv_len: int, dtype: torch.dtype, num_kv_splits: int = 8):
    """Create all inputs for Triton decode attention."""
    total_kv = batch_size * kv_len
    q = rand_tensor(batch_size, NUM_Q_HEADS, HEAD_DIM, dtype=dtype)
    k = rand_tensor(total_kv, NUM_KV_HEADS, HEAD_DIM, dtype=dtype)
    v = rand_tensor(total_kv, NUM_KV_HEADS, HEAD_DIM, dtype=dtype)
    o = empty_tensor(batch_size, NUM_Q_HEADS, HEAD_DIM, dtype=dtype)
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * kv_len
    kv_indices = torch.arange(0, total_kv, dtype=torch.int64, device="cuda")
    sm_scale = 1.0 / (HEAD_DIM**0.5)
    attn_logits = torch.empty((batch_size, NUM_Q_HEADS, num_kv_splits, HEAD_DIM), dtype=torch.float32, device="cuda")
    attn_lse = torch.empty((batch_size, NUM_Q_HEADS, num_kv_splits), dtype=torch.float32, device="cuda")
    num_splits = torch.full((batch_size,), num_kv_splits, dtype=torch.int32, device="cuda")
    return q, k, v, o, kv_indptr, kv_indices, attn_logits, attn_lse, num_splits, sm_scale


def decode_triton(q, k, v, o, kv_indptr, kv_indices, attn_logits, attn_lse, num_splits, num_kv_splits, sm_scale):
    """Triton decode attention (kernel call only)."""
    decode_attention_fwd(q, k, v, o, kv_indptr, kv_indices,
                         attn_logits, attn_lse, num_splits, num_kv_splits, sm_scale)
    return o


def create_decode_trtllm_inputs(batch_size: int, kv_len: int, dtype: torch.dtype, page_size: int = 64):
    """Create all inputs for TRTLLM decode attention."""
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    # Total pages must match page_table indices (batch_size * num_pages_per_seq)
    num_pages = batch_size * num_pages_per_seq

    q = rand_tensor(batch_size, NUM_Q_HEADS, HEAD_DIM, dtype=dtype)
    k_cache = rand_tensor(num_pages, NUM_KV_HEADS, page_size, HEAD_DIM, dtype=dtype)
    v_cache = rand_tensor(num_pages, NUM_KV_HEADS, page_size, HEAD_DIM, dtype=dtype)
    workspace = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    seq_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device="cuda")
    page_table = torch.arange(0, batch_size * num_pages_per_seq, dtype=torch.int32, device="cuda").view(batch_size, -1)
    bmm1_scale = 1.0 / (HEAD_DIM**0.5)
    return q, k_cache, v_cache, workspace, seq_lens, page_table, bmm1_scale


def decode_trtllm(q, k_cache, v_cache, workspace, seq_lens, page_table, bmm1_scale, kv_len):
    """TRTLLM MHA decode attention (kernel call only)."""
    o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        query=q, kv_cache=(k_cache, v_cache), workspace_buffer=workspace,
        block_tables=page_table, seq_lens=seq_lens, max_seq_len=kv_len,
        bmm1_scale=bmm1_scale, bmm2_scale=1.0, window_left=-1, sinks=None, out_dtype=q.dtype,
    )
    return o


# ============================================================================
# Correctness Tests
# ============================================================================


def calculate_diff(batch_size: int, seq_len: int, mode: str = "prefill"):
    """Calculate difference between Triton and TRTLLM outputs using SAME data."""
    dtype = DTYPE
    page_size = 64
    num_kv_splits = 8

    print(f"\nShape batch={batch_size}, seq_len={seq_len}, mode={mode}, dtype={DTYPE_STR}:")

    torch.manual_seed(42)

    if mode == "prefill":
        total_tokens = batch_size * seq_len
        num_pages_per_seq = (seq_len + page_size - 1) // page_size
        num_pages = batch_size * num_pages_per_seq  # Must match page_table indices
        padded_tokens = num_pages * page_size

        # Generate shared Q, K, V data
        q = rand_tensor(total_tokens, NUM_Q_HEADS, HEAD_DIM, dtype=dtype)
        k_flat = rand_tensor(total_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=dtype)
        v_flat = rand_tensor(total_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=dtype)

        # Triton inputs
        o_triton = empty_tensor(total_tokens, NUM_Q_HEADS, HEAD_DIM, dtype=dtype)
        qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * seq_len
        kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
        kv_indices = torch.empty(0, dtype=torch.int64, device="cuda")
        sm_scale = 1.0 / (HEAD_DIM**0.5)

        # TRTLLM inputs - reshape to paged format
        k_padded = zeros_tensor(padded_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=dtype)
        v_padded = zeros_tensor(padded_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=dtype)
        k_padded[:total_tokens] = k_flat
        v_padded[:total_tokens] = v_flat
        k_trtllm = k_padded.view(num_pages, page_size, NUM_KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3).contiguous()
        v_trtllm = v_padded.view(num_pages, page_size, NUM_KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3).contiguous()
        workspace = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        seq_lens_t = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
        cu_seqlens = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * seq_len
        page_table = torch.arange(0, batch_size * num_pages_per_seq, dtype=torch.int32, device="cuda").view(batch_size, -1)
        bmm1_scale = 1.0 / (HEAD_DIM**0.5)

        out_triton = prefill_triton(q, k_flat, v_flat, o_triton, qo_indptr, kv_indptr, kv_indices, sm_scale, seq_len)
        out_trtllm = prefill_trtllm(q, k_trtllm, v_trtllm, workspace, seq_lens_t, cu_seqlens, page_table, bmm1_scale, batch_size, seq_len)

    else:  # decode
        kv_len = seq_len
        total_kv = batch_size * kv_len
        num_pages_per_seq = (kv_len + page_size - 1) // page_size
        num_pages = batch_size * num_pages_per_seq  # Must match page_table indices
        padded_tokens = num_pages * page_size

        # Generate shared Q, K, V data
        q = rand_tensor(batch_size, NUM_Q_HEADS, HEAD_DIM, dtype=dtype)
        k_flat = rand_tensor(total_kv, NUM_KV_HEADS, HEAD_DIM, dtype=dtype)
        v_flat = rand_tensor(total_kv, NUM_KV_HEADS, HEAD_DIM, dtype=dtype)

        # Triton inputs
        o_triton = empty_tensor(batch_size, NUM_Q_HEADS, HEAD_DIM, dtype=dtype)
        kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * kv_len
        kv_indices = torch.arange(0, total_kv, dtype=torch.int64, device="cuda")
        sm_scale = 1.0 / (HEAD_DIM**0.5)
        attn_logits = torch.empty((batch_size, NUM_Q_HEADS, num_kv_splits, HEAD_DIM), dtype=torch.float32, device="cuda")
        attn_lse = torch.empty((batch_size, NUM_Q_HEADS, num_kv_splits), dtype=torch.float32, device="cuda")
        num_splits = torch.full((batch_size,), num_kv_splits, dtype=torch.int32, device="cuda")

        # TRTLLM inputs - reshape to paged format
        k_padded = zeros_tensor(padded_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=dtype)
        v_padded = zeros_tensor(padded_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=dtype)
        k_padded[:total_kv] = k_flat
        v_padded[:total_kv] = v_flat
        k_trtllm = k_padded.view(num_pages, page_size, NUM_KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3).contiguous()
        v_trtllm = v_padded.view(num_pages, page_size, NUM_KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3).contiguous()
        workspace = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        seq_lens_t = torch.full((batch_size,), kv_len, dtype=torch.int32, device="cuda")
        page_table = torch.arange(0, batch_size * num_pages_per_seq, dtype=torch.int32, device="cuda").view(batch_size, -1)
        bmm1_scale = 1.0 / (HEAD_DIM**0.5)

        out_triton = decode_triton(q, k_flat, v_flat, o_triton, kv_indptr, kv_indices, attn_logits, attn_lse, num_splits, num_kv_splits, sm_scale)
        out_trtllm = decode_trtllm(q, k_trtllm, v_trtllm, workspace, seq_lens_t, page_table, bmm1_scale, kv_len)

    # Compare outputs (convert to float32 for FP8 comparison)
    out_triton_f32 = out_triton.float()
    out_trtllm_f32 = out_trtllm.float()
    diff = torch.abs(out_triton_f32.view(-1) - out_trtllm_f32.view(-1))
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Triton output: {out_triton_f32.view(-1)[:5]}")
    print(f"TRTLLM output: {out_trtllm_f32.view(-1)[:5]}")
    print(f"Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

    # FP8 has much lower precision (3 mantissa bits) than BF16 (7 mantissa bits)
    # Use higher tolerance for FP8
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        atol, rtol = 0.5, 0.1  # FP8: higher tolerance
    else:
        atol, rtol = 1e-2, 1e-2  # BF16: standard tolerance

    if torch.allclose(out_triton_f32, out_trtllm_f32, atol=atol, rtol=rtol):
        print("✅ Outputs match")
    else:
        print("❌ Outputs differ")


# ============================================================================
# TFLOPS Calculation
# ============================================================================


def compute_attention_flops(batch_size: int, seq_len: int, num_heads: int, head_dim: int, mode: str) -> int:
    """
    Compute FLOPs for attention operation.

    For attention: Q @ K^T and Attn @ V each require 2 * batch * heads * seq * seq * head_dim FLOPs
    Total: 4 * batch * heads * seq_len * seq_len * head_dim (for prefill)

    For decode (single query token): 4 * batch * heads * 1 * kv_len * head_dim
    """
    if mode == "prefill":
        # Q @ K^T: batch * heads * seq * seq * head_dim * 2 (multiply-add)
        # Attn @ V: batch * heads * seq * seq * head_dim * 2
        flops = 4 * batch_size * num_heads * seq_len * seq_len * head_dim
    else:  # decode
        # Single query attending to kv_len tokens
        kv_len = seq_len
        flops = 4 * batch_size * num_heads * kv_len * head_dim
    return flops


def compute_tflops(flops: int, time_us: float) -> float:
    """Convert FLOPs and time (in microseconds) to TFLOPS."""
    time_seconds = time_us * 1e-6
    return flops / time_seconds / 1e12


def compute_decode_memory_bytes(batch_size: int, kv_len: int, num_q_heads: int, num_kv_heads: int, head_dim: int, dtype) -> int:
    """
    Compute memory bytes accessed for decode attention.
    
    Decode is memory-bound. Memory accessed:
    - Q: (batch, num_q_heads, head_dim) - read
    - K: (batch * kv_len, num_kv_heads, head_dim) - read  
    - V: (batch * kv_len, num_kv_heads, head_dim) - read
    - O: (batch, num_q_heads, head_dim) - write
    """
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        bytes_per_elem = 1
    elif dtype == torch.bfloat16:
        bytes_per_elem = 2
    else:
        bytes_per_elem = 2  # Default to 2 bytes
    
    q_bytes = batch_size * num_q_heads * head_dim * bytes_per_elem
    k_bytes = batch_size * kv_len * num_kv_heads * head_dim * bytes_per_elem
    v_bytes = batch_size * kv_len * num_kv_heads * head_dim * bytes_per_elem
    o_bytes = batch_size * num_q_heads * head_dim * bytes_per_elem
    
    return q_bytes + k_bytes + v_bytes + o_bytes


def compute_bandwidth_gbps(memory_bytes: int, time_us: float) -> float:
    """Convert memory bytes and time (in microseconds) to GB/s."""
    time_seconds = time_us * 1e-6
    return memory_bytes / time_seconds / 1e9


# ============================================================================
# Benchmark
# ============================================================================


def create_benchmark_configs(tp_size: int):
    """Create benchmark configurations with TP info."""
    batch_sizes = [1, 128, 1024]
    seq_lens = [256, 512, 1024, 2048]
    num_q_heads, num_kv_heads = get_heads_per_gpu(tp_size)
    configs = []
    for bs, sl in itertools.product(batch_sizes, seq_lens):
        # Skip extreme configs to avoid OOM (batch*seq > 1M tokens)
        # 1024*2048 = 2M tokens would need ~32GB just for Q tensor
        if bs * sl > 1024 * 1024:
            continue
        # (tp_size, num_q_heads, num_kv_heads, batch_size, seq_len, dtype_str)
        configs.append((tp_size, num_q_heads, num_kv_heads, bs, sl, DTYPE_STR))
    return configs


def get_benchmark(tp_size: int):
    """Get benchmark function using triton.testing."""
    all_configs = create_benchmark_configs(tp_size)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["tp_size", "num_q_heads", "num_kv_heads", "batch_size", "seq_len", "dtype"],
            x_vals=[list(config) for config in all_configs],
            line_arg="provider",
            line_vals=["triton_prefill", "trtllm_prefill", "triton_decode", "trtllm_decode"],
            line_names=["Triton Prefill", "TRTLLM Prefill", "Triton Decode", "TRTLLM Decode"],
            styles=[("blue", "-"), ("blue", "--"), ("red", "-"), ("red", "--")],
            ylabel="us",
            plot_name=f"trtllm-mha-vs-triton-qwen3next-tp{tp_size}-{DTYPE_STR}",
            args={},
        )
    )
    def benchmark(tp_size, num_q_heads, num_kv_heads, batch_size, seq_len, dtype, provider):
        # tp_size, num_q_heads, num_kv_heads, dtype are for display; actual values use globals
        dtype = DTYPE  # Use global dtype, not the string param
        quantiles = [0.5, 0.2, 0.8]

        # Pre-allocate all tensors outside do_bench
        if provider == "triton_prefill":
            q, k, v, o, qo_indptr, kv_indptr, kv_indices, sm_scale = \
                create_prefill_triton_inputs(batch_size, seq_len, dtype)
            torch.cuda.synchronize()
            fn = lambda: prefill_triton(q, k, v, o, qo_indptr, kv_indptr, kv_indices, sm_scale, seq_len)

        elif provider == "trtllm_prefill":
            q, k_cache, v_cache, workspace, seq_lens, cu_seqlens, page_table, bmm1_scale = \
                create_prefill_trtllm_inputs(batch_size, seq_len, dtype)
            torch.cuda.synchronize()
            fn = lambda: prefill_trtllm(q, k_cache, v_cache, workspace, seq_lens, cu_seqlens, page_table, bmm1_scale, batch_size, seq_len)

        elif provider == "triton_decode":
            num_kv_splits = 8
            q, k, v, o, kv_indptr, kv_indices, attn_logits, attn_lse, num_splits, sm_scale = \
                create_decode_triton_inputs(batch_size, seq_len, dtype, num_kv_splits)
            torch.cuda.synchronize()
            fn = lambda: decode_triton(q, k, v, o, kv_indptr, kv_indices, attn_logits, attn_lse, num_splits, num_kv_splits, sm_scale)

        else:  # trtllm_decode
            q, k_cache, v_cache, workspace, seq_lens, page_table, bmm1_scale = \
                create_decode_trtllm_inputs(batch_size, seq_len, dtype)
            torch.cuda.synchronize()
            fn = lambda: decode_trtllm(q, k_cache, v_cache, workspace, seq_lens, page_table, bmm1_scale, seq_len)

        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        return ms * 1000, max_ms * 1000, min_ms * 1000  # convert to us

    return benchmark


def run_benchmark_with_tflops(tp_size: int, single_batch: int = None, single_seq: int = None):
    """Run benchmark and display results with TFLOPS for prefill and GB/s for decode."""
    if single_batch is not None and single_seq is not None:
        # Run single config
        num_q_heads, num_kv_heads = get_heads_per_gpu(tp_size)
        configs = [(tp_size, num_q_heads, num_kv_heads, single_batch, single_seq, DTYPE_STR)]
    else:
        configs = create_benchmark_configs(tp_size)
    num_q_heads, num_kv_heads = get_heads_per_gpu(tp_size)

    dtype = DTYPE
    quantiles = [0.5, 0.2, 0.8]

    # Collect results
    prefill_results = []
    decode_results = []

    for tp, nq, nkv, batch_size, seq_len, dtype_str in configs:
        for mode in ["prefill", "decode"]:
            # Create inputs and benchmark
            if mode == "prefill":
                q, k, v, o, qo_indptr, kv_indptr, kv_indices, sm_scale = \
                    create_prefill_triton_inputs(batch_size, seq_len, dtype)
                torch.cuda.synchronize()
                fn_triton = lambda: prefill_triton(q, k, v, o, qo_indptr, kv_indptr, kv_indices, sm_scale, seq_len)

                q2, k_cache, v_cache, workspace, seq_lens, cu_seqlens, page_table, bmm1_scale = \
                    create_prefill_trtllm_inputs(batch_size, seq_len, dtype)
                torch.cuda.synchronize()
                fn_trtllm = lambda: prefill_trtllm(q2, k_cache, v_cache, workspace, seq_lens, cu_seqlens, page_table, bmm1_scale, batch_size, seq_len)
            else:
                num_kv_splits = 8
                q, k, v, o, kv_indptr, kv_indices, attn_logits, attn_lse, num_splits, sm_scale = \
                    create_decode_triton_inputs(batch_size, seq_len, dtype, num_kv_splits)
                torch.cuda.synchronize()
                fn_triton = lambda: decode_triton(q, k, v, o, kv_indptr, kv_indices, attn_logits, attn_lse, num_splits, num_kv_splits, sm_scale)

                q2, k_cache, v_cache, workspace, seq_lens, page_table, bmm1_scale = \
                    create_decode_trtllm_inputs(batch_size, seq_len, dtype)
                torch.cuda.synchronize()
                fn_trtllm = lambda: decode_trtllm(q2, k_cache, v_cache, workspace, seq_lens, page_table, bmm1_scale, seq_len)

            # Benchmark
            triton_ms, _, _ = triton.testing.do_bench(fn_triton, quantiles=quantiles)
            trtllm_ms, _, _ = triton.testing.do_bench(fn_trtllm, quantiles=quantiles)

            # Convert to microseconds
            triton_us = triton_ms * 1000
            trtllm_us = trtllm_ms * 1000
            speedup = triton_us / trtllm_us

            if mode == "prefill":
                flops = compute_attention_flops(batch_size, seq_len, num_q_heads, HEAD_DIM, mode)
                triton_tflops = compute_tflops(flops, triton_us)
                trtllm_tflops = compute_tflops(flops, trtllm_us)
                prefill_results.append((batch_size, seq_len, dtype_str, triton_us, trtllm_us, 
                                        triton_tflops, trtllm_tflops, speedup))
            else:
                mem_bytes = compute_decode_memory_bytes(batch_size, seq_len, num_q_heads, num_kv_heads, HEAD_DIM, dtype)
                triton_gbps = compute_bandwidth_gbps(mem_bytes, triton_us)
                trtllm_gbps = compute_bandwidth_gbps(mem_bytes, trtllm_us)
                decode_results.append((batch_size, seq_len, dtype_str, triton_us, trtllm_us,
                                       triton_gbps, trtllm_gbps, speedup))

            # Clean up to avoid OOM
            torch.cuda.empty_cache()

    # Print Prefill results (compute-bound, show TFLOPS)
    print(f"\n{'='*130}")
    print(f"PREFILL Results - Compute Bound (TP={tp_size}, Q={num_q_heads}, KV={num_kv_heads}, HEAD_DIM={HEAD_DIM})")
    print(f"{'='*130}")
    print(f"{'Batch':>6} {'SeqLen':>7} {'Dtype':>6} {'Triton (us)':>12} {'TRTLLM (us)':>12} "
          f"{'Triton TFLOPS':>14} {'TRTLLM TFLOPS':>14} {'Speedup':>8}")
    print("-" * 130)
    for batch_size, seq_len, dtype_str, triton_us, trtllm_us, triton_tflops, trtllm_tflops, speedup in prefill_results:
        print(f"{batch_size:>6} {seq_len:>7} {dtype_str:>6} {triton_us:>12.2f} {trtllm_us:>12.2f} "
              f"{triton_tflops:>14.2f} {trtllm_tflops:>14.2f} {speedup:>7.2f}x")
    print("=" * 130)

    # Print Decode results (memory-bound, show GB/s)
    print(f"\n{'='*130}")
    print(f"DECODE Results - Memory Bound (TP={tp_size}, Q={num_q_heads}, KV={num_kv_heads}, HEAD_DIM={HEAD_DIM})")
    print(f"{'='*130}")
    print(f"{'Batch':>6} {'SeqLen':>7} {'Dtype':>6} {'Triton (us)':>12} {'TRTLLM (us)':>12} "
          f"{'Triton GB/s':>12} {'TRTLLM GB/s':>12} {'Speedup':>8}")
    print("-" * 130)
    for batch_size, seq_len, dtype_str, triton_us, trtllm_us, triton_gbps, trtllm_gbps, speedup in decode_results:
        print(f"{batch_size:>6} {seq_len:>7} {dtype_str:>6} {triton_us:>12.2f} {trtllm_us:>12.2f} "
              f"{triton_gbps:>12.2f} {trtllm_gbps:>12.2f} {speedup:>7.2f}x")
    print("=" * 130)


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tp_size",
        type=int,
        default=None,
        choices=[1, 2, 4],
        help="Tensor parallelism size. If not set, runs all of [1, 2, 4]",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./benchmark_results/",
        help="Path to save benchmark results",
    )
    parser.add_argument(
        "--run_correctness",
        action="store_true",
        default=True,
        help="Whether to run correctness test",
    )
    parser.add_argument(
        "--show_tflops",
        action="store_true",
        default=False,
        help="Show TFLOPS in addition to timing results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Run single batch size (requires --seq_len)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Run single seq_len (requires --batch_size)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["bf16", "fp8"],
        help="Attention dtype: bf16 (bfloat16) or fp8 (float8_e4m3fn). If not set, sweeps both.",
    )
    args = parser.parse_args()

    # Determine which TP sizes and dtypes to run
    tp_sizes = [args.tp_size] if args.tp_size is not None else [1, 2, 4]
    dtypes = [args.dtype] if args.dtype is not None else ["bf16", "fp8"]

    # Set random seed
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Check FlashInfer
    if not is_flashinfer_available():
        print("ERROR: FlashInfer not available")
        exit(1)

    os.makedirs(args.save_path, exist_ok=True)

    for dtype_str in dtypes:
        # Set dtype for this iteration
        set_dtype_config(dtype_str)

        for tp_size in tp_sizes:
            print(f"\n{'='*60}")
            print(f"Running with TP size = {tp_size}, dtype = {DTYPE_STR}")
            print(f"{'='*60}")

            # Configure heads based on TP size
            set_tp_config(tp_size)

            # Run correctness tests
            if args.run_correctness:
                print("\nRunning correctness tests...")
                calculate_diff(4, 512, "prefill")
                calculate_diff(4, 1024, "decode")

            # Run benchmark
            if args.show_tflops or (args.batch_size is not None and args.seq_len is not None):
                run_benchmark_with_tflops(tp_size, args.batch_size, args.seq_len)
            else:
                benchmark = get_benchmark(tp_size)
                print("\nRunning performance benchmark...")
                benchmark.run(print_data=True, save_path=args.save_path)
