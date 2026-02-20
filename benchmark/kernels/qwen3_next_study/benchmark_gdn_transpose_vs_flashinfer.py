"""Benchmark SGLang CuteDSL Transpose vs FlashInfer CuteDSL for GDN decode/MTP.

Compares the two best-performing GDN kernel implementations:
- SGLang CuteDSL Transpose (PR #17981): state [B, HV, K, V] stride[-2]=1 (K-contiguous)
- FlashInfer CuteDSL (PR #2498):        state [B, HV, V, K] float32 (K-last)

Kernel selection:
- T=1  (decode): SGLang: cutedsl_fused_recurrent_sigmoid_gated_delta_rule_update
                 FlashInfer: gated_delta_rule
- T>1  (MTP):   SGLang: cutedsl_fused_recurrent_gated_delta_rule_update
                 FlashInfer: gated_delta_rule_mtp
Both MTP kernels support intermediate state caching for speculative decoding retraction.

Note: T>1 correctness is not checked - SGLang MTP takes pre-computed g while
FlashInfer MTP takes A_log/a/dt_bias, so their outputs are not directly comparable.

Target shapes (Qwen3-Next):
- QK num_heads (H): 16
- V num_heads (HV): 64
- Head dimension (K, V): 128
"""

import argparse
import sys

import torch
import triton

try:
    import cuda.bindings.driver as cuda_driver  # noqa: F401

    from sglang.jit_kernel.cutedsl_gdn_transpose import (
        cutedsl_fused_recurrent_gated_delta_rule_update,
        cutedsl_fused_recurrent_sigmoid_gated_delta_rule_update,
    )

    SGLANG_TRANSPOSE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SGLang CuteDSL Transpose not available: {e}")
    SGLANG_TRANSPOSE_AVAILABLE = False
    cutedsl_fused_recurrent_sigmoid_gated_delta_rule_update = None
    cutedsl_fused_recurrent_gated_delta_rule_update = None

try:
    from flashinfer.gdn_kernels import gated_delta_rule as flashinfer_gated_delta_rule
    from flashinfer.gdn_decode import gated_delta_rule_mtp as flashinfer_gated_delta_rule_mtp

    FLASHINFER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: FlashInfer GDN kernels not available: {e}")
    FLASHINFER_AVAILABLE = False
    flashinfer_gated_delta_rule = None
    flashinfer_gated_delta_rule_mtp = None


# ---------------------------------------------------------------------------
# State creation helpers
# ---------------------------------------------------------------------------

def _make_decode_states(N, HV, K, V):
    """States for T=1 decode.

    SGLang: [N, HV, K, V] with stride[-2]=1 (K-contiguous), bfloat16
    FlashInfer: [N, HV, V, K] contiguous, bfloat16
    """
    base = torch.randn(N, HV, K, V, dtype=torch.float32, device="cuda")
    state_sglang = base.clone().transpose(-2, -1).contiguous().transpose(-2, -1).to(torch.bfloat16)
    state_flashinfer = base.clone().transpose(-2, -1).contiguous().to(torch.bfloat16)
    return state_sglang, state_flashinfer


def _make_mtp_states(N, T, HV, K, V):
    """States and intermediate buffers for T>1 MTP.

    SGLang:
      state: [N, HV, K, V] stride[-2]=1, float32 (FlashInfer MTP requires float32, so both use float32)
      intermediate: [N+1, T, HV, K, V] stride[-2]=1, float32
      indices: [N], cu_seqlens: [N+1]

    FlashInfer:
      state: [N, HV, V, K] contiguous, float32
      intermediate: [N, T, HV, V, K] contiguous, float32
      indices: [N]
    """
    base = torch.randn(N, HV, K, V, dtype=torch.float32, device="cuda")

    # SGLang state + intermediate
    state_sglang = base.clone().transpose(-2, -1).contiguous().transpose(-2, -1)
    inter_sglang_base = torch.zeros(N + 1, T, HV, K, V, dtype=torch.float32, device="cuda")
    inter_sglang = inter_sglang_base.transpose(-2, -1).contiguous().transpose(-2, -1)
    inter_indices_sglang = torch.arange(N, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.arange(0, (N + 1) * T, T, dtype=torch.int32, device="cuda")

    # FlashInfer state + intermediate
    state_flashinfer = base.clone().transpose(-2, -1).contiguous()  # [N, HV, V, K] float32
    inter_flashinfer = torch.zeros(N, T, HV, V, K, dtype=torch.float32, device="cuda")
    inter_indices_flashinfer = torch.arange(N, dtype=torch.int32, device="cuda")

    return (
        state_sglang, inter_sglang, inter_indices_sglang, cu_seqlens,
        state_flashinfer, inter_flashinfer, inter_indices_flashinfer,
    )


# ---------------------------------------------------------------------------
# Kernel wrappers
# ---------------------------------------------------------------------------

def run_sglang_decode(A_log, dt_bias, q, k, v, a, b, state, indices, scale):
    """SGLang T=1 decode."""
    return cutedsl_fused_recurrent_sigmoid_gated_delta_rule_update(
        A_log=A_log, a=a, dt_bias=dt_bias,
        softplus_beta=1.0, softplus_threshold=20.0,
        q=q, k=k, v=v, b=b,
        initial_state_source=state, initial_state_indices=indices,
        scale=scale, use_qk_l2norm_in_kernel=True, cu_seqlens=None,
    )


def run_sglang_mtp(q, k, v, g, b, state, indices, intermediate_state,
                   intermediate_indices, cu_seqlens, scale, T):
    """SGLang T>1 MTP with intermediate state caching."""
    return cutedsl_fused_recurrent_gated_delta_rule_update(
        q=q, k=k, v=v, g=g, beta=b,
        initial_state_source=state, initial_state_indices=indices,
        scale=scale, use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens, disable_state_update=True,
        intermediate_states_buffer=intermediate_state,
        intermediate_state_indices=intermediate_indices,
        cache_steps=T,
    )


def run_flashinfer_decode(A_log, dt_bias, q, k, v, a, b, state, indices, scale):
    """FlashInfer T=1 decode."""
    return flashinfer_gated_delta_rule(
        A_log=A_log, a=a, dt_bias=dt_bias,
        softplus_beta=1.0, softplus_threshold=20.0,
        q=q, k=k, v=v, b=b,
        initial_state_source=state, initial_state_indices=indices,
        use_qk_l2norm_in_kernel=True, scale=scale,
    )


def run_flashinfer_mtp(A_log, dt_bias, q, k, v, a, b, state, indices,
                       intermediate_state, scale):
    """FlashInfer T>1 MTP with intermediate state caching."""
    return flashinfer_gated_delta_rule_mtp(
        q=q, k=k, v=v,
        initial_state=state, initial_state_indices=indices,
        A_log=A_log, a=a, dt_bias=dt_bias, b=b,
        scale=scale, use_qk_l2norm=True,
        intermediate_states_buffer=intermediate_state,
        disable_state_update=True,
    )


# ---------------------------------------------------------------------------
# Gate precomputation helper
# ---------------------------------------------------------------------------

def _compute_g_from_params(A_log, a, dt_bias, softplus_beta=1.0, softplus_threshold=20.0):
    """Compute log-space gate g = -exp(A_log) * softplus(a + dt_bias).

    SGLang MTP kernel takes this pre-computed g.
    FlashInfer MTP kernel takes A_log/a/dt_bias and computes this internally.
    """
    x = a.float() + dt_bias.float()  # [N, T, HV] + [HV] -> [N, T, HV]
    beta_x = softplus_beta * x
    softplus_x = torch.where(
        beta_x <= softplus_threshold,
        (1.0 / softplus_beta) * torch.log1p(torch.exp(beta_x.clamp(max=softplus_threshold))),
        x,
    )
    g = -torch.exp(A_log.float()) * softplus_x  # [N, T, HV]
    return g.to(a.dtype)


# ---------------------------------------------------------------------------
# Correctness (T=1 and T>1)
# ---------------------------------------------------------------------------

def test_correctness(batch_size, H, HV, K, V, atol=0.15, fail_rate_threshold=10.0):
    torch.manual_seed(2025)
    N, T = batch_size, 1
    scale = K**-0.5

    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
    a = torch.randn(N, T, HV, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, T, HV, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(N, T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(N, T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(N, T, HV, V, dtype=torch.bfloat16, device="cuda")
    indices = torch.arange(N, dtype=torch.int32, device="cuda")

    state_sglang, state_flashinfer = _make_decode_states(N, HV, K, V)

    out_sglang = run_sglang_decode(A_log, dt_bias, q, k, v, a, b, state_sglang, indices, scale)
    out_flashinfer = run_flashinfer_decode(A_log, dt_bias, q, k, v, a, b, state_flashinfer, indices, scale)

    abs_diff = (out_sglang.float() - out_flashinfer.float()).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    fail_rate = (abs_diff > atol).float().mean().item() * 100
    passed = fail_rate < fail_rate_threshold

    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status} | T=1, B={N:4d}: max={max_diff:.2e}, mean={mean_diff:.2e}, fail={fail_rate:.2f}%")
    return passed


def test_correctness_mtp(batch_size, seq_len, H, HV, K, V, atol=0.15, fail_rate_threshold=10.0):
    """Correctness for T>1 MTP: pre-compute g from A_log/a/dt_bias to align with FlashInfer."""
    torch.manual_seed(2025)
    N, T = batch_size, seq_len
    scale = K**-0.5

    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
    a = torch.randn(N, T, HV, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, T, HV, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(N, T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(N, T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(N, T, HV, V, dtype=torch.bfloat16, device="cuda")
    indices = torch.arange(N, dtype=torch.int32, device="cuda")

    # Pre-compute g for SGLang MTP from A_log/a/dt_bias
    g = _compute_g_from_params(A_log, a, dt_bias)

    (state_sglang, inter_sglang, inter_idx_sglang, cu_seqlens,
     state_flashinfer, inter_flashinfer, inter_idx_flashinfer) = _make_mtp_states(N, T, HV, K, V)

    # SGLang needs flat [1, N*T, ...] layout
    q_sg = q.reshape(1, N * T, H, K)
    k_sg = k.reshape(1, N * T, H, K)
    v_sg = v.reshape(1, N * T, HV, V)
    g_sg = g.reshape(1, N * T, HV)
    b_sg = b.reshape(1, N * T, HV)

    out_sglang = run_sglang_mtp(
        q_sg, k_sg, v_sg, g_sg, b_sg, state_sglang, indices,
        inter_sglang, inter_idx_sglang, cu_seqlens, scale, T,
    )
    out_flashinfer, _ = run_flashinfer_mtp(
        A_log, dt_bias, q, k, v, a, b,
        state_flashinfer, inter_idx_flashinfer, inter_flashinfer, scale,
    )

    # SGLang output is [1, N*T, HV, V], reshape to [N, T, HV, V] for comparison
    out_sglang_r = out_sglang.reshape(N, T, HV, V)

    abs_diff = (out_sglang_r.float() - out_flashinfer.float()).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    fail_rate = (abs_diff > atol).float().mean().item() * 100
    passed = fail_rate < fail_rate_threshold

    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status} | T={T}, B={N:4d}: max={max_diff:.2e}, mean={mean_diff:.2e}, fail={fail_rate:.2f}%")
    return passed


# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------

def benchmark_decode(batch_size, seq_len, H, HV, K, V, warmup=25):
    torch.manual_seed(2025)
    N, T = batch_size, seq_len
    scale = K**-0.5

    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
    a = torch.randn(N, T, HV, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, T, HV, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(N, T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(N, T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(N, T, HV, V, dtype=torch.bfloat16, device="cuda")
    indices = torch.arange(N, dtype=torch.int32, device="cuda")

    if T == 1:
        state_sglang, state_flashinfer = _make_decode_states(N, HV, K, V)

        def bench_sglang():
            return run_sglang_decode(A_log, dt_bias, q, k, v, a, b, state_sglang, indices, scale)

        def bench_flashinfer():
            return run_flashinfer_decode(A_log, dt_bias, q, k, v, a, b, state_flashinfer, indices, scale)

        sglang_label = "SGLang(decode)"
    else:
        g = torch.randn(N, T, HV, dtype=torch.bfloat16, device="cuda")
        (state_sglang, inter_sglang, inter_idx_sglang, cu_seqlens,
         state_flashinfer, inter_flashinfer, inter_idx_flashinfer) = _make_mtp_states(N, T, HV, K, V)

        # SGLang MTP expects flat [1, N*T, ...] layout with cu_seqlens
        q_sg = q.reshape(1, N * T, H, K)
        k_sg = k.reshape(1, N * T, H, K)
        v_sg = v.reshape(1, N * T, HV, V)
        g_sg = g.reshape(1, N * T, HV)
        b_sg = b.reshape(1, N * T, HV)

        def bench_sglang():
            return run_sglang_mtp(
                q_sg, k_sg, v_sg, g_sg, b_sg, state_sglang, indices,
                inter_sglang, inter_idx_sglang, cu_seqlens, scale, T,
            )

        def bench_flashinfer():
            return run_flashinfer_mtp(
                A_log, dt_bias, q, k, v, a, b,
                state_flashinfer, inter_idx_flashinfer, inter_flashinfer, scale,
            )

        sglang_label = "SGLang(mtp)"

    # Warmup / compile
    bench_sglang()
    bench_flashinfer()
    torch.cuda.synchronize()

    sglang_ms = triton.testing.do_bench(bench_sglang, warmup=warmup, return_mode="median")
    flashinfer_ms = triton.testing.do_bench(bench_flashinfer, warmup=warmup, return_mode="median")

    sglang_us = sglang_ms * 1000
    flashinfer_us = flashinfer_ms * 1000
    speedup = sglang_us / flashinfer_us

    print(
        f"  B={N:4d}, T={T} ({sglang_label}): "
        f"SGLang={sglang_us:8.2f}μs, "
        f"FlashInfer={flashinfer_us:8.2f}μs, "
        f"FlashInfer speedup={speedup:.2f}x"
    )
    return {
        "batch_size": N, "seq_len": T,
        "sglang_us": sglang_us, "flashinfer_us": flashinfer_us, "speedup": speedup,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SGLang CuteDSL Transpose vs FlashInfer CuteDSL for GDN decode/MTP"
    )
    parser.add_argument("--batch-sizes", type=str, default="1,32,64,128,256,512")
    parser.add_argument("--seq-lengths", type=str, default="1,2,3,4",
                        help="T=1 decode, T>1 MTP")
    parser.add_argument("--qk-heads", type=int, default=16)
    parser.add_argument("--v-heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file")
    args = parser.parse_args()

    if not SGLANG_TRANSPOSE_AVAILABLE:
        print("Error: SGLang CuteDSL Transpose kernel not available.")
        sys.exit(1)
    if not FLASHINFER_AVAILABLE:
        print("Error: FlashInfer CuteDSL not available.")
        sys.exit(1)

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lengths = [int(x) for x in args.seq_lengths.split(",")]
    H, HV, K, V = args.qk_heads, args.v_heads, args.head_dim, args.head_dim

    print("=" * 80)
    print("GDN: SGLang CuteDSL Transpose (PR #17981) vs FlashInfer CuteDSL (PR #2498)")
    print("  T=1 SGLang:    cutedsl_fused_recurrent_sigmoid_gated_delta_rule_update")
    print("  T>1 SGLang:    cutedsl_fused_recurrent_gated_delta_rule_update (MTP)")
    print("  T=1 FlashInfer: gated_delta_rule")
    print("  T>1 FlashInfer: gated_delta_rule_mtp")
    print("  Both MTP kernels support intermediate state caching for retraction.")
    print("=" * 80)
    print(f"  QK Heads: {H}, V Heads: {HV}, Head Dim: {K}")
    print(f"  Batch sizes: {batch_sizes}, Seq lengths (T): {seq_lengths}")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print()

    results = []

    if not args.skip_correctness:
        print("-" * 80)
        print("CORRECTNESS")
        print("-" * 80)
        all_passed = True
        print("T=1 (decode):")
        for N in batch_sizes:
            passed = test_correctness(N, H, HV, K, V)
            all_passed = all_passed and passed
        print()
        mtp_lengths = [t for t in seq_lengths if t > 1]
        if mtp_lengths:
            print("T>1 (MTP, g pre-computed from A_log/a/dt_bias):")
            for T in mtp_lengths:
                for N in batch_sizes:
                    passed = test_correctness_mtp(N, T, H, HV, K, V)
                    all_passed = all_passed and passed
            print()
        if all_passed:
            print("✓ All correctness tests PASSED")
        else:
            print("✗ Some correctness tests FAILED")
            sys.exit(1)
        print()

    if not args.skip_benchmark:
        print("-" * 80)
        print("PERFORMANCE")
        print("-" * 80)
        for T in seq_lengths:
            for N in batch_sizes:
                try:
                    r = benchmark_decode(N, T, H, HV, K, V, warmup=args.warmup)
                    results.append(r)
                except Exception as e:
                    print(f"  Error B={N}, T={T}: {e}")
        print()

    if args.output and results:
        import csv
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {args.output}")

    print("=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
