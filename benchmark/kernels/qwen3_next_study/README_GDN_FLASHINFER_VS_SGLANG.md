# GDN Kernel Benchmark: SGLang CuteDSL Transpose vs FlashInfer CuteDSL

Compares two high-performance GDN (Gated Delta Net) kernel implementations for Qwen3.5.

| | SGLang | FlashInfer |
|---|---|---|
| PR | [#17981](https://github.com/sgl-project/sglang/pull/17981) | [#2498](https://github.com/flashinfer-ai/flashinfer/pull/2498) |
| T=1 kernel | `cutedsl_fused_recurrent_sigmoid_gated_delta_rule_update` | `gated_delta_rule` (from `flashinfer.gdn_kernels`) |
| T>1 kernel | `cutedsl_fused_recurrent_gated_delta_rule_update` | `gated_delta_rule_mtp` (from `flashinfer.gdn_decode`) |
| State layout | `[B, HV, K, V]` stride[-2]=1 (K-contiguous in memory) | `[B, HV, V, K]` contiguous (K-contiguous in memory) |
| State dtype (T=1) | bfloat16 | bfloat16 (hardcoded in kernel SMEM) |
| State dtype (T>1) | float32 | float32 (required, no bf16 path) |
| MTP retraction | ✓ `intermediate_states_buffer` | ✓ `intermediate_states_buffer` |

Both MTP kernels support intermediate state caching for speculative decoding retraction.

**Key difference in MTP calling convention:**
- SGLang MTP expects flat `[1, N×T, ...]` inputs + `cu_seqlens`, gate `g = -exp(A_log)·softplus(a + dt_bias)` pre-computed.
- FlashInfer MTP expects batched `[N, T, ...]` inputs, computes gate internally from `A_log/a/dt_bias`.

**Note on MTP bfloat16 state:**
FlashInfer MTP has no native bfloat16 kernel path. Even if the dtype assert is
removed, passing a bfloat16 state triggers a Python-level
`initial_state.to(torch.float32)` copy before the kernel call — a full
allocation + memcopy of the entire state tensor, causing ~5–6x slowdown.
SGLang MTP also computes internally in float32, but the conversion is
block-level and on-the-fly inside the CUDA kernel (`tHgH.load().to(Float32)`),
so no extra copy or allocation occurs. Because the kernel is
memory-bandwidth-bound, using bfloat16 state actually makes SGLang MTP
**faster**: measured ~**1.4–1.5x** speedup over float32 at B≥32 (B=1: ~1.1x),
with half the state memory footprint as a bonus. The benchmark tables above use
float32 states for both sides to keep the comparison fair.

## Setup

**GPU:** NVIDIA B200
**Model config (Qwen3.5):** QK heads=16, V heads=64, head dim=128
**Batch sizes:** 1, 32, 64, 128, 256, 512
**Seq lengths (T):** 1 (decode), 2, 3, 4 (MTP)

## Correctness

All tests pass (atol=0.15, fail\_rate threshold=10%).

| Mode | Typical max diff | Typical mean diff |
|---|---|---|
| T=1 (decode) | ~2e-3 | ~5e-5 |
| T>1 (MTP) | ~1.4e-1 – 3.1e-1 | ~5e-3 |

MTP diffs are larger due to accumulated floating-point errors across multiple tokens in bfloat16, but all fail rates are 0.00%.

## Performance

### T=1 Decode

| Batch | SGLang (μs) | FlashInfer (μs) | FlashInfer speedup |
|------:|------------:|----------------:|-------------------:|
|     1 |        2.98 |            2.77 |             **1.08x** |
|    32 |       34.37 |           21.39 |             **1.61x** |
|    64 |       69.96 |           48.60 |             **1.44x** |
|   128 |      139.18 |           92.75 |             **1.50x** |
|   256 |      286.90 |          182.98 |             **1.57x** |
|   512 |      571.98 |          361.27 |             **1.58x** |

### T=2 MTP

| Batch | SGLang (μs) | FlashInfer (μs) | FlashInfer speedup |
|------:|------------:|----------------:|-------------------:|
|     1 |        5.09 |            6.45 |              0.79x (SGLang faster) |
|    32 |       86.16 |           78.97 |             **1.09x** |
|    64 |      166.49 |          153.18 |             **1.09x** |
|   128 |      329.78 |          291.46 |             **1.13x** |
|   256 |      659.91 |          575.36 |             **1.15x** |
|   512 |     1315.06 |         1142.00 |             **1.15x** |

### T=3 MTP

| Batch | SGLang (μs) | FlashInfer (μs) | FlashInfer speedup |
|------:|------------:|----------------:|-------------------:|
|     1 |        6.48 |            8.07 |              0.80x (SGLang faster) |
|    32 |      119.76 |          114.83 |             **1.04x** |
|    64 |      232.42 |          214.33 |             **1.08x** |
|   128 |      457.53 |          423.85 |             **1.08x** |
|   256 |      915.58 |          841.79 |             **1.09x** |
|   512 |     1843.36 |         1700.66 |             **1.08x** |

### T=4 MTP

| Batch | SGLang (μs) | FlashInfer (μs) | FlashInfer speedup |
|------:|------------:|----------------:|-------------------:|
|     1 |        7.80 |            9.42 |              0.83x (SGLang faster) |
|    32 |      153.27 |          147.22 |             **1.04x** |
|    64 |      296.37 |          267.37 |             **1.11x** |
|   128 |      589.42 |          533.23 |             **1.11x** |
|   256 |     1181.40 |         1068.76 |             **1.11x** |
|   512 |     2397.72 |         2157.96 |             **1.11x** |

## Summary

Benchmarked with `do_bench_cudagraph` to eliminate kernel launch overhead.

- **T=1 decode:** FlashInfer is **1.1x–1.6x** faster. Both kernels use bfloat16 state and the same physical memory layout (K-contiguous). The speedup reflects differences in kernel implementation and algorithm design.
- **T>1 MTP, B=1:** **SGLang is faster** (1.2x–1.3x) — without launch overhead, SGLang's MTP kernel has better compute efficiency at very small batch.
- **T>1 MTP, B≥32:** FlashInfer leads by **1.04x–1.15x** across all batch sizes and seq lengths.

**Recommendation:** Use FlashInfer CuteDSL kernels for both decode and MTP paths in Qwen3.5 serving (especially at batch≥32). At batch=1 MTP, SGLang is competitive.
