# GDN Kernel Benchmark: SGLang CuteDSL Transpose vs FlashInfer CuteDSL

Compares two high-performance GDN (Gated Delta Net) kernel implementations for Qwen3-Next.

| | SGLang | FlashInfer |
|---|---|---|
| PR | [#17981](https://github.com/sgl-project/sglang/pull/17981) | [#2498](https://github.com/flashinfer-ai/flashinfer/pull/2498) |
| T=1 kernel | `cutedsl_fused_recurrent_sigmoid_gated_delta_rule_update` | `gated_delta_rule` (from `flashinfer.gdn_kernels`) |
| T>1 kernel | `cutedsl_fused_recurrent_gated_delta_rule_update` | `gated_delta_rule_mtp` (from `flashinfer.gdn_decode`) |
| State layout | `[B, HV, K, V]` stride[-2]=1 (K-contiguous) | `[B, HV, V, K]` contiguous |
| State dtype (T=1) | bfloat16 | bfloat16 (hardcoded in kernel SMEM) |
| State dtype (T>1) | float32 | float32 (required, no bf16 path) |
| MTP retraction | ✓ `intermediate_states_buffer` | ✓ `intermediate_states_buffer` |

Both MTP kernels support intermediate state caching for speculative decoding retraction.

**Key difference in MTP calling convention:**
- SGLang MTP expects flat `[1, N×T, ...]` inputs + `cu_seqlens`, gate `g = -exp(A_log)·softplus(a + dt_bias)` pre-computed.
- FlashInfer MTP expects batched `[N, T, ...]` inputs, computes gate internally from `A_log/a/dt_bias`.

## Setup

**GPU:** NVIDIA B200
**Model config (Qwen3-Next):** QK heads=16, V heads=64, head dim=128
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
|     1 |       75.07 |           12.29 |             **6.11x** |
|    32 |       76.83 |           30.72 |             **2.50x** |
|    64 |       77.86 |           53.28 |             **1.46x** |
|   128 |      145.41 |           96.29 |             **1.51x** |
|   256 |      290.82 |          182.18 |             **1.60x** |
|   512 |      575.42 |          356.22 |             **1.62x** |

### T=2 MTP

| Batch | SGLang (μs) | FlashInfer (μs) | FlashInfer speedup |
|------:|------------:|----------------:|-------------------:|
|     1 |      108.80 |           13.31 |             **8.17x** |
|    32 |      109.60 |           85.02 |             **1.29x** |
|    64 |      171.97 |          158.62 |             **1.08x** |
|   128 |      329.70 |          294.02 |             **1.12x** |
|   256 |      655.47 |          580.58 |             **1.13x** |
|   512 |     1316.82 |         1157.22 |             **1.14x** |

### T=3 MTP

| Batch | SGLang (μs) | FlashInfer (μs) | FlashInfer speedup |
|------:|------------:|----------------:|-------------------:|
|     1 |      108.42 |           15.36 |             **7.06x** |
|    32 |      124.99 |          121.89 |             **1.03x** |
|    64 |      235.55 |          216.16 |             **1.09x** |
|   128 |      456.70 |          418.85 |             **1.09x** |
|   256 |      915.60 |          844.96 |             **1.08x** |
|   512 |     1848.29 |         1699.47 |             **1.09x** |

### T=4 MTP

| Batch | SGLang (μs) | FlashInfer (μs) | FlashInfer speedup |
|------:|------------:|----------------:|-------------------:|
|     1 |      109.33 |           17.41 |             **6.28x** |
|    32 |      159.71 |          152.64 |             **1.05x** |
|    64 |      301.06 |          271.36 |             **1.11x** |
|   128 |      586.66 |          532.61 |             **1.10x** |
|   256 |     1181.60 |         1067.04 |             **1.11x** |
|   512 |     2386.98 |         2167.94 |             **1.10x** |

## Summary

**FlashInfer is consistently faster across all configurations:**

- **T=1 decode:** FlashInfer is **1.5x–6.1x** faster. Both kernels use bfloat16 state (FlashInfer's kernel hardcodes it in SMEM; SGLang is configured to match). The speedup reflects algorithmic differences: VK layout (`[B,HV,V,K]` K-fast) vs SGLang's KV-transposed layout with stride[-2]=1.
- **T>1 MTP, B=1:** FlashInfer is **6–8x** faster (kernel launch overhead dominates at small batch).
- **T>1 MTP, B≥32:** FlashInfer leads by **1.03x–1.14x** across all batch sizes and seq lengths.

**Recommendation:** Use FlashInfer CuteDSL kernels for both decode and MTP paths in Qwen3-Next serving.
