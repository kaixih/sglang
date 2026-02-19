# GDN Kernel Benchmark

Benchmarks GDN (Gated Delta Net) kernels for Qwen3-Next linear attention models.

## Quick Start

```bash
# Decode mode (T=1,2,3,4 for MTP): FlashInfer vs SGLang CuTe DSL vs Triton
python benchmark_gdn_kernels.py --decode

# Prefill mode (T≥128): Triton chunk-based
python benchmark_gdn_kernels.py

# Custom decode sequence lengths
python benchmark_gdn_kernels.py --decode --decode-seq-lengths 1,4
```

## Reproducing Results with FlashInfer

To benchmark all 3 kernels including FlashInfer, install FlashInfer PR#2498:

```bash
# Clone FlashInfer and checkout PR#2498
git clone https://github.com/flashinfer-ai/flashinfer.git
cd flashinfer
git fetch origin pull/2498/head:pr-2498
git checkout pr-2498

# Install without dependencies (assumes SGLang environment already has required deps)
pip install -e . --no-deps --no-build-isolation

# Verify installation
python -c "from flashinfer.cute_dsl.gated_delta_rule import gated_delta_rule; print('FlashInfer available')"

# Run benchmark
cd /path/to/sglang/benchmark/kernels/qwen3_next_study
python benchmark_gdn_kernels.py --decode
```

**Note:** Without FlashInfer installed, the benchmark will compare only SGLang CuTe DSL vs Triton (2-way comparison).

## Model Architecture

Qwen3-Next model configuration (shape details omitted for sensitivity)

## Decode Mode (T=1,2,3,4): Memory-Bound (MTP)

Tests decode kernels with multi-token prediction (MTP) on GB200:
- **FlashInfer CuTe DSL**: Optimized kernel from [FlashInfer PR#2498](https://github.com/flashinfer-ai/flashinfer/pull/2498)
- **SGLang CuTe DSL**: GB200-specific optimized kernel (current SGLang implementation)
- **Triton**: Portable fallback for all GPUs

### Results

```
DECODE PERFORMANCE (T=1)
--------------------------------------------------------------------------------
  B=   1, T=   1 (SmallBatch): Triton=   31.33μs, CuTeDSL=  173.70μs, speedup=0.18x, FlashInfer=   40.83μs, speedup=0.77x
  B=  32, T=   1 (SmallBatch): Triton=   65.54μs, CuTeDSL=  177.54μs, speedup=0.37x, FlashInfer=   45.66μs, speedup=1.44x
  B=  64, T=   1 (SmallBatch): Triton=  120.86μs, CuTeDSL=  176.74μs, speedup=0.68x, FlashInfer=   52.29μs, speedup=2.31x
  B= 128, T=   1 (LargeBatch): Triton=  231.46μs, CuTeDSL=  191.49μs, speedup=1.21x, FlashInfer=   92.16μs, speedup=2.51x
  B= 256, T=   1 (LargeBatch): Triton=  455.68μs, CuTeDSL=  368.67μs, speedup=1.24x, FlashInfer=  174.08μs, speedup=2.62x
  B= 512, T=   1 (LargeBatch): Triton=  903.62μs, CuTeDSL=  724.99μs, speedup=1.25x, FlashInfer=  338.91μs, speedup=2.67x

DECODE PERFORMANCE (T=2)
--------------------------------------------------------------------------------
  B=   1, T=   2 (SmallBatch): Triton=   30.11μs, CuTeDSL=  167.94μs, speedup=0.18x, FlashInfer=   41.15μs, speedup=0.73x
  B=  32, T=   2 (SmallBatch): Triton=   82.72μs, CuTeDSL=  176.70μs, speedup=0.47x, FlashInfer=   47.57μs, speedup=1.74x
  B=  64, T=   2 (SmallBatch): Triton=  154.59μs, CuTeDSL=  177.79μs, speedup=0.87x, FlashInfer=   61.70μs, speedup=2.51x
  B= 128, T=   2 (LargeBatch): Triton=  287.74μs, CuTeDSL=  191.52μs, speedup=1.50x, FlashInfer=  110.62μs, speedup=2.60x
  B= 256, T=   2 (LargeBatch): Triton=  563.14μs, CuTeDSL=  368.67μs, speedup=1.53x, FlashInfer=  208.64μs, speedup=2.70x
  B= 512, T=   2 (LargeBatch): Triton= 1114.08μs, CuTeDSL=  724.99μs, speedup=1.54x, FlashInfer=  399.68μs, speedup=2.79x

DECODE PERFORMANCE (T=3)
--------------------------------------------------------------------------------
  B=   1, T=   3 (SmallBatch): Triton=   30.72μs, CuTeDSL=  168.10μs, speedup=0.18x, FlashInfer=   41.10μs, speedup=0.75x
  B=  32, T=   3 (SmallBatch): Triton=   98.30μs, CuTeDSL=  176.80μs, speedup=0.56x, FlashInfer=   46.34μs, speedup=2.12x
  B=  64, T=   3 (SmallBatch): Triton=  182.66μs, CuTeDSL=  177.50μs, speedup=1.03x, FlashInfer=   71.90μs, speedup=2.54x
  B= 128, T=   3 (LargeBatch): Triton=  347.71μs, CuTeDSL=  191.66μs, speedup=1.81x, FlashInfer=  131.07μs, speedup=2.65x
  B= 256, T=   3 (LargeBatch): Triton=  674.82μs, CuTeDSL=  368.67μs, speedup=1.83x, FlashInfer=  248.51μs, speedup=2.72x
  B= 512, T=   3 (LargeBatch): Triton= 1337.58μs, CuTeDSL=  724.99μs, speedup=1.84x, FlashInfer=  480.26μs, speedup=2.79x

DECODE PERFORMANCE (T=4)
--------------------------------------------------------------------------------
  B=   1, T=   4 (SmallBatch): Triton=   30.69μs, CuTeDSL=  168.30μs, speedup=0.18x, FlashInfer=   42.42μs, speedup=0.72x
  B=  32, T=   4 (SmallBatch): Triton=  116.74μs, CuTeDSL=  175.47μs, speedup=0.67x, FlashInfer=   52.61μs, speedup=2.22x
  B=  64, T=   4 (SmallBatch): Triton=  219.17μs, CuTeDSL=  175.42μs, speedup=1.25x, FlashInfer=   88.16μs, speedup=2.49x
  B= 128, T=   4 (LargeBatch): Triton=  416.03μs, CuTeDSL=  191.07μs, speedup=2.18x, FlashInfer=  161.79μs, speedup=2.57x
  B= 256, T=   4 (LargeBatch): Triton=  783.78μs, CuTeDSL=  369.02μs, speedup=2.12x, FlashInfer=  306.94μs, speedup=2.55x
  B= 512, T=   4 (LargeBatch): Triton= 1556.50μs, CuTeDSL=  724.99μs, speedup=2.15x, FlashInfer=  598.02μs, speedup=2.60x
```

### Key Findings (Decode)

**🥇 FlashInfer dominates at large batch sizes:**
- **T=1, B=512**: FlashInfer (339μs) is **2.67× faster than Triton** and **2.14× faster than SGLang CuTe DSL**
- **T=4, B=512**: FlashInfer (598μs) is **2.60× faster than Triton** and **1.21× faster than SGLang CuTe DSL**
- **Consistent performance**: Maintains ~2.6× speedup across T=1,2,3,4
- **Best at**: B≥32 for all T values

**🥈 SGLang CuTe DSL competitive:**
- **T=4, B=512**: 1.21× slower than FlashInfer but **2.15× faster than Triton**
- **Best at**: B≥64 when FlashInfer not available

**🥉 Triton baseline:**
- Better only at very small batches (B≤32)
- Portable to all GPUs

**Recommendation:** Use FlashInfer for production workloads with B≥32

**Note on batch size labels:** The "SmallBatch"/"LargeBatch" labels in results are for demonstration purposes only. They don't reflect SGLang's actual kernel selection, which is controlled by the `SGLANG_USE_CUTEDSL_GDN_DECODE` environment variable (default: always Triton). For optimal performance, dynamic kernel selection based on batch size would be beneficial.

## Prefill Mode (T≥128): Compute-Bound

Tests multi-token prefill with Triton's chunk-based `chunk_gated_delta_rule`:
- Processes sequences in 64-token chunks
- No CuTe DSL version available (Triton only)

### Results

```
PREFILL PERFORMANCE
--------------------------------------------------------------------------------
  B=   1, T= 128 (SmallBatch): Triton=  444.48μs
  B=  32, T= 128 (LargeBatch): Triton=  621.57μs
  B=  64, T= 128 (LargeBatch): Triton= 1179.65μs
  B= 128, T= 128 (LargeBatch): Triton= 2294.82μs
  B= 256, T= 128 (LargeBatch): Triton= 4542.53μs

  B=   1, T=1024 (SmallBatch): Triton=  424.83μs
  B=  32, T=1024 (LargeBatch): Triton= 3647.71μs
  B=  64, T=1024 (LargeBatch): Triton= 7464.96μs
  B= 128, T=1024 (LargeBatch): Triton=15102.48μs
  B= 256, T=1024 (LargeBatch): Triton=30416.90μs
```

### Key Findings (Prefill)

- **Linear scaling**: Time roughly doubles when T doubles (621μs @ T=128 → 3648μs @ T=1024)
- **Efficiency**: Reaches peak at T≥1024 for B=32
- **Batch parallelism**: Good scaling up to B=256

## Summary

| Mode | Winner | Workload | Best Config | Performance |
|------|--------|----------|-------------|-------------|
| Decode (T=1) | **FlashInfer** | Memory-bound | B=512, T=1 | **2.67× speedup** (904μs → 339μs) |
| Decode (T=2) | **FlashInfer** | Memory-bound | B=512, T=2 | **2.79× speedup** (1114μs → 400μs) |
| Decode (T=3) | **FlashInfer** | Memory-bound | B=512, T=3 | **2.79× speedup** (1338μs → 480μs) |
| Decode (T=4) | **FlashInfer** | Memory-bound | B=512, T=4 | **2.60× speedup** (1557μs → 598μs) |
| Prefill | Triton chunk | Compute-bound | B=32, T=1024 | 3648μs (linear scaling) |

**GPU**: NVIDIA GB200

### Kernel Comparison (B=512)

| T | Triton | SGLang CuTe DSL | FlashInfer | Winner |
|---|--------|-----------------|------------|--------|
| 1 | 904μs | 725μs (1.25×) | **339μs (2.67×)** | FlashInfer |
| 2 | 1114μs | 725μs (1.54×) | **400μs (2.79×)** | FlashInfer |
| 3 | 1338μs | 725μs (1.84×) | **480μs (2.79×)** | FlashInfer |
| 4 | 1557μs | 725μs (2.15×) | **598μs (2.60×)** | FlashInfer |

## Metrics

- **Time** (μs): Median kernel execution time
- **Speedup** (Decode): Triton time / CuTe DSL time

## Related Files

### SGLang Kernels
- **Decode kernel (SGLang CuTe DSL)**: `python/sglang/jit_kernel/cutedsl_gdn.py`
- **Decode kernel (Triton)**: `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`
- **Prefill kernel (Triton)**: `python/sglang/srt/layers/attention/fla/chunk.py`
- **Model**: `python/sglang/srt/models/qwen3_next.py:408-414`
- **Backend**: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`

### FlashInfer Kernel (External)
- **Decode kernel (FlashInfer CuTe DSL)**: [FlashInfer PR#2498](https://github.com/flashinfer-ai/flashinfer/pull/2498)
- **Location**: `flashinfer/cute_dsl/gated_delta_rule.py`
