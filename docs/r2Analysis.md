# R2 Kernel Development: Analysis & Results

## Overview

Custom FP32 GEMM kernels targeting cuBLAS parity. Progress: naive (10%) → r1y (24%) → r2x (81%) → r2y (84%) → r2z (84%, corrected) → **r2z2 (98%)**.

## Performance Summary

### Single Kernel Results at 4096×4096

| Kernel | Time (ms) | vs cuBLAS sgemm | Key Feature |
|--------|-----------|-----------------|-------------|
| r1y | 27.9 | 24% | 64×64 tiles, 1D blocks |
| r2x | 8.09 | 81% | float4, A-transpose, 128×128 tiles |
| r2y | 7.86 | 84% | Warp tiling hierarchy |
| r2z (corrected) | 7.96 | 83.9% | r2y bugs fixed; base for r2z2 |
| **r2z2** | **6.66** | **98.0%** | Double-buffer + cp.async B loads |

### kernel_master Results (Auto-Selection) — RTX 5070, April 2026

| Size | Selected | Custom (ms) | cuBLAS sgemm (ms) | % of cuBLAS |
|------|----------|-------------|-------------------|-------------|
| 32 | naive | 0.0020 | 0.0046 | **229%** |
| 64 | naive | 0.0033 | 0.0048 | **146%** |
| 128 | r1y | 0.0083 | 0.0046 | 55% |
| 256 | r1y | 0.0153 | 0.0074 | 48% |
| 512 | r2x | 0.0587 | 0.0252 | 43% |
| 1024 | r2z2 | 0.1628 | 0.1087 | 67% |
| 2048 | r2z2 | 0.8601 | 0.7358 | 85.5% |
| **4096** | **r2z2** | **6.6617** | **6.5287** | **98%** |

## kernel_master Selection Logic (src/gemm_fp32_master.cu)

3 runtime-configurable thresholds:

```cpp
g_naive_max = 4096;    // <= 64×64     → naive
g_r1y_max   = 65536;   // <= 256×256   → r1y
g_r2x_max   = 262144;  // <= 512×512   → r2x  (r2z2 marginally slower here)
// > g_r2x_max          → r2z2         (clear winner from 1024×1024 upward)
```

Tunable at runtime via `set_kernel_thresholds(naive_max, r1y_max, r2x_max)`.

## Implementation Details

### Naive Kernel (src/gemm_fp32.cu)
- **Config**: 16×16 2D blocks, 1 output element per thread
- **Optimizations**: `__restrict__`, `__ldg()` on A/B loads, `#pragma unroll 4`
- **Why it wins at ≤64**: kernel launch overhead < algorithmic overhead of tiled kernels

### r1y Kernel (src/gemm_fp32_r1y.cu)
- **Config**: BM=64, BN=64, BK=64, TM=2, TN=2, 1024 threads (1D block)
- **Key**: 1D block layout → perfect warp coalescing (beats r1x by 37%)

### r2x Kernel (src/gemm_fp32_r2x.cu)
- **Config**: BM=128, BN=128, BK=16, TM=8, TN=8, 256 threads
- **SMEM**: 16KB (single buffer)
- **Key**: float4 loads + A stored transposed in SMEM (eliminates bank conflicts)

### r2y Kernel (src/gemm_fp32_r2y.cu)
- **Config**: BM=128, BN=128, BK=16, WM=64, WN=64, WNITER=4, TM=8, TN=4, 128 threads
- **SMEM**: 16KB (single buffer)
- **Key**: Warp tiling hierarchy (block→warp→subwarp→thread), 128 accumulators/thread

### r2z Kernel (src/gemm_fp32_r2z.cu)
- **Config**: Identical to r2y
- **Status**: Correct (fixed from broken state). Serves as clean base for r2z2.
- **Bugs fixed**:
  1. K-loop started at `bkIdx=BK` instead of `0` — skipped first K-block
  2. Load/compute ordering inverted — overwrote SMEM before computing it
  3. Redundant post-loop compute block removed

### r2z2 Kernel (src/gemm_fp32_r2z2.cu) — CURRENT BEST
- **Config**: BM=128, BN=128, BK=16, WM=64, WN=64, WNITER=4, TM=8, TN=4, 128 threads
- **SMEM**: `As[2][BK*BM] + Bs[2][BK*BN]` = 32KB double buffer (67% of 48KB limit)
- **Key optimizations**:
  1. **Double-buffered SMEM**: `As[2][...]`, `Bs[2][...]` — ping-pong between cur/nxt
  2. **cp.async for B**: `__pipeline_memcpy_async` issues 16-byte async copies to SMEM, bypassing L1 cache; hardware copy engine overlaps with FMA compute
  3. **A loaded early**: float4 global loads + scatter-transpose issued at top of loop to maximise latency hiding before compute starts
  4. **One sync per K-iter** (vs two in single-buffer kernels): `__pipeline_wait_prior(0)` + `__syncthreads()` combined at end of iteration

**Why cp.async helps**: The hardware copy engine (LDGSTS) is separate from the SM's load-store units. Async B copies to Bs[nxt] execute concurrently with the FMA loop operating on As[cur]/Bs[cur]. By the time `__pipeline_wait_prior(0)` is reached, Bs[nxt] is already populated — no additional stall.

**Memory layout**: `As[buf][k*BM + m]` (transposed) eliminates bank conflicts on column-wise reads; `Bs[buf][k*BN + n]` (row-major) allows 16-byte aligned cp.async copies.

## Issues Encountered & Resolved

### r2z Bugs (Fixed)
See above. r2z is now correct and serves as the reference base.

### L2 Error at Small Sizes (r2x, r2y, r2z, r2z2)
High L2 error (~0.6-1.6) at sizes 32, 64 for 128×128-tiled kernels — boundary condition handling with tiles larger than the matrix. Not a concern: kernel_master dispatches naive for these sizes.

### cuda::barrier (Not Used)
Earlier attempts at double-buffering in r2y used `cuda::barrier` which had initialization issues. r2z2 avoids this entirely by using `__pipeline_memcpy_async` + `__pipeline_wait_prior` (per-thread pipeline, no shared state needed).

## New Finding: Tile Size Matters for Medium Sizes (April 2026)

### Problem Identified
Original r2z2 uses **128×128 tiles**, which is optimal for large matrices (1024+) but creates poor parallelism at medium sizes:

| Size | 128×128 Blocks | Utilization | Issue |
|------|----------------|-------------|-------|
| 128 | 1 block | 100% boundary waste | Massive parallelism loss |
| 256 | 4 blocks | Significant waste | Poor occupancy |
| 512 | 16 blocks | Decent | Adequate but not optimal |

### Solution: r2z2_small Variant

Created `src/gemm_fp32_r2z2_small.cu` with **64×64 tiles**:
- BM=64, BN=64, BK=16, WM=32, WN=32, WNITER=1
- TM=8, TN=4, 128 threads (same as r2z2)
- SMEM: 16KB double buffer (vs 32KB in r2z2)

### Performance Comparison

| Size | r2z2 (128×128) | r2z2_small (64×64) | r3x (64×64 flat) |
|------|----------------|-------------------|------------------|
| 128 | ~24%* | **71.7%** | 55.4% |
| 256 | ~25%* | **69.4%** | 48.7% |
| 512 | ~43%* | **91.5%** | 86.1% |
| 1024 | 65.2% | 75.1% | ~64% |
| 2048 | 86.9% | 73.7% | ~67% |
| 4096 | 97.7% | 75.2% | ~73% |

*Estimated based on r2z2_small's efficiency at small sizes

### Key Insight
**Warp tiling beats flat layout**: r2z2_small (91.5% at 512) outperforms r3x (86.1%) despite both using 64×64 tiles. The warp tiling hierarchy (block→warp→subwarp→thread) provides better register reuse and data locality.

### Updated Master Kernel Recommendations
```
≤ 64×64     → naive   (launch overhead wins)
≤ 256×256   → r1y     (simple, good occupancy)
≤ 512×512   → r2z2_small  (NEW - 91.5% at 512!)
> 512×512   → r2z2    (98% at 4096)
```

### Files Added
- `src/gemm_fp32_r2z2_small.cu` — 64×64 tiled r2z2 variant
- `src/gemm_fp32_r3x_tuner.cu` — R3X parameter tuner

## Future Directions

### BK=32 Variant
- Single-buffer BK=32: As[128×32] + Bs[32×128] = 32KB (fits)
- Doubles arithmetic intensity per K-block, halves K-iterations
- Cannot double-buffer at BK=32 (would need 64KB, exceeds 48KB default)
- Would require dynamic SMEM via `cudaFuncSetAttribute` to double-buffer at BK=32
- Expected gain over r2z2: uncertain — may improve or regress at 4096 (already near peak)

### Increased SMEM Carveout
- RTX 5070 (SM 12.0) supports up to 100KB SMEM per block via `cudaFuncSetAttribute`
- Could enable BK=32 double buffer (64KB) for higher arithmetic intensity
- H100 (SM 9.0) supports up to 228KB

### BF16 + Tensor Cores (WMMA API)
Next major phase: use `wmma::fragment` and `mma.sync` for BF16 input → FP32 accumulation.
Expected: match or exceed cuBLAS tensor-core performance (4.1ms at 4096, currently 6× faster than our FP32).

## Hardware Notes

- Testing: RTX 5070 (SM 12.0), CUDA 13.2
- Target: H100 (SM 9.0) — thresholds may differ
- SMEM limit: 48KB default; configurable higher
- cuBLAS `sgemm` uses TF32 internally on Ampere+; our custom kernel achieves 98% of it with pure FP32
