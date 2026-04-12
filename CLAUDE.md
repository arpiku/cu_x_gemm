# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Custom CUDA GEMM kernels (FP32, then BF16 with tensor cores) to match or beat cuBLAS performance. Current best: **~84% of cuBLAS at 4096×4096** (r2y kernel). Target: 90%+ via double buffering (r2z), then full parity via BF16 + WMMA tensor cores.

## Build & Run

```bash
mkdir -p build && cd build && cmake .. && make -j$(nproc)
cd build && ./cu_x_gemm
```

Results are written to `results/benchmark_results.csv`. Stdout shows a formatted table comparing custom kernel vs cuBLAS (sgemm, CUDA, pedantic, tensor-core modes) across sizes 32–4096.

### Switching the Active FP32 Variant

Edit `src/main.cu` line 26:
```cpp
#define FP32_VARIANT 6   // 1=naive, 2=r1x, 3=r1y, 4=r2x, 5=r2y, 6=master, 7=r2z
```

Rebuild and re-run. All variants compile regardless; only the selected one is benchmarked (unless `TEST_ALL_VARIANTS = true`).

### NCU Profiling

```bash
./scripts/ncu_profile.sh <variant_number> "<sizes>"
# e.g.: ./scripts/ncu_profile.sh 5 "2048 4096"
# Output lands in results/ncu/
```

## Architecture

### Kernel Conventions

Every kernel file exposes three symbols:
```cpp
void launch_gemm_fp32_<name>(const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta, cudaStream_t stream);
const char* get_variant_id_fp32_<name>();
const char* get_variant_desc_fp32_<name>();
```

`main.cu` selects one via `#define FP32_VARIANT` and maps it to the generic `launch_gemm_fp32` / `get_variant_id_fp32` / `get_variant_desc_fp32` macros. Adding a new kernel requires: adding the file to `CMakeLists.txt`, adding an `extern` declaration and `#elif` case in `main.cu`, and assigning a new variant number.

### Kernel Hierarchy (FP32)

| Variant | File | Config | Key Technique | ~4096 perf |
|---------|------|--------|---------------|------------|
| naive | `gemm_fp32.cu` | 16×16 2D blocks | None | 10% cuBLAS |
| r1x | `gemm_fp32_r1x.cu` | BM=128, BN=64, BK=64, TM=4, TN=4, 512t | SMEM tiling, 2D blocks | 15% |
| r1y | `gemm_fp32_r1y.cu` | BM=64, BN=64, BK=64, TM=2, TN=2, 1024t | 1D blocks → perfect warp coalescing | 24% |
| r2x | `gemm_fp32_r2x.cu` | BM=128, BN=128, BK=16, TM=8, TN=8, 256t | float4 loads + transposed-A in SMEM | 81% |
| r2y | `gemm_fp32_r2y.cu` | BM=128, BN=128, BK=16, WM=64, WN=64, WNITER=4, TM=8, TN=4, 128t | Warp tiling hierarchy, `__launch_bounds__` | 84% |
| r2z | `gemm_fp32_r2z.cu` | Same as r2y | Fixed K-loop (base for r2z2, not dispatched by master) | 84% |
| **r2z2** | **`gemm_fp32_r2z2.cu`** | Same as r2y | **Double-buffer SMEM + cp.async B loads** | **98%** |

r2x → r2y: float4 vectorized global loads, A stored **transposed** in shared memory (`As[k*BM + m]` layout) to eliminate bank conflicts when threads read along the M dimension.

r2y adds a warp-level tiling layer: each block (128t = 4 warps) is subdivided into warp tiles (64×64), then sub-warp tiles (WNITER=4 column strips), then thread tiles (TM=8, TN=4). This adds a 3rd level of register reuse.

### kernel_master (`gemm_fp32_master.cu`)

Auto-selects kernel based on `M*N` with 3 runtime-configurable thresholds:
- `≤ 4096` (64×64): naive — launch overhead wins, beats cuBLAS
- `≤ 65536` (256×256): r1y
- `≤ 262144` (512×512): r2x — marginally faster than r2z2 here
- `> 262144`: r2z2 — double-buffer + cp.async, clear winner from 1024×1024 up

Tunable via `set_kernel_thresholds(naive_max, r1y_max, r2x_max)`.

### r2z2 — Double Buffer + cp.async (Current Best)

SMEM layout: `As[2][BK*BM]` (transposed) + `Bs[2][BK*BN]` (row-major) = 32KB total (fits in 48KB limit).

K-loop structure:
```
Prefetch tile[0] → As[0], Bs[0]:  A via float4+scatter, B via __pipeline_memcpy_async
__pipeline_wait_prior(0) + __syncthreads()
cur=0, nxt=1
for bkIdx = BK to K-1:
    Issue A loads for tile[bkIdx] → As[nxt]   (float4 + scatter)
    Issue B async copies → Bs[nxt]              (__pipeline_memcpy_async)
    __pipeline_commit()
    Compute BK outer products on As[cur], Bs[cur]   ← overlaps with B async copies
    __pipeline_wait_prior(0) + __syncthreads()
    cur ^= 1; nxt ^= 1
Compute final tile on As[cur], Bs[cur]
```

The hardware async copy engine (LDGSTS) fills `Bs[nxt]` concurrently with FMA on `As[cur]/Bs[cur]`, bypassing L1 to preserve it for compute.

### Shared Memory Layout Convention

- A is stored **transposed**: `As[k * BM + m]` — threads reading along M dimension get stride-1 bank access.
- B is stored row-major: `Bs[k * BN + n]` — standard layout.
- Total SMEM for r2x/r2y/r2z = `BM*BK + BK*BN` floats = 128×16 + 16×128 = 4096 floats = **16 KB** (well within 48 KB limit).

### Benchmarking

`main.cu` runs 10 warmup + 50 measured iterations per size, using `cudaEvent_t` timing. L2 relative error is computed against `cublas_gemm_fp32_pedantic` (pure FP32, no TF32). All four cuBLAS modes are timed: sgemm, CUDA (may use TF32), pedantic (pure FP32), tensor-core (TF32 accelerated).

### Archived Kernels

`scratch/` holds deprecated variants (r1a–r1d, r1x2). See `scratch/SCRATCH_INDEX.md` to re-enable one. Key lesson from r1x2: changing B's SMEM layout from `Bs[BK][BN]` to `Bs[BN][BK]` caused severe bank conflicts and 3× slowdown.

## Hardware Context

- **Dev machine**: RTX 5070, SM 12.0, CUDA arch 120
- **Target**: H100, SM 9.0, CUDA arch 90
- SMEM limit: 48 KB (default), configurable to 96 KB with `cudaFuncSetAttribute`
- CUDA 13.x, C++17
- `CMAKE_CUDA_ARCHITECTURES 90 120` — both arches compiled

## Key Documentation

- `docs/r1Analysis.md` — detailed development journal for r1 series; common CUDA mistakes and their fixes
- `docs/r2Analysis.md` — r2 series results, r2z strategy, known issues
- `docs/ncu_guide.md` — how to interpret NCU metrics for GEMM (stall types, roofline, SMEM throughput)
- `docs/hwAnalysis.md` / `docs/RTX5070Specs.md` / `docs/H100specs.md` — hardware specs for roofline analysis
