# CLAUDE.md

This file provides guidance for working in this repository.

## Current Goal

FP32 is now reduced to two public kernels:
- `naive` for tiny matrices
- `r2z` for larger matrices, with three compile-time configs selected by size

BF16 stays as a separate baseline for now. TF32 / tensor-core work is planned next, so the benchmark output already keeps both the CUDA-core and TF32 reference paths visible.

## Build And Run

```bash
mkdir -p build && cmake -S . -B build && cmake --build build -j$(nproc)
./scripts/benchmark.sh
```

The benchmark writes `results/benchmark_results.csv` and prints three report sections:
- `BF16`
- `FP32 (CU Cores)`
- `FP32 (TF32 / Tensor Cores)`

## FP32 Architecture

### Public Entry Points

- `launch_gemm_fp32_naive(...)`
- `launch_gemm_fp32_master(...)`
- `launch_gemm_fp32_master_debug(...)`

`launch_gemm_fp32_master` is the user-visible FP32 selector. It uses `M * N` and dispatches to either `naive` or `r2z`. The r2z launcher lives in a single file, `src/gemm_fp32_r2z.cu`, and internally selects one of three compile-time configs:
- small: 64x64, BK=32
- medium: 64x64, BK=16
- large: 128x128, BK=16

### Important References

- `cublas_gemm_fp32_pedantic(...)` is the primary CUDA-core reference.
- `cublas_gemm_fp32_tc(...)` is the TF32 / tensor-core reference.

Do not reintroduce `FP32_VARIANT`. The code is intentionally moving away from compile-time variant ladders.

## Repository Layout

- `fp32Optimizations.md` is the consolidated FP32 findings log.
- `docs/bf16Prep.md` is the BF16 / future tensor-core strategy doc.
- `docs/hwAnalysis.md`, `docs/ncu_guide.md`, `docs/RTX5070Specs.md`, `docs/H100specs.md` remain active.
- Old FP32 analysis docs are archived in `scratch/archive/docs/`.
- Old size-specific FP32 source files are archived in `scratch/archive/fp32/`.

## Notes For Future Changes

- Keep the selector key as `M * N` for now. The project is only benchmarking square matrices.
- Preserve the current benchmark sections in stdout and in the CSV.
- `scripts/optimizer.sh` has been removed. `scripts/benchmark.sh` is the active entry point.
- `plot_results.py` is now optional / legacy and should not drive the normal workflow.
