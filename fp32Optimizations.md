# FP32 Optimizations Log

## Status

FP32 is now organized around two public kernels:
- `naive` for tiny matrices
- `r2z` for all larger matrices

The `r2z` launcher is a single file with three compile-time config buckets selected by `M * N`:
- small: `64x64`, `BK=32`
- medium: `64x64`, `BK=16`
- large: `128x128`, `BK=16`

The user-visible dispatcher remains `launch_gemm_fp32_master(...)`. It selects `naive` for the smallest sizes and `r2z` otherwise. `launch_gemm_fp32_master_debug(...)` is used by the benchmark to record which sub-config was chosen.

## Current Performance Snapshot

Reference for CUDA-core comparison is `cublas_gemm_fp32_pedantic(...)`. The current custom FP32 path on RTX 5070 is approximately:

| Size | Selected | Custom vs Pedantic |
|------|----------|--------------------|
| 32 | naive | ~232% |
| 64 | naive | ~139% |
| 128 | r2z_small | ~72% |
| 256 | r2z_small | ~70% |
| 512 | r2z_medium | ~92% |
| 1024 | r2z_medium | ~80% |
| 2048 | r2z_large | ~89% |
| 4096 | r2z_large | ~98% |

Takeaway:
- `naive` still wins for tiny matrices because launch overhead dominates.
- `r2z_small` is the best current choice for 128-256.
- `r2z_medium` is the best current choice for 512-1024.
- `r2z_large` reaches near parity at 4096.

## Why The R2Z Consolidation Exists

The old layout had one source file per tuned size, but the kernel body was almost identical and only the compile-time config changed. Consolidating the implementation:
- removes the FP32 variant ladder from `main.cu`
- keeps the useful size-specific tuning
- makes future TF32/BF16 dispatch easier to add without multiplying source files

## Architecture Notes

- `src/gemm_fp32_master.cu` owns the policy decision between `naive` and `r2z`.
- `src/gemm_fp32_r2z.cu` owns the compile-time config buckets and the templated kernel implementation.
- `M * N` remains the selector key because the benchmark only targets square matrices for now.
- `cublas_gemm_fp32_pedantic(...)` is the primary CUDA-core reference.
- `cublas_gemm_fp32_tc(...)` is kept as the TF32 reference for the future tensor-core path.

## What Changed

- Removed `FP32_VARIANT` from the active code path.
- Removed the old per-size `r2z_*` files from `src/`.
- Consolidated the FP32 findings into this file.
- Archived the historical FP32 docs in `scratch/archive/docs/`.
- Archived the old per-size FP32 source files in `scratch/archive/fp32/`.
- Archived the old `r3x` tuner utility in `scratch/archive/fp32/`.

## Follow-Up Work

- Add the TF32 custom kernel when ready and wire it into the same reporting structure.
- Keep BF16 separate until the BF16 tensor-core path is implemented.
- Update any old scripts or notes that still assume variant-number dispatch.
