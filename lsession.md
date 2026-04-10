Implement a high-performance GEMM (General Matrix Multiply) benchmark for H100 GPU targeting BF16 and FP32 data types, achieving at least 70% of CuBLAS performance. The project has two parts:
1. Part 1: Implement and benchmark custom GEMM kernels for square matrices (power-of-2 dimensions from 32 to 4096) with BF16 and FP32 data types
2. Part 2: Write a Strassen's algorithm design document (already completed at docs/strassen_design.md)
Current phase: Focus on FP32 (CUDA-core) optimizations step-by-step until achieving 70% of CuBLAS Pedantic performance. BF16 Tensor Core optimizations will come after FP32 is complete.
Instructions
- Target GPU: H100 (SM 9.0) for final testing, develop locally on RTX 5070 (SM 12.0, Blackwell)
- Architecture support: Compile for both SM 90 (Hopper) and SM 120 (Blackwell)
- Code style: Modern C++17, avoid over-engineering, keep code simple and consolidated
- No third-party libraries without explicit permission
- Build system: CMake
- Variant naming convention: gemm_{dtype}_rN_suffix.cu (hybrid approach - revision number + optional descriptive suffix)
- Configuration: Define benchmark config in main.cu header section (constexpr), not CLI arguments
- TEST_ALL_VARIANTS flag: Controls whether to benchmark all variants or only latest revisions
- CuBLAS separation: CuBLAS wrapper functions in separate cublas_gemm.cu file
- Correctness: Compare against CuBLAS output (compute L2 relative error)
- CRITICAL: Be conservative, implement changes in small bite-sized steps. Do not over-engineer. Small, targeted build phases.
- FP32 performance target: 70% of CuBLAS Pedantic (CUDA-core) performance
- Phase separation: CUDA-core kernels (FP32) and Tensor Core kernels (BF16/TF32) are separate files
Discoveries
- CuBLAS FP32 insights: cublasSgemm, cublasGemmEx(32F), and cublasGemmEx(32F_PEDANTIC) are nearly identical on RTX 5070 (~6.8ms at 4096), confirming they all use CUDA cores only. TF32 (Tensor Core) is 35% faster at 4.4ms.
- BF16 CuBLAS: Only runs on Tensor Cores. The CUBLAS_COMPUTE_32F and CUBLAS_COMPUTE_32F_FAST_TF32 variants produce nearly identical timings (~2.1-2.2ms at 4096).
- Fair comparison targets: FP32 should compare against Pedantic (CUDA-core path, same hardware). BF16 should compare against TC (Tensor Core path). Comparing CUDA cores vs Tensor Cores is apples-to-oranges.
- WMMA BF16 correctness issues: Earlier attempts at WMMA-based BF16 kernels had matrix B layout mismatch (row-major vs col-major) and accumulator fragment handling issues.
- L2 error for FP32: The L2 error for FP32 naive (~2.6e-4) is computed against the last CuBLAS reference called (TF32), not against the Pedantic reference. This should be fixed to compare against the matching reference.
- NCU profiling: Hit ERR_NVGPUCTRPERM on consumer RTX 5070 — needs elevated permissions. Will work on H100 without issues.
- Git tracking of build artifacts: build/ directory was committed to git before .gitignore was added. Need git rm -r --cached build/ to remove from tracking index.
Accomplished
1. Established fair baselines: Both BF16 and FP32 now use matching naive O(N³) kernels with identical structure (16×16 thread blocks, no shared memory)
2. Expanded CuBLAS references: 
   - BF16: 2 references (32F compute, TF32 compute) — both Tensor Core
   - FP32: 4 references (Sgemm, CUDA 32F, Pedantic 32F, TC/TF32)
3. Verified performance baselines on RTX 5070:
Dim	BF16 Custom	vs CuBLAS TC	FP32 Custom
512	0.14ms	7.0%	0.13ms
1024	1.08ms	3.5%	1.04ms
2048	8.53ms	3.0%	8.28ms
4096	68.7ms	3.1%	72.7ms
4. Added CSV output to results/benchmark_results.csv
5. Rewrote scripts: benchmark.sh, plot_results.py, profile_ncu.sh
6. Generated plot images: BF16, FP32, and summary charts
7. Committed: 69559ec on main branch
Remaining Work
Immediate (before starting FP32 optimizations):
- Run git rm -r --cached build/ and commit to clean up tracked build artifacts
FP32 Optimization Roadmap (step-by-step, conservative):
1. FP32 r1: Tiled kernel (shared memory, 64×64×64 or 128×128×64) — target ~40-50% CuBLAS
2. FP32 r2: Larger tiles + register blocking — target ~50-60% CuBLAS
3. FP32 r3: Double buffering / software pipelining — target ~60-70% CuBLAS
4. FP32 r4: Fine-tuning (swizzling, prefetching) — target 70%+ CuBLAS
After FP32 complete:
- BF16 Tensor Core kernels (separate file: gemm_bf16_tc.cu)
- TF32 Tensor Core kernels (separate file: gemm_tf32.cu)
Relevant files / directories
cu_x_gemm/
├── .gitignore                          # Added: ignores build/, results/, *.csv, *.png, *.log
├── CMakeLists.txt                      # SM 90, 120; updated source file names
├── src/
│   ├── main.cu                         # Config, benchmark harness, CSV output (~270 lines)
│   │                                   # Key config: DIMENSIONS, WARMUP/MEASURE_ITERATIONS, CSV_PATH
│   │                                   # BF16Result/FP32Result structs, bench_cublas lambdas
│   ├── cublas_gemm.cu                  # 6 CuBLAS wrappers: bf16, bf16_tc, fp32_sgemm, fp32_cuda, fp32_pedantic, fp32_tc
│   ├── gemm_bf16.cu                    # BF16 naive r0 kernel (~48 lines)
│   │                                   # Exports: launch_gemm_bf16(), get_variant_id_bf16(), get_variant_desc_bf16()
│   └── gemm_fp32.cu                    # FP32 naive r0 kernel (~48 lines) — NEXT TO OPTIMIZE
│                                       # Exports: launch_gemm_fp32(), get_variant_id_fp32(), get_variant_desc_fp32()
├── scripts/
│   ├── benchmark.sh                    # Build + run + plot workflow
│   ├── plot_results.py                 # Generates 3 PNG charts from CSV
│   └── profile_ncu.sh                  # NCU profiling (needs permissions on consumer GPUs)
├── docs/
│   └── strassen_design.md              # Strassen design document (completed)
└── results/                            # Generated output (gitignored)
    ├── benchmark_results.csv
    ├── gemm_bf16_performance.png
    ├── gemm_fp32_performance.png
    └── gemm_summary.png
Planned future file structure (not yet created):
src/
├── gemm_bf16.cu          # BF16 CUDA-core kernels (current: r0 naive)
├── gemm_bf16_tc.cu       # BF16 Tensor Core kernels (Phase 2)
├── gemm_fp32.cu           # FP32 CUDA-core kernels (current: r0 naive, next: r1 tiled)
└── gemm_tf32.cu           # TF32 Tensor Core kernels (Phase 2)
