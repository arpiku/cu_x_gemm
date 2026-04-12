# Scrapped Kernels Index

## Overview

This folder contains kernels that have been archived due to poor performance or being superseded by better implementations. They are kept for reference, experimentation, and potential future use.

## Performance Summary (4096×4096)

| Kernel | Time (ms) | vs CuBLAS | SMEM | AI | Status |
|--------|-----------|-----------|------|-----|--------|
| gemm_fp32_r1_tiled.cu | - | - | - | - | Old version, superseded |
| gemm_fp32_r1a.cu | 321 | 2% | 8 KB | 16 | Small tiles, too slow |
| gemm_fp32_r1b.cu | 56 | 12% | 32 KB | 32 | Outdated by r1x |
| gemm_fp32_r1c.cu | 45 | 14% | 64 KB | 43 | Superseded by r1x |
| gemm_fp32_r1d.cu | 45 | 14% | 64 KB | 43 | Identical to r1c |
| gemm_fp32_r1x2.cu | 43 | 15% | 64 KB | 43 | Identical to r1x |

## Active Kernels (in src/)

| Kernel | Time (ms) | vs CuBLAS | Block Type |
|--------|-----------|-----------|------------|
| gemm_fp32.cu | 67 | 10% | Naive baseline |
| gemm_fp32_r1x.cu | 43 | 15% | 2D blocks |
| gemm_fp32_r1y.cu | **27** | **24%** | 1D blocks (BEST) |

## Key Issues & Lessons Learned

### gemm_fp32_r1_tiled.cu
- **Issue**: Early tiled implementation with bugs
- **Lesson**: Replaced by cleaner r1a/r1b versions
- **Config**: Unknown (early development)

### gemm_fp32_r1a.cu (32×32 baseline)
- **Issue**: Small tiles (32×32) = low arithmetic intensity (AI=16)
- **Lesson**: Larger tiles improve performance up to SMEM limit
- **Config**: BM=32, BN=32, BK=32, TM=1, TN=1, Threads=256
- **Future Use**: Good for testing boundary conditions with small tiles

### gemm_fp32_r1b.cu (64×64 medium)
- **Issue**: Good baseline but surpassed by r1c
- **Lesson**: 128×64 gives better arithmetic intensity
- **Config**: BM=64, BN=64, BK=64, TM=4, TN=4, Threads=64
- **Future Use**: Reference for 64×64 tile implementation

### gemm_fp32_r1c.cu / r1d.cu (128×64 large tiles)
- **Issue**: Maxed out at 128×64 due to SMEM limit (48KB)
- **Lesson**: 128×128 would exceed SMEM limit (128KB needed)
- **Config**: BM=128, BN=64, BK=64, TM=4, TN=4, Threads=64
- **Future Use**: Reference for register blocking patterns

### gemm_fp32_r1x2.cu (SMEM Layout Experiment)
- **Issue**: Attempted SMEM layout change (Bs[BN][BK] instead of Bs[BK][BN])
- **Lesson**: Layout caused severe bank conflicts, performance dropped to 150ms (from 43ms)
- **Lesson**: Original Bs[BK][BN] layout was correct
- **Config**: Same as r1x but with transposed B SMEM layout
- **Future Use**: Example of how SMEM layout changes can hurt performance

## How to Re-enable a Scrapped Kernel

### Step 1: Copy file back to src/
```bash
cp scratch/gemm_fp32_r1a.cu src/
```

### Step 2: Add extern declaration to main.cu
Add after existing FP32 extern declarations:
```cpp
#if FP32_VARIANT == 8
extern void launch_gemm_fp32_r1a(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);
extern const char* get_variant_id_fp32_r1a();
extern const char* get_variant_desc_fp32_r1a();
#define launch_gemm_fp32 launch_gemm_fp32_r1a
#define get_variant_id_fp32 get_variant_id_fp32_r1a
#define get_variant_desc_fp32 get_variant_desc_fp32_r1a
#endif
```

### Step 3: Add variant case in main.cu print section
```cpp
#elif FP32_VARIANT == 8
    printf("# FP32_VARIANT: r1a (32x32 baseline)\n\n");
```

### Step 4: Update CMakeLists.txt
Add the file to the add_executable list:
```cmake
src/gemm_fp32_r1a.cu
```

### Step 5: Change FP32_VARIANT and build
```cpp
#define FP32_VARIANT 8
```
```bash
cmake --build build
```

## Experiments to Try (Future)

1. **r1a with different thread counts**: Test 32×32 with 1024 threads
2. **r1b with 1D blocks**: Try 64×64 with 1D structure
3. **r1c with double buffering**: If SMEM can be reduced
4. **Vectorized loads on any kernel**: Add float4 loads

## Architecture Comparison

| Aspect | 2D Blocks (r1x) | 1D Blocks (r1y) |
|--------|-------------------|------------------|
| Thread layout | dim3(16, 32) | dim3(1024) |
| Warp coalescing | Strided (stride 16) | Perfect (stride 1) |
| SMEM layout | Bs[BK][BN] | Bs[BK][BN] |
| Performance | 15% | **24%** |

**Key insight**: 1D blocks with perfect coalescing significantly outperform 2D blocks on RTX 5070.
