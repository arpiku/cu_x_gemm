# FP32 r1 Kernel Development: Complete Analysis

## Overview

The FP32 r1 tiled kernels were developed iteratively over multiple iterations. This document chronicles the mistakes made, their symptoms, the lessons learned, the variant comparison results, and future optimization strategies.

---

## Mistakes Summary

### Mistake 1: Incorrect Block Size (Critical, Runtime Error)

**Error:**
```cpp
dim3 block(tile_config::BN, tile_config::BM);  // 64, 64
```

**Problem:**
- `BM = 64`, `BN = 64`
- Total threads: `64 × 64 = 4096`
- **Max threads per block on RTX 5070: 1024**

**Symptom:**
```
CUDA error: invalid argument
```

**Fix:**
```cpp
constexpr int THREAD_M = 16;
constexpr int THREAD_N = 16;
dim3 block(THREAD_N, THREAD_M);  // 16, 16 = 256 threads
```

---

### Mistake 2: Register Blocking Indexing Bug

**Error:**
```cpp
for (int kk = 0; kk < BK; ++kk) {
    float a_reg = As[thread_row][kk];  // Same for all TM elements
    float b_reg = Bs[kk][thread_col];  // Same for all TN elements
    for (int m = 0; m < TM; ++m) {
        for (int n = 0; n < TN; ++n) {
            acc[m * TN + n] += a_reg * b_reg;  // WRONG!
        }
    }
}
```

**Problem:**
- `a_reg` depends only on `thread_row`, not on `m`
- `b_reg` depends only on `thread_col`, not on `n`
- **Result**: All TM×TN elements got identical values

**Symptom:**
```
Expected C[0][:8]:  5462016 5464032 5466048 5468064 5470080 5472096 5474112 5476128 
Custom C[0][:8]:   776.85  776.85  776.85  776.85  793.62  793.62  793.62  793.62 
```
All columns in a row had identical values.

**Correct Approach:**
```cpp
for (int m = 0; m < TM; m++) {
    int row_in_tile = thread_row * TM + m;
    for (int n = 0; n < TN; n++) {
        int col_in_tile = thread_col * TN + n;
        // Now use row_in_tile and col_in_tile for indexing
    }
}
```

---

### Mistake 3: Load Loop Stride = Block Dimension

**Error:**
```cpp
for (int load_row = 0; load_row < BM; load_row += blockDim.y) {
    int row = thread_row + load_row;
    As[row][thread_col] = A[global_row * K + bk + thread_col];
}
```

**Problem:**
- `blockDim.y = 64`
- `BM = 64`
- Loop: `load_row = 0; 0 < 64; load_row += 64` → **only 1 iteration**
- Only row 0-15 of As was populated (thread_row = 0-15)

**Correct Approach:**
```cpp
for (int i = thread_row; i < BM && tile_row + i < M; i += THREAD_M) {
    // Now each thread loads multiple rows if needed
}
```

---

### Mistake 4: Grid Dimension Calculation Error

**Error:**
```cpp
dim3 grid(
    (N + tile_config::BN * TN - 1) / (tile_config::BN * TN),
    (M + tile_config::BM * TM - 1) / (tile_config::BM * TM)
);
```

**Problem:**
- For N=256, TN=4: `(256 + 256 - 1) / 256 = 1` block in N dimension
- But each block covers only BN=64 columns
- **Result**: Grid too small, many output elements not covered

**Correct Calculation:**
```cpp
dim3 grid(
    (N + tile_config::BN - 1) / tile_config::BN,
    (M + tile_config::BM - 1) / tile_config::BM
);
```

---

### Mistake 5: Shared Memory Indexing for K Dimension

**Error:**
```cpp
As[thread_row][thread_col] = A[global_row * K + bk + thread_col];
Bs[thread_row][thread_col] = B[(bk + thread_row) * N + global_col];
```

**Correct Approach:**
```cpp
// A: row-major, thread loads along K dimension
As[row][kk - bk] = A[row * K + kk];

// B: column of B at position kk, needs B[kk * N + col]
Bs[kk - bk][col] = B[kk * N + col];
```

---

### Mistake 6: K Loop Boundary Condition

**Error:**
```cpp
for (int kk = 0; kk < BK; ++kk) {
    acc += As[thread_row][kk] * Bs[kk][thread_col];
}
```

**Problem:**
- When `K < BK` (e.g., K=32, BK=64), the loop runs 64 times
- But shared memory only has valid data for kk=0..31
- **Reading garbage data for kk=32..63**

**Fix:**
```cpp
int k_end = min(bk + BK, K);
for (int kk = 0; kk < k_end - bk; ++kk) {
    acc += As[thread_row][kk] * Bs[kk][thread_col];
}
```

---

### Mistake 7: Not Testing Incrementally

**Error:** Tried to implement tiling, register blocking, K boundary handling, and multiple tile sizes all at once.

**Lesson:** Implement and test one optimization at a time.

---

## Mistakes Summary Table

| Mistake | Category | Severity | Symptom |
|---------|----------|----------|---------|
| Block size 64×64 | Configuration | Critical | Runtime error |
| Register blocking index | Logic | Critical | All-same output |
| Loop stride = blockDim | Logic | Critical | Zeros in output |
| Grid calculation | Configuration | High | Partial output |
| SMEM K indexing | Logic | High | Wrong values |
| K boundary condition | Logic | High | L2 error at small sizes |
| Not testing incrementally | Process | Medium | Debug difficulty |

---

## Mathematical Framework

### Arithmetic Intensity Formula

The arithmetic intensity (AI) for a tiled GEMM kernel is:

```
AI = (BM × BN) / (BM + BN) FLOPs/byte
```

**Derivation:**
- **FLOPs per tile:** 2 × BM × BK × BN (multiply-accumulate for each element)
- **Bytes transferred:** 2 × BK × (BM + BN) (A and B tiles)
- **AI = FLOPs / Bytes** = (2 × BM × BK × BN) / (2 × BK × (BM + BN)) = (BM × BN) / (BM + BN)

### Roofline Model Position

To be compute-bound, AI must exceed the hardware's balance point:

```
Balance Point = Peak Compute / Memory Bandwidth
```

| GPU | Peak FP32 | Memory BW | Balance Point |
|-----|-----------|-----------|---------------|
| RTX 5070 (est.) | ~30 TFLOPS | ~1000 GB/s | ~30 FLOPs/byte |
| H100 | ~67 TFLOPS | ~3350 GB/s | ~20 FLOPs/byte |

### Shared Memory Calculation

```
SMEM per block = 2 × BM × BK × sizeof(float)
```

With `sizeof(float) = 4 bytes`:
```
SMEM = 8 × BM × BK bytes
```

### Occupancy Analysis

Occupancy is limited by shared memory, registers, and threads:

```
Blocks per SM = min(
    SMEM_total / SMEM_per_block,
    max_threads / threads_per_block,
    max_registers / (registers_per_thread)
)
```

---

## Variant Comparison Results

> **Note**: Archived variants (r1a, r1b, r1c, r1d, r1x2) are in `scratch/`. See `scratch/SCRATCH_INDEX.md` for details.

### Active Variant Configurations

| Variant | BM | BN | BK | TM | TN | Threads | Block Type | SMEM | AI | Notes |
|---------|----|----|----|----|----|---------|------------|------|----|-------|
| naive | - | - | - | - | - | 16×16 | 2D | - | - | Baseline |
| r1x | 128 | 64 | 64 | 4 | 4 | 512 | 2D | 64 KB | 43 | Good reference |
| **r1y** | 64 | 64 | 64 | 2 | 2 | 1024 | **1D** | 32 KB | 32 | **Best** |

### Performance Results at 4096×4096

| Variant | Time (ms) | vs Naive | vs CuBLAS | AI | SMEM | Notes |
|---------|-----------|----------|-----------|----|------|-------|
| naive | 67 | 1.0× | 10% | - | - | Baseline |
| r1x | 43 | 1.6× | 15% | 43 | 64 KB | 2D blocks |
| **r1y** | **27** | **2.5×** | **24%** | 32 | 32 KB | **1D blocks (BEST)** |
| CuBLAS | 6.5 | 10× | 100% | - | - | Target |

### Key Findings

1. **1D blocks > 2D blocks**: r1y (1D) is 37% faster than r1x (2D)
2. **Perfect coalescing**: 1D blocks enable warp-level coalescing
3. **All variants correct**: L2 error ~2.6e-04

### Architecture Comparison: 2D vs 1D

| Aspect | 2D Blocks (r1x) | 1D Blocks (r1y) |
|--------|-------------------|------------------|
| Thread layout | dim3(16, 32) | dim3(1024) |
| Warp coalescing | Strided (stride 16) | Perfect (stride 1) |
| SMEM layout | Bs[BK][BN] | Bs[BK][BN] |
| Performance | 15% of CuBLAS | **24% of CuBLAS** |

### SMEM Constraints

| Tile Size | SMEM Calculation | SMEM Used | Fits 48KB? |
|-----------|-----------------|-----------|-------------|
| 32×32 | 2 × 32 × 32 × 4B | 8 KB | ✓ Yes |
| 64×64 | 2 × 64 × 64 × 4B | 32 KB | ✓ Yes |
| 128×64 | 2 × 128 × 64 × 4B | 64 KB | ✓ Yes |
| 128×128 | 2 × 128 × 128 × 4B | 128 KB | ✗ No (exceeds limit) |

---

## Potential Optimizations

### Micro-Optimizations (No Renaming Required)

These can be added to existing variants without changing the variant naming scheme.

#### 1. LDG Intrinsics

Read-only cache hint for global memory loads.

```cpp
// Global memory reads with cache hint
As[i][j] = __ldg(&A[index]);
Bs[kk - bk][j] = __ldg(&B[kk * N + col]);
```

**Expected Impact:** ~5-10% improvement

#### 2. Vectorized Loads (float4)

128-bit memory transactions vs 32-bit loads.

```cpp
// Load 4 floats at once using float4
float4 tmp = __ldg(reinterpret_cast<const float4*>(&A[index]));
```

**Expected Impact:** ~10-15% improvement

**Implementation Strategy:**
- For A tile: Load row-major → natural vectorization
- For B tile: Load row-by-row then transpose in shared memory

#### 3. Const Correctness

Compiler hints for read-only data.

```cpp
__global__ void kernel(
    const float* __restrict__ const A,  // A is read-only
    const float* __restrict__ const B,  // B is read-only
    float* __restrict__ C,
    ...
)
```

**Expected Impact:** Minor (compiler optimization)

#### 4. Bank Conflict Reduction

Shared memory padding to avoid bank conflicts.

```cpp
// With padding to avoid 64-stride bank conflicts
__shared__ float As[BM][BK + 1];  // +1 column padding
__shared__ float Bs[BK][BN + 1];
```

**Expected Impact:** ~5% improvement

#### 5. Loop Unrolling

Explicit unroll hints for fixed-size loops.

```cpp
#pragma unroll 8
for (int kk = 0; kk < k_iterations; ++kk) { ... }
```

**Expected Impact:** Minimal (-O3 already auto-unrolls)

---

### Advanced Optimizations (r2+)

These change the kernel architecture enough to warrant renaming.

#### 1. Double Buffering

Hide memory latency by overlapping loads with compute.

- Uses 2× shared memory for A and B tiles
- While computing on tile N, load tile N+1
- Requires careful synchronization

#### 2. Warp-Level Tiling

Threads within a warp cooperate on loads.

- Reduces redundant memory accesses
- More complex synchronization

#### 3. Async Memory Operations

`cuda::memcpy_async` for asynchronous data movement.

- Overlaps data transfer with compute
- Requires CUDA 11+ features

#### 4. Tensor Core Operations

WMMA API for BF16/FP16.

- Separate file for TC kernels
- Not applicable for FP32 CUDA-core path

---

## Next Steps

### Immediate: r1x Variants

- **r1cx**: Based on r1c (128×64), add micro-optimizations
- **r1dx**: Based on r1d (128×64), add micro-optimizations

**Implementation Order:**
1. LDG intrinsics + const correctness ✓ (r1x)
2. Bank conflict padding ✗ (SMEM exceeded limit)
3. Double buffering ✗ (SMEM exceeded limit with BK=32)
4. Warp tiling ✗ (complex, incorrect implementation)
5. Vectorized loads (planned)

### Future: r2 Variants

1. Double buffering with reduced BM (e.g., BM=64)
2. Warp tiling (proper implementation)
3. Autotuning framework
4. Tensor Core kernels (separate file)

## Micro-Optimization Test Results

### What Worked
- **LDG intrinsics**: ~4% improvement (43ms vs 45ms at 4096)

### What Didn't Work
- **Bank padding (PAD=1 with BM=128, BK=64)**: SMEM = 49664 bytes exceeds 49152 byte limit
- **Double buffering**: Requires 2× SMEM, tested with BK=32 (74ms - slower due to more K iterations)
- **BM=96 with padding**: 55ms - slower due to reduced tile efficiency
- **Transposed SMEM layout (Bs[BN][BK])**: 150ms - severe bank conflicts

### Conclusions
- SMEM is the limiting factor for RTX 5070 (48KB per block)
- Micro-optimizations provide marginal gains (<5%)
- Need more fundamental architectural changes for significant improvement
- Double buffering requires either smaller tiles or different architecture
- SMEM layout changes can cause severe bank conflicts

---

## r1x2 Development: Memory Coalescing Investigation

### Goal
Implement proper warp-level memory coalescing for B-matrix loads.

### Approach Tested
Changed SMEM layout from `Bs[BK][BN]` to `Bs[BN][BK]` to enable:
- Threads loading consecutive columns
- Better warp-level coalescing

### Result
**FAILED** - Performance dropped from 43ms to 150ms at 4096×4096.

### Root Cause
- Bs[BN][BK] layout causes severe shared memory bank conflicts
- Threads accessing consecutive columns hit same banks
- Better to keep original Bs[BK][BN] layout

### Lesson Learned
Simple SMEM layout changes can have significant negative impact.
Original r1x kernel already has effective memory access patterns.

### Current Status (r1x2)
- r1x2 is currently identical to r1x (baseline with LDG)
- Coalescing attempts did not improve performance
- Need to explore other optimization directions

---

## r1y Development: 1D Block Structure (Next Phase)

### Goal
Convert from 2D blocks to 1D blocks for better memory coalescing, matching article's architecture.

### Hypothesis
The article achieves 95% of cuBLAS partly due to:
1. 1D blocks with perfect warp-level coalescing
2. Natural support for vectorized (float4) loads
3. Square 64×64 tiles

### Key Differences: 1D vs 2D Blocks

| Aspect | 2D Block (r1x) | 1D Block (r1y) |
|--------|-----------------|-----------------|
| Thread layout | dim3(16, 32) | dim3(1024) |
| Warp composition | threadIdx.x varies 0-15 | threadIdx.x varies 0-1023 |
| GMEM coalescing | Strided (stride 16) | Perfect (stride 1) |
| float4 support | Complex | Natural |

### Configuration
- BM = 64, BN = 64 (square tiles)
- BK = 64
- 1024 threads per block (1D)
- SMEM = 32KB
- AI = 32 FLOPs/byte

### Results

| Kernel | Time (ms) | vs CuBLAS | Improvement |
|--------|-----------|-----------|-------------|
| r1x (2D) | 43 | 15% | baseline |
| **r1y (1D)** | **27** | **24%** | **+37% faster** |

### Conclusion
1D blocks with 1024 threads significantly improved performance:
- 37% faster than r1x
- 24% of CuBLAS vs 15% previously
- Validates the hypothesis that warp-level coalescing matters

### Implementation
See `src/gemm_fp32_r1y.cu`

---

## Future Optimizations (Next Phase)

Based on analysis and article study, the following optimizations are planned:

### 1. Async Memory Transfer (cuda::memcpy_async)
- **Description**: Overlap data transfer with computation using asynchronous memory operations
- **Benefit**: Hides memory latency without increasing SMEM usage
- **Implementation**: Use CUDA's `cuda::memcpy_async` with pipeline API
- **Challenge**: Requires careful synchronization but doesn't increase SMEM footprint
- **Reference**: NVIDIA CUDA async memory documentation

### 2. Pipeline Intrinsics
- **Description**: Hardware-level async operations using warp-level intrinsics
- **Benefit**: Direct hardware support for memory/compute overlap
- **Implementation**: Use `__pipeline_*` intrinsics
- **Challenge**: Complex synchronization requirements

### 3. Slightly Reduced BM for Double Buffering
- **Description**: Reduce BM from 128 to 96 or 64 to enable double buffering within SMEM limit
- **Benefit**: Overlaps loads with compute, better latency hiding
- **Trade-off**: Reduced arithmetic intensity vs better pipelining
- **Calculation**: With BM=64, BK=64: 2×64×64×4 = 32KB SMEM (fits with room)

### 4. Vectorized Loads (float4)
- **Description**: Load 4 floats at once using 128-bit transactions
- **Benefit**: ~10-15% improvement (from article)
- **Implementation**: Use `float4` with `__ldg()`
- **Challenge**: Must ensure alignment, may require layout changes

### 5. Warp-Level Cooperative Loading
- **Description**: Threads in warp cooperatively load data
- **Benefit**: Better memory transaction efficiency
- **Implementation**: Use warp shuffle and reduction primitives
- **Challenge**: Complex coordination, must balance with compute

---

## Thread Scaling Experiments (1024 Thread Investigation)

### Goal
Match article's performance by increasing threads per block from 512 to 1024.

### Experiments Conducted

| Config | BM | BN | BK | TM | TN | Threads | SMEM | 4096×4096 | vs r1x |
|--------|----|----|----|----|----|---------|------|-----------|--------|
| r1x (baseline) | 128 | 64 | 64 | 4 | 4 | 512 | 64KB | 43ms | 1.0× |
| TM=2, TN=4 | 128 | 64 | 64 | 2 | 4 | 1024 | 64KB | 67ms | 0.64× |
| BN=128 | 128 | 128 | 64 | 4 | 4 | 1024 | 128KB | SMEM ERROR | - |
| BM=256, BN=64 | 256 | 64 | 32 | 4 | 4 | 1024 | 40KB | ZEROS | - |
| TM=4, TN=2 | 128 | 64 | 64 | 4 | 2 | 1024 | 64KB | 68ms | 0.63× |

### Key Findings

1. **More threads ≠ better performance**
   - Increasing from 512 to 1024 threads consistently **decreased** performance by ~60%
   - Suggests kernel is already well-optimized for current thread count

2. **Compute-to-Load Ratio Matters**
   - TN=4: 16 outputs / 8 loads = 2.0 FLOPs/load
   - TN=2: 8 outputs / 6 loads = 1.33 FLOPs/load
   - Lower ratio = more memory bound

3. **Article's Success Factors (Hypothesized)**
   - Different tile dimensions (BLOCKSIZE=64 for all)
   - Cooperative warp-level loading
   - Different GPU architecture (A6000 vs RTX 5070)
   - Warp tiling (step 10 in article)

### Conclusion
- RTX 5070 performs optimally with 512 threads (16×32 block)
- The bottleneck is NOT thread count but other factors:
  - Memory access patterns
  - SMEM bank conflicts
  - Lack of double buffering
  - Vectorized loads

### Recommended Next Steps (Priority Order)
1. **Vectorized loads (float4)** - Clean addition, ~10-15% expected improvement
2. **Double buffering** - Requires SMEM reduction (BM=64 or BK=32)
3. **Profile with NCU** - Confirm actual bottlenecks
4. **Warp tiling** - Article's key optimization (78%→94%)

### Why Not More Threads?
The RTX 5070 kernel is already well-optimized at 512 threads. Additional threads introduce:
- Higher SMEM contention
- Lower compute-to-load ratio (when TN reduced)
- Diminishing returns on latency hiding

**Key insight**: The article's A6000 (data center GPU) may tolerate more threads due to better SMEM bandwidth and different architecture.

---

## Appendix: Source Files

### Active Kernels (src/)

| File | Variant | Performance | Description |
|------|---------|------------|-------------|
| `gemm_fp32.cu` | naive | 10% | Baseline naive kernel |
| `gemm_fp32_r1x.cu` | r1x | 15% | 2D blocks with LDG |
| `gemm_fp32_r1y.cu` | r1y | **24%** | 1D blocks (BEST) |
| `gemm_bf16.cu` | bf16 | varies | BF16 naive kernel |

### Archived Kernels (scratch/)

| File | Performance | Reason |
|------|------------|--------|
| `gemm_fp32_r1_tiled.cu` | - | Old version |
| `gemm_fp32_r1a.cu` | 2% | Small tiles |
| `gemm_fp32_r1b.cu` | 12% | Outdated |
| `gemm_fp32_r1c.cu` | 14% | Replaced by r1x |
| `gemm_fp32_r1d.cu` | 14% | Identical to r1c |
| `gemm_fp32_r1x2.cu` | 15% | Identical to r1x |

See `scratch/SCRATCH_INDEX.md` for details on how to re-enable.

---

## Appendix: Useful Formulas

### Grid Size Calculation
```cpp
grid_x = (N + BN - 1) / BN
grid_y = (M + BM - 1) / BM
```

### Thread Block Size
```cpp
block_x = BN / TN
block_y = BM / TM
threads_per_block = block_x × block_y
```

### Shared Memory Size
```cpp
SMEM_bytes = 2 × BM × BK × sizeof(float)
```

### Arithmetic Intensity
```cpp
AI = (BM × BN) / (BM + BN)  // FLOPs/byte
```

### Effective FLOPs/second
```cpp
// For memory-bound kernel
effective_flops = memory_bandwidth × AI

// For compute-bound kernel
effective_flops = peak_flops
```

### Roofline Ridge Point
```cpp
ridge_point = peak_flops / memory_bandwidth
```
