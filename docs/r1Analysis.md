# FP32 r1 Kernel Development: Mistake Analysis

## Overview

The FP32 r1 tiled kernel (`src/gemm_fp32_r1_tiled.cu`) was developed iteratively over multiple iterations. This document chronicles the mistakes made, their symptoms, and the lessons learned.

---

## Mistake 1: Incorrect Block Size (Critical, Runtime Error)

### Error
```cpp
dim3 block(tile_config::BN, tile_config::BM);  // 64, 64
```

### Problem
- `BM = 64`, `BN = 64`
- Total threads: `64 × 64 = 4096`
- **Max threads per block on RTX 5070: 1024**

### Symptom
```
CUDA error: invalid argument
```

### Lesson
Always verify block dimensions against `cudaDeviceProp::maxThreadsPerBlock`. For this GPU:
- Max threads: 1024
- Max thread dimensions: 1024 × 1024 × 64

### Fix
```cpp
constexpr int THREAD_M = 16;
constexpr int THREAD_N = 16;
dim3 block(THREAD_N, THREAD_M);  // 16, 16 = 256 threads
```

---

## Mistake 2: Register Blocking Indexing Bug

### Error (First Attempt)
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

### Problem
- `a_reg` depends only on `thread_row`, not on `m`
- `b_reg` depends only on `thread_col`, not on `n`
- **Result**: All TM×TN elements got identical values

### Symptom
```
Expected C[0][:8]:  5462016 5464032 5466048 5468064 5470080 5472096 5474112 5476128 
Custom C[0][:8]:   776.85  776.85  776.85  776.85  793.62  793.62  793.62  793.62 
```
All columns in a row had identical values.

### Root Cause
For a thread computing multiple output elements, each element `(m, n)` should use different A and B values:
- Element `(m, n)` needs `A[row + m][k]` and `B[k][col + n]`
- But code used `A[thread_row][k]` and `B[k][thread_col]` for all elements

### Correct Approach
```cpp
for (int m = 0; m < TM; m++) {
    int row_in_tile = thread_row * TM + m;
    for (int n = 0; n < TN; n++) {
        int col_in_tile = thread_col * TN + n;
        // Now use row_in_tile and col_in_tile for indexing
    }
}
```

### Lesson
When implementing register blocking, each output element `(m, n)` must access different A and B tiles. The indexing must account for the thread's tile position AND its position within that tile.

---

## Mistake 3: Load Loop Stride = Block Dimension

### Error (First Attempt)
```cpp
for (int load_row = 0; load_row < BM; load_row += blockDim.y) {
    int row = thread_row + load_row;
    As[row][thread_col] = A[global_row * K + bk + thread_col];
}
```

### Problem
- `blockDim.y = 64`
- `BM = 64`
- Loop: `load_row = 0; 0 < 64; load_row += 64` → **only 1 iteration**
- Only row 0-15 of As was populated (thread_row = 0-15)

### Symptom
- Output was mostly zeros for rows > 16
- L2 error ~1.0 (completely wrong)

### Correct Approach
```cpp
for (int i = thread_row; i < BM && tile_row + i < M; i += THREAD_M) {
    // Now each thread loads multiple rows if needed
}
```

### Lesson
The loop stride must be smaller than the dimension being iterated. Use `THREAD_M` (number of threads in M dimension) instead of `blockDim.y`.

---

## Mistake 4: Grid Dimension Calculation Error

### Error (First Attempt)
```cpp
dim3 grid(
    (N + tile_config::BN * TN - 1) / (tile_config::BN * TN),
    (M + tile_config::BM * TM - 1) / (tile_config::BM * TM)
);
```

### Problem
- For N=256, TN=4: `(256 + 256 - 1) / 256 = 1` block in N dimension
- But each block covers only BN=64 columns
- **Result**: Grid too small, many output elements not covered

### Correct Calculation
```cpp
dim3 grid(
    (N + tile_config::BN - 1) / tile_config::BN,
    (M + tile_config::BM - 1) / tile_config::BM
);
```

### Lesson
Grid dimensions should cover the output matrix based on tile size (BM, BN), not per-thread output size (TM, TN).

---

## Mistake 5: Shared Memory Indexing for K Dimension

### Error
```cpp
As[thread_row][thread_col] = A[global_row * K + bk + thread_col];
Bs[thread_row][thread_col] = B[(bk + thread_row) * N + global_col];
```

### Problem
- `As[thread_row][thread_col]` writes to wrong position
- The K dimension offset (`bk`) needs to be subtracted for shared memory storage
- But the access pattern for A (along K) and B (along K) are different

### Correct Approach
```cpp
// A: row-major, thread loads along K dimension
As[row][kk - bk] = A[row * K + kk];

// B: column of B at position kk, needs B[kk * N + col]
Bs[kk - bk][col] = B[kk * N + col];
```

### Lesson
When storing to shared memory, the K offset must be rebased to start at 0. The global K position (`bk + kk`) becomes local position (`kk - bk`).

---

## Mistake 6: K Loop Boundary Condition

### Error
```cpp
for (int kk = 0; kk < BK; ++kk) {
    acc += As[thread_row][kk] * Bs[kk][thread_col];
}
```

### Problem
- When `K < BK` (e.g., K=32, BK=64), the loop runs 64 times
- But shared memory only has valid data for kk=0..31
- **Reading garbage data for kk=32..63**

### Symptom
- L2 error ~1.0 at small matrix sizes (32×32, 64×64)
- Correct at large sizes (K=BK=64)

### Fix
```cpp
int k_end = min(bk + BK, K);
for (int bk = 0; bk < K; bk += BK) {
    int k_end = min(bk + BK, K);
    // ... load only up to k_end ...
    for (int kk = 0; kk < k_end - bk; ++kk) {
        acc += As[thread_row][kk] * Bs[kk][thread_col];
    }
}
```

### Lesson
Always bound loops by the actual data size, not just the tile size.

---

## Mistake 7: Not Testing Incrementally

### Error
Tried to implement:
1. Tiled memory access
2. Register blocking
3. K boundary handling
4. Multiple tile sizes

All at once.

### Problem
When results were wrong, couldn't identify which change caused the issue.

### Lesson
Implement and test one optimization at a time:
1. First: Naive kernel (verify baseline)
2. Then: Basic tiled kernel (TM=TN=1)
3. Then: Register blocking
4. Then: K boundary handling
5. Then: Performance tuning

---

## Summary Table

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

## Final Working Configuration

```cpp
namespace tile_config {
    constexpr int BM = 64;           // Output tile rows
    constexpr int BN = 64;           // Output tile cols
    constexpr int BK = 64;           // K-dimension tile
    constexpr int THREAD_TILE_M = 4; // Elements per thread (M)
    constexpr int THREAD_TILE_N = 4; // Elements per thread (N)
}
// 16×16 = 256 threads per block
// Each thread computes 4×4 = 16 output elements
// Total: 64×64 = 4096 elements per block ✓
```

---

## Performance Results

| Size | r1 Tiled | Naive | CuBLAS | % of CuBLAS |
|------|-----------|-------|--------|-------------|
| 32 | 0.006ms | 0.0007ms | 0.005ms | 80% |
| 64 | 0.015ms | 0.002ms | 0.005ms | 31% |
| 256 | 0.058ms | 0.021ms | 0.007ms | 13% |
| 1024 | 0.78ms | 1.05ms | 0.11ms | 14% |
| 4096 | 57.7ms | 67.9ms | 6.7ms | 11% |

**L2 Error**: ~2.6e-04 (correct)

**Observation**: Performance decreases as size increases, indicating memory-bound behavior. The tiled kernel helps but more optimization needed.

---

## Next Steps (Based on Lessons)

1. **Test with TM=TN=1 first** - verify basic tiling works
2. **Add register blocking incrementally** - verify each change
3. **Profile to identify bottleneck** - is it memory or compute bound?
4. **Consider larger tiles** - 128×64 or 128×128 for higher arithmetic intensity
