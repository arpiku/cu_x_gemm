# Strassen's Matrix Multiplication on H100: Design Document

## Overview

Strassen's algorithm reduces the complexity of matrix multiplication from O(n^3) to O(n^2.807) through recursive decomposition. This document outlines the implementation strategy for NVIDIA H100 GPUs targeting square matrices of dimensions 1024, 2048, 4096, and 8192 with BF16 and FP32 data types.

## Algorithm Foundation

Standard matrix multiplication C = A × B requires 8 multiplications for 2×2 blocks. Strassen reduces this to 7 multiplications:

```
M1 = (A11 + A22) × (B11 + B22)
M2 = (A21 + A22) × B11
M3 = A11 × (B12 - B22)
M4 = A22 × (B21 - B11)
M5 = (A11 + A12) × B22
M6 = (A21 - A11) × (B11 + B12)
M7 = (A12 - A22) × (B21 + B22)

C11 = M1 + M4 - M5 + M7
C12 = M3 + M5
C21 = M2 + M4
C22 = M1 - M2 + M3 + M6
```

## Implementation Levels

### Level 0: Base Case (Standard GEMM)
- **Threshold**: 512×512 or smaller
- **Method**: Use optimized WMMA/Tensor Core kernel from Part 1
- **Rationale**: Below this size, overhead of Strassen decomposition exceeds savings

### Level 1: Single Recursion
- **Matrix Split**: Divide into 4 submatrices of size N/2 × N/2
- **Operations**:
  - 10 matrix additions (A/B submatrix combinations)
  - 7 matrix multiplications (M1-M7)
  - 8 matrix additions (C submatrix assembly)
- **Memory**: Requires temporary storage for M1-M7 intermediates

**Pseudocode**:
```cpp
void strassen_level1(A, B, C, N) {
    half_t A11[N/2][N/2], A12[N/2][N/2], A21[N/2][N/2], A22[N/2][N/2];
    // Split A, B into quadrants
    
    // Compute intermediates (parallel on GPU)
    S1 = A11 + A22; T1 = B11 + B22;
    S2 = A21 + A22; T2 = B21 - B11;
    S3 = A11 + A12; T3 = B12 - B22;
    S4 = A21 - A11; T4 = B11 + B12;
    S5 = A12 - A22; T5 = B21 + B22;
    
    // 7 GEMM calls (can be batched)
    M1 = GEMM(S1, T1);
    M2 = GEMM(S2, B11);
    M3 = GEMM(A11, T3);
    M4 = GEMM(A22, T2);
    M5 = GEMM(S3, B22);
    M6 = GEMM(S4, T4);
    M7 = GEMM(S5, T5);
    
    // Assemble C
    C11 = M1 + M4 - M5 + M7;
    C12 = M3 + M5;
    C21 = M2 + M4;
    C22 = M1 - M2 + M3 + M6;
}
```

### Level 2+: Multiple Recursions
- **Dimensions**: N ≥ 2048 benefits from 2 levels; N ≥ 4096 benefits from 3 levels
- **Submatrix sizes after k recursions**: N / 2^k
- **Number of multiplications**: 7^k (e.g., 49 for 2 levels, 343 for 3 levels)

**Recursive Structure**:
```
Level 2 (N=4096 → 1024 base case):
  49 multiplications of 1024×1024 matrices
  18× more additions than Level 1

Level 3 (N=8192 → 1024 base case):
  343 multiplications of 1024×1024 matrices
  Significant memory pressure
```

## Memory Management

### Memory Requirements (BF16, N=4096)

| Component | Level 0 | Level 1 | Level 2 |
|-----------|---------|---------|---------|
| Input matrices | 64 MB | 64 MB | 64 MB |
| Output matrix | 32 MB | 32 MB | 32 MB |
| M intermediates | 0 | 224 MB | 1.5 GB |
| Add/sub temps | 0 | 32 MB | 256 MB |
| **Total** | ~100 MB | ~350 MB | ~1.8 GB |

### H100 Memory (80 GB HBM3)
- Sufficient for N=8192 with 2 recursion levels
- 3+ levels may require out-of-core techniques or recomputation

### Optimization Strategies
1. **In-place additions**: Use output buffer for intermediate sums
2. **Buffer reuse**: M1-M7 buffers reused across recursion levels
3. **Streamed execution**: Pipeline M computation with C assembly
4. **Packed layout**: Store submatrices contiguously to improve cache locality

## Kernel Design

### Addition/Subtraction Kernels
```cpp
__global__ void matrix_add_bf16(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    size_t N)
{
    // Vectorized load/store for bandwidth efficiency
    // Use float4 or uint4 for coalesced access
}
```

### Batched GEMM
- Use `cublasGemmStridedBatchedEx` for M1-M7
- Single kernel launch for all 7 multiplications
- Enables GPU-wide parallelism

### Fusion Opportunities
```cpp
// Fused: S = A + B; followed by GEMM(S, ...)
// Avoids write-back to global memory
__global__ void add_gemm_fused(...)
```

## Challenges and Bottlenecks

### 1. Memory Bandwidth
- **Issue**: Strassen trades computation for memory operations
- **Impact**: On memory-bound small matrices, Strassen may be slower
- **Mitigation**: Fuse operations, use shared memory for submatrices

### 2. Numerical Stability
- **Issue**: BF16 limited precision; cancellation errors in additions
- **Impact**: Results may diverge from standard GEMM
- **Mitigation**: Use FP32 accumulation, careful subtraction ordering

### 3. Load Balancing
- **Issue**: 7 multiplications have different operand sizes after padding
- **Impact**: SM underutilization
- **Mitigation**: Dynamic parallelism or multi-stream execution

### 4. Workspace Allocation
- **Issue**: Large temporary buffers needed
- **Impact**: Memory fragmentation, allocation overhead
- **Mitigation**: Pre-allocated workspace pool, arena allocator

### 5. Non-Power-of-2 Dimensions
- **Issue**: Strassen requires even dimensions
- **Impact**: Padding overhead
- **Mitigation**: Hybrid approach - pad to next power of 2 or use standard GEMM for remainder

## Optimization Techniques

### 1. Tiled Storage
```
Store submatrices in tiled layout matching GEMM kernel
Reduces strided access during split/merge operations
```

### 2. Asynchronous Execution
```cpp
cudaStream_t streams[7];
// Launch each M computation in separate stream
// Overlap computation with memory transfers
```

### 3. Kernel Fusion
- Fuse S/T computation with GEMM input loading
- Fuse C assembly with GEMM output storage
- Reduces global memory traffic by ~40%

### 4. Shared Memory Workspace
- Use shared memory for intermediate sums within a CTA
- Reduces global memory pressure

### 5. Adaptive Recursion Depth
```cpp
int compute_recursion_depth(int N) {
    if (N <= 512) return 0;
    if (N <= 2048) return 1;
    if (N <= 4096) return 2;
    return 3; // For N=8192+
}
```

## Performance Estimates

### Theoretical Speedup
- Level 1: 7/8 multiplications = 12.5% fewer operations
- Level 2: 49/64 multiplications = 23.4% fewer operations
- Practical speedup: 5-15% due to overhead

### Expected TFLOPS (BF16 on H100)

| Dimension | Standard GEMM | Strassen L1 | Strassen L2 |
|-----------|---------------|-------------|-------------|
| 1024 | ~350 | ~320 | N/A |
| 2048 | ~450 | ~420 | ~380 |
| 4096 | ~500 | ~490 | ~460 |
| 8192 | ~520 | ~510 | ~500 |

## Implementation Timeline

1. **Phase 1**: Single-level Strassen with CuBLAS batched GEMM
2. **Phase 2**: Custom kernels for add/sub operations
3. **Phase 3**: Recursive implementation with depth selection
4. **Phase 4**: Optimization (fusion, streaming, memory pooling)

## Conclusion

Strassen's algorithm on H100 requires careful memory management and kernel optimization. The main benefit appears at larger dimensions (N≥4096) where the reduced operation count outweighs memory overhead. A hybrid approach—using standard GEMM for smaller tiles and Strassen at higher levels—provides the best balance of performance and practicality.
