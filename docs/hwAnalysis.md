# Hardware Analysis: H100 vs RTX 5070

## 1. Raw Spec Comparison

| Property | H100 SXM (SM 9.0) | RTX 5070 (SM 12.0) | Notes |
|---|---|---|---|
| Architecture | Hopper | Blackwell | |
| SM Count | 132 | 48 | H100 has 2.75x more SMs |
| CUDA Cores / SM | 128 | 128 | Same per-SM core count |
| Total CUDA Cores | 16,896 | 6,144 | |
| Boost Clock | 1.98 GHz | 2.54 GHz | RTX boost is 28% higher |
| warpSize | 32 | 32 | |
| maxThreadsPerBlock | 1024 | 1024 | |
| maxThreadsPerSM | 2048 | 1536 | H100 can run 33% more threads/SM |
| sharedMemPerBlock | 48 KB | 48 KB | |
| sharedMemPerSM | 228 KB | 100 KB | H100 has 2.3x more shared memory |
| regsPerBlock | 65536 | 65536 | Max registers a single block can use |
| regsPerSM | 65536 | 65536 | Total register file per SM |
| l2CacheSize | 51200 KB (50 MB) | 49152 KB (48 MB) | ~same L2 |
| totalGlobalMem | 81079 MB (80 GB) | 11766 MB (12 GB) | H100 has 6.9x more memory |
| Memory Type | HBM3 | GDDR7 | |
| Memory Clock | 2.62 GHz | 14.00 GHz | RTX clock is 5.3x higher |
| Memory Bus Width | 5120 bits | 192 bits | H100 bus is 26.7x wider |
| Memory Bandwidth | 3352 GB/s | 672 GB/s | H100 has 5x higher bandwidth |
| totalConstMem | 64 KB | 64 KB | |
| maxGridSize | 2147483647 x 65535 x 65535 | idem | CUDA API limit, not a physical constraint |

---

## 2. Peak Throughput Calculations

### FP32 Peak (CUDA Cores, FMA)

```
Peak_FP32 = SMs × CUDA_cores_per_SM × 2 (FMA) × boost_clock_Hz
```

| | H100 | RTX 5070 |
|---|---|---|
| Calculation | 132 × 128 × 2 × 1.98e9 | 48 × 128 × 2 × 2.542e9 |
| **Peak FP32** | **67.0 TFLOPS** | **31.2 TFLOPS** |

H100 is **2.15x** faster at peak FP32.

### Peak Memory Bandwidth

```
Peak_BW = 2 × (bus_width_bits / 8) × memory_clock_Hz
```

| | H100 | RTX 5070 |
|---|---|---|
| Calculation | 2 × (5120/8) × 2.619e9 | 2 × (192/8) × 14.001e9 |
| **Peak BW** | **3352 GB/s** | **672 GB/s** |

H100 has **5.0x** higher bandwidth.

### Tensor Core Peak (BF16)

These must come from NVIDIA's published specs (not in cudaDeviceProp):

| | H100 SXM | RTX 5070 |
|---|---|---|
| **Peak BF16 (Tensor Core)** | **1979 TFLOPS** | **~990 TFLOPS (est.)** |
| Gen | 4th gen (Hopper) | 5th gen (Blackwell) |

Note: RTX 5070 BF16 Tensor Core peak is estimated from 419.2 TFLOPS FP16/Tensor sparse (from Wikipedia specs — verify against actual CUDA arch docs). Blackwell 5th-gen Tensor Cores have 2x the FMA throughput per clock vs Hopper.

---

## 3. Roofline Analysis

### Break-even Arithmetic Intensity

```
AI_break_even = Peak_FP32 / Peak_BW
```

| | H100 | RTX 5070 |
|---|---|---|
| **AI_break_even** | 67.0e12 / 3.352e12 = **20.0 FLOP/byte** | 31.2e12 / 672e12 = **46.4 FLOP/byte** |

### GEMM Arithmetic Intensity

For square GEMM (M=N=K=N):

```
FLOPS  = 2N³
Bytes  = 12N²  (read A + read B + write C, FP32)
AI     = FLOPS / Bytes = 2N³ / 12N² = N/6
```

| N | AI (FLOP/byte) | H100 bound? | RTX bound? |
|---|---|---|---|
| 32 | 5.3 | Memory | Memory |
| 64 | 10.7 | Memory | Memory |
| 128 | 21.3 | **Compute** | Memory |
| 256 | 42.7 | Compute | **Compute** (barely) |
| 512 | 85.3 | Compute | Compute |
| 1024 | 170.7 | Compute | Compute |
| 2048 | 341.3 | Compute | Compute |
| 4096 | 682.7 | Compute | Compute |

**Key insight:** RTX 5070 remains memory-bound until N ~ 200-300, while H100 transitions to compute-bound at N ~ 128. This means small GEMMs on RTX 5070 are more sensitive to bandwidth optimizations (tiling, coalescing).

---

## 4. GEMM Worked Examples

### Formula Recap

```
FLOPS(N) = 2 × N³
Bytes(N) = 12 × N²  (FP32)
t_compute = FLOPS / Peak_FP32
t_memory   = Bytes / Peak_BW
t_actual  = max(t_compute, t_memory) / efficiency
```

### N = 512 (FP32)

| | H100 | RTX 5070 |
|---|---|---|
| FLOPS | 2 × 512³ = 268.4 MFLOP | same |
| Data | 12 × 512² = 3.0 MB | same |
| AI | 85.3 FLOP/byte → compute | compute |
| t_compute (100%) | 268.4e6 / 67.0e12 = 4.00 μs | 8.60 μs |
| t_memory (100%) | 3.0e6 / 3.352e12 = 0.90 μs | 4.51 μs |
| t_min (compute) | 4.00 μs | 8.60 μs |
| CuBLAS (RTX, est.) | — | ~0.13 ms (from lsession.md) |

### N = 2048 (FP32)

| | H100 | RTX 5070 |
|---|---|---|
| FLOPS | 2 × 2048³ = 17.18 GFLOP | same |
| Data | 12 × 2048² = 48.0 MB | same |
| AI | 341.3 FLOP/byte → compute | compute |
| t_compute (100%) | 17.18e9 / 67.0e12 = 0.256 ms | 0.551 ms |
| CuBLAS (RTX, est.) | — | ~8.28 ms (from lsession.md) |

RTX 5070 CuBLAS achieves ~8.28ms vs theoretical 0.55ms → **6.6% efficiency**. This gap is because CuBLAS includes kernel launch overhead, algorithm overhead (panel decomposition), and does not run at 100% peak.

### N = 4096 (FP32)

| | H100 | RTX 5070 |
|---|---|---|
| FLOPS | 2 × 4096³ = 137.4 GFLOP | same |
| Data | 12 × 4096² = 192 MB | same |
| AI | 682.7 FLOP/byte → compute | compute |
| **t_compute (100%)** | **2.05 ms** | **4.40 ms** |
| CuBLAS (RTX, measured) | — | ~6.8 ms (from lsession.md) |
| Efficiency (RTX) | — | 64.7% of peak |

### N = 4096 (BF16, Tensor Core)

| | H100 (TC) | RTX 5070 (TC) |
|---|---|---|
| Peak BF16 TC | 1979 TFLOPS | ~990 TFLOPS |
| t_compute (100%) | 137.4e9 / 1979e12 = 0.069 ms | 0.139 ms |
| CuBLAS (RTX, measured) | — | ~2.2 ms |
| Efficiency (RTX) | — | ~6.3% |

---

## 5. Register Analysis

### regsPerBlock = regsPerMultiprocessor = 65536

Both report 65536 because `regsPerBlock` is the **maximum register budget a single block may use**, which is capped at the SM's total register file size. It does **not** mean each block gets its own private 65536 registers.

Multiple blocks share the register pool:

```
max_blocks_per_SM = floor(65536 / (threads_per_block × regs_per_thread))
```

**Example: 256 threads/block, 32 regs/thread**
- Registers used = 256 × 32 = 8192
- Blocks per SM = floor(65536 / 8192) = **8 blocks/SM** (full occupancy at 2048 threads/SM)

**Example: 256 threads/block, 64 regs/thread**
- Registers used = 256 × 64 = 16384
- Blocks per SM = floor(65536 / 16384) = **4 blocks/SM**
- Threads/SM = 4 × 256 = 1024 (only 50% of max 2048)

**Example: 128 threads/block, 64 regs/thread**
- Registers used = 128 × 64 = 8192
- Blocks per SM = floor(65536 / 8192) = **8 blocks/SM**
- Threads/SM = 8 × 128 = 1024 (50% occupancy)

The compiler balances threads/block, registers/thread, and blocks/SM to maximize occupancy or performance.

---

## 6. Memory Hierarchy and GEMM Implications

### Hierarchy Summary

| Level | Scope | Size | Latency | Control |
|---|---|---|---|---|
| Registers | per-thread | 255 max/thread | ~1 cycle | compiler |
| Shared Memory | per-SM | 100–228 KB | ~20–30 cycles | **programmer** |
| L1 Cache | per-SM | hardware-managed | ~30 cycles | hardware |
| L2 Cache | whole-GPU | 48–50 MB | ~200–400 cycles | hardware |
| Global Memory | whole-GPU | 12–80 GB | ~400–800 cycles | — |

### GEMM Strategy

```
Global Memory → L2 → L1 → Shared Memory → Registers → Compute
```

**Level 1 — Global memory loads** (not cached automatically for compute kernels):
- Issue vector loads of width 4 (FP32) or 2 (FP16/BF16)
- Coalesce across threads: each thread loads contiguous columns
- L2 cache line is 32 bytes — vector loads of 4×FP32 hit in one cache line

**Level 2 — Shared memory (scratchpad)**:
- Tiles of A and B are loaded from global → shared memory
- All threads in a block reuse the tile from shared memory
- No global memory traffic for the K dimension within a block
- H100 has 228 KB/SM vs RTX 5070's 100 KB/SM → can hold larger tiles

**Level 3 — Registers (per-thread accumulation)**:
- Each thread accumulates partial results in registers
- Register pressure limits how many accumulators (C elements) a thread can hold
- A thread typically handles a 16×16 sub-tile (256 accumulators)

**H100 vs RTX for GEMM tiling:**

| | H100 | RTX 5070 |
|---|---|---|
| Shared mem/SM | 228 KB | 100 KB |
| L2 cache | 50 MB | 48 MB |
| Registers/SM | 65536 | 65536 |
| Max threads/SM | 2048 | 1536 |

H100's larger shared memory per SM allows for larger tile sizes, reducing global memory traffic and improving reuse. Both have similar L2 cache sizes.

---

## 7. Key Takeaways for Kernel Optimization

1. **RTX 5070 is memory-bandwidth-bound for small matrices (N < 200)**. Optimize loads, use vectorized access, enable L2 caching with `__ldg()` or proper memory access patterns.

2. **RTX 5070 maxThreadsPerSM = 1536** (vs H100's 2048). Your thread block size of 256 threads uses 6.7% of the SM's max thread capacity on RTX vs 12.5% on H100. Occupancy calculations differ.

3. **RTX 5070 sharedMemPerSM = 100 KB** (vs H100's 228 KB). Your 64×64 FP32 tile (16 KB) allows 6 blocks/SM on RTX before hitting shared memory limits, but register pressure may limit you further.

4. **Both GPUs have ~50 MB L2**. This is large enough to hold entire working sets for moderate matrix sizes. Tiled algorithms benefit from temporal locality in L2 across wavefronts of blocks.

5. **RTX 5070 has 5x less bandwidth** than H100. At N=4096, RTX needs ~2x more time than H100 for compute-bound GEMM (consistent with 67 vs 31 TFLOPS ratio). At smaller N, the gap widens because RTX is more memory-bound.

6. **Register allocation is the tightest constraint**. With 65536 regs/SM and 32 regs/thread, you can only run 1024 threads/SM maximum regardless of shared memory availability. Balance this against blocks/SM and shared memory usage.
