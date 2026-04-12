# BF16 Preparation: Medium-Size Gaps & Tensor Core Strategy

## Preamble: Where We Stand

Current `kernel_master` performance vs cuBLAS `sgemm` (RTX 5070, April 2026):

| Size | Kernel | % of sgemm | Gap to parity |
|------|--------|------------|---------------|
| 32–64 | naive | 146–229% | — (we beat it) |
| 128–256 | r1y | 48–55% | large, but small matrices |
| 512 | r2x | 43% | large, but small matrices |
| 1024 | r2z2 | **67%** | **33% gap** ← primary concern |
| 2048 | r2z2 | **85.5%** | **15% gap** |
| 4096 | r2z2 | **98%** | 2% gap — essentially parity |

The 1024 gap (33%) is the most actionable. This document analyses why it exists,
whether tile tuning can close it, and what the correct strategy is for BF16
tensor core kernels and TF32.

---

## Part 1: The 1024 Gap — Root Cause Analysis

### 1.1 Warp Occupancy: The Real Problem

The RTX 5070 has **48 SMs**, each with a maximum of **1536 threads** (48 warps).
Our r2z2 kernel: 128 threads/block = 4 warps/block.

SMEM usage per block: `As[2][BK*BM] + Bs[2][BK*BN]` = 32 KB.  
SMEM per SM: 100 KB. Maximum blocks per SM (SMEM-limited): `floor(100/32)` = **3 blocks/SM**.  
Maximum warps per SM (from r2z2): `3 blocks × 4 warps` = **12 warps/SM**.  
Warp occupancy: `12 / 48` = **25%**.

Now consider what happens at each grid size:

| Matrix | Grid (BM=128) | Blocks | Blocks fit at once | Warps/SM |
|--------|---------------|--------|--------------------|----------|
| 512×512 | 4×4 | 16 | all (16 < 144) | 4–12 (avg 5) |
| 1024×1024 | 8×8 | 64 | all (64 < 144) | 4–12 (avg 5) |
| 2048×2048 | 16×16 | 256 | all (256 < 144)? No: 256 / 48 = 5.3 waves | 12 (waves 1–5) |
| 4096×4096 | 32×32 | 1024 | 1024 / 144 ≈ 7.1 waves | 12 (most waves) |

Wait — at 1024×1024, all 64 blocks fit simultaneously (64 < 144 capacity). But the
**per-SM warp count is the bottleneck**: 64 blocks spread across 48 SMs gives an
average of 1.33 blocks/SM, so most SMs run only **4–8 warps** instead of the
maximum 12. With 4 warps active per SM, the GPU has almost no ability to hide
memory latency by switching to a different warp.

At 4096×4096, each wave fills all SMs to 3 blocks = 12 warps, giving 3× better
latency-hiding capacity. The "tail wave" (1024 mod 144 = ~16 blocks in the final
partial wave) contributes only ~1.5% of total runtime, so its impact is negligible.

**Diagnosis**: The 1024 gap is primarily a **warp occupancy problem**, not an
arithmetic intensity problem. Our kernel architecture (128 threads, 32KB SMEM) is
well-matched to large matrices but starves the SM of warps at medium sizes.

### 1.2 Why cuBLAS Doesn't Have This Problem

cuBLAS uses size-specific kernel configurations selected from a lookup table built
by exhaustive offline autotuning. For 1024×1024, it likely uses:
- Smaller tiles (64×64 or 96×96) to increase block count and warp density
- Different thread counts (256 threads → 8 warps/block → 24 warps/SM with 3 blocks)
- Or StreamK decomposition (see §1.4)

cuBLAS is not a single kernel — it is a library of hundreds of kernels, one
selected per (M, N, K, dtype, hardware) tuple.

---

## Part 2: Tile Size Tuning — Options and Critique

### 2.1 What Tile Tuning Changes

The fundamental tradeoff: **tile size ↑ → arithmetic intensity ↑, block count ↓**.

Global-memory arithmetic intensity for a BM×BN output tile:
```
AI_global = (2 × BM × BN × K) / ((BM + BN + BM×BN) × K/BK × bytes_per_elem)
          ≈ BM × BN / (BM + BN)     [for large K]
```

| Tile | AI_global (FLOPs/byte) | Grid at 1024 | Blocks | Avg warps/SM |
|------|------------------------|--------------|--------|--------------|
| 64×64 | 32 | 16×16 = 256 | 256 | ~21 |
| 96×96 | 48 | 11×11 = 121 | 121 | ~10 |
| 128×128 | 64 | 8×8 = 64 | 64 | ~5 |
| 128×64 | 43 | 8×16 = 128 | 128 | ~9 |

A 64×64 tile at 1024 gives 256 blocks → 5.3 blocks/SM average → 21 warps/SM.
That is **4× better warp occupancy** than the current 128×128 configuration,
at the cost of **2× lower arithmetic intensity**.

### 2.2 The 64×64 Tile Variant — Detailed Analysis

Config: BM=64, BN=64, BK=16, TM=4, TN=4, 128 threads (same warp-tiling structure
but scaled down). SMEM single-buffer: `As[64×16] + Bs[16×64]` = 8 KB.

Or double-buffered: `As[2][64×16] + Bs[2][16×64]` = 16 KB.  
Max blocks/SM (SMEM): `floor(100/16)` = **6 blocks/SM**.  
Max warps/SM: `6 × 4` = **24 warps/SM** = **50% occupancy**.

At 1024×1024 with 256 blocks and 6-block/SM capacity: all 256 blocks fit in
`ceil(256/288)` = 1 wave, with SMs averaging 5.3 blocks = 21 warps. That is a
radical improvement.

**But**: the arithmetic intensity drops from 64 to 32 FLOPs/byte. This means
each byte fetched from global memory contributes half as many FLOPs. Whether the
occupancy gain outweighs the intensity loss depends on which resource is the
binding constraint. Given the roofline analysis shows 1024 is deep in the
compute-bound regime (AI = 170 FLOPs/byte >> break-even of 46 FLOPs/byte), both
tile configs are compute-bound. The occupancy gain should dominate, and 64×64 tiles
should win at 1024.

At 4096×4096 with 64×64 tiles: grid = 64×64 = 4096 blocks. This is vastly
more blocks than necessary and causes significant overhead from block scheduling,
L2 cache thrashing (4096 tiles touching overlapping regions), and reduced data
reuse. The 128×128 tile clearly wins here.

**Verdict**: 64×64 tiles should improve 1024 substantially, but will hurt 2048+.
This is not a single kernel solution — it requires a separate variant dispatched
by kernel_master.

### 2.3 Non-Power-of-2 Tiles (96×96 etc.) — Why They Are Problematic

NVIDIA GPU warps are 32 threads wide. Thread block dimensions should be multiples
of 32 for full warp utilization. Tile dimensions drive thread layouts:
```
threads_per_block = (BM / TM) × (BN / TN)     [for register tiling]
```

For TM=4, TN=4: `threads_per_block = (BM/4) × (BN/4)`.  
With BM=BN=96: `threads_per_block = 24 × 24 = 576` — not a multiple of 32.  
This creates partial warps, reducing efficiency.

Workarounds (1D linearised thread indexing, padding) exist but add complexity
and typically cannot recover the wasted lanes. For warp-tiled kernels (r2z2
style), the warp decomposition also fails cleanly: `96 / WM` needs WM to divide
96 evenly. With WM=32: `96/32 = 3` warps in M — works. But `BN/WN = 96/32 = 3`,
so 9 warps total per block (288 threads, not a power of 2). Bank conflict analysis
becomes non-trivial.

**Recommendation**: Stick to power-of-2 tile dimensions. The 64×64 and 128×128
are the practical choices on current NVIDIA hardware. 128×64 asymmetric tiles
are worth exploring (grid at 1024: 128 blocks, ~9 warps/SM average) but add
complexity to the warp decomposition logic.

### 2.4 Increasing Thread Count — An Alternative Angle

Rather than shrinking the output tile, keep BM=BN=128 but double thread count
from 128 → 256 (8 warps/block). This directly improves warp density:

- Max blocks/SM (SMEM, 32 KB/block): 3 blocks — unchanged
- Warps/SM: `3 × 8` = **24 warps/SM** = 50% occupancy

At 1024×1024: 64 blocks, 3 blocks/SM → avg 1.33 blocks/SM → avg 5.3 warps/SM.
Better than 128-thread case (4 warps/block × 1.33 = 5.3 → 10.7 warps), but still
below the 24-warp ceiling since we don't have enough blocks.

The deeper issue: at 1024×1024 the grid is simply too small. More warps per block
helps 2048+ (where blocks fill all SMs to capacity) but not 1024 (where we don't
have enough blocks).

Register pressure is the other concern: doubling threads halves the register budget
per thread from ~256 to ~128. r2z2's register-heavy inner loop (128 accumulator
floats + warp indexing variables) uses 80–100 registers per thread — doubling
threads risks spilling to local memory with 128 threads' worth of budget per block.

**Verdict**: 256 threads helps 2048+ (currently 85.5%), marginal at 1024, risky
for register pressure. Worth benchmarking as a separate variant.

### 2.5 Quantitative Expectations for Tile Tuning

Rough expected gains at 1024×1024:

| Approach | Expected improvement | Confidence | Risk |
|----------|---------------------|------------|------|
| 64×64 tile (separate kernel) | +15–25% (to ~82–92%) | Medium | Low |
| 128×128, 256 threads | +5–12% (to ~72–79%) | Low-Medium | Register spill |
| 128×64 asymmetric tile | +8–15% (to ~75–82%) | Low | Complex warp layout |
| StreamK decomposition | +20–35% (to ~87–102%) | Medium-High | Complex implementation |

These are estimates. Actual numbers require benchmarking. The range is wide
because the binding constraint (warp occupancy vs memory latency vs pipeline
throughput) is not measured yet — NCU profiling is needed to confirm.

### 2.6 The Honest Critique of Tile Tuning

Tile tuning is an approximation of a scheduling problem. The fundamental issue
is that cuBLAS uses **auto-generated lookup tables** from exhaustive autotuning
across thousands of configurations. Our tile tuning is manual and covers at most
3–4 variants.

What tile tuning **cannot fix**:
- The L2 cache thrashing from reordering block execution (thread block swizzling
  solves this but is separate from tile size)
- Imbalanced wavefront execution (a 64-block grid with 48 SMs will always have
  one wasted partial wave)
- The instruction-level pipeline stalls inside the inner compute loop, which NCU's
  "warp stall — math pipeline throttle" or "warp stall — MIO" metrics would reveal

**Recommendation**: Before implementing a new 64×64 variant, run NCU on r2z2 at
1024×1024 to confirm the bottleneck is warp occupancy, not something else (e.g.,
L2 misses, instruction cache pressure). The `sm__warps_active.avg.per_cycle_active`
metric directly measures warp occupancy; if it's < 12 (out of 48), occupancy is the
bottleneck and 64×64 tiles will help. If it's already near 12, the problem is
elsewhere.

---

## Part 3: Other Approaches for Medium Sizes

### 3.1 Thread Block Swizzling

The default CUDA grid launches blocks in row-major order (block (0,0), (1,0),
(2,0), ...). For GEMM, adjacent row-blocks access very different rows of A and
the same column of B. This leads to poor L2 reuse for A (each block loading
independent rows).

Swizzling reorders the (cRow, cCol) mapping so adjacent blocks access the same
column strip of A, improving L2 temporal reuse:
```cpp
// Standard linear block index
uint block_id = blockIdx.y * gridDim.x + blockIdx.x;

// Swizzle: groups of SWIZZLE_WIDTH columns process the same rows
uint swizzle_width = 8;  // tune per hardware
uint cCol = (block_id / (gridDim.y * swizzle_width)) * swizzle_width
            + block_id % swizzle_width;
uint cRow = (block_id / swizzle_width) % gridDim.y;
```

This adds 5–10 arithmetic instructions per block but can reduce global/L2 traffic
by 10–20% at medium sizes where the working set doesn't fit entirely in L2.
At 1024 (192 KB data), the working set fits in RTX 5070's 48 MB L2 easily,
so swizzling gives marginal benefit here. More impactful at larger sizes or
on H100 where streaming bandwidth is a larger fraction of runtime.

### 3.2 StreamK Decomposition

StreamK partitions the K dimension across multiple blocks. Instead of each block
being responsible for a fixed output tile, a StreamK block processes a fixed
number of K-iterations from anywhere in the matrix, then uses atomics to
accumulate into the output.

**Effect at 1024×1024**: Instead of 64 blocks (one per 128×128 output tile),
could use e.g. 256 blocks each computing K/4 iterations per output tile. This
fills all 48 SMs uniformly, eliminating the partial-wave problem entirely.

**Cost**:
- Requires an atomic accumulation step (FP32 atomicAdd on C, or a custom split-K
  reduction kernel)
- Atomic accumulation introduces a synchronization barrier and potential contention
- Implementation complexity is significantly higher than tile tuning
- For sizes already at 98% (4096), adds overhead with no gain

StreamK is the right solution for the medium-size gap if maximising performance
at those sizes matters. It's the technique used by CUTLASS 3.x and Triton's
matmul kernel for load balancing. Expected improvement at 1024: +20–35%.

### 3.3 Persistent Kernels

Instead of launching a new kernel per GEMM, a persistent kernel loops over
multiple output tiles inside the same kernel, with work distributed via a global
atomic counter. Blocks self-schedule by atomically claiming the next tile.

**Advantage**: perfect load balancing, eliminates the quantisation penalty entirely.
**Disadvantage**: Requires global atomic coordination, adds memory traffic for the
counter, and complicates register management (the block must reset state between
tiles).

For a single GEMM (our use case), persistent kernels add overhead for small-medium
sizes without benefit. They shine in batched GEMM or repeated GEMM scenarios where
the kernel stays alive across multiple operations.

---

## Part 4: BF16 Tensor Core Kernel Strategy

### 4.1 Hardware Capability

Both target GPUs support BF16 tensor cores:

| GPU | SM | TC Gen | BF16 TC Peak | MMA shape (WMMA) |
|-----|-----|--------|--------------|-----------------|
| RTX 5070 | 12.0 | 5th gen (Blackwell) | ~500–990 TFLOPS | m=16, n=16, k=16 |
| H100 SXM | 9.0 | 4th gen (Hopper) | 1979 TFLOPS | m=16, n=16, k=16 |

Measured cuBLAS TC at 4096×4096 on RTX 5070: **4.1ms** vs our FP32 r2z2 at
**6.7ms** → TC is **1.63× faster** than our best FP32 kernel.

Theoretical min at 990 TFLOPS: 0.14ms. cuBLAS reaches ~33 TFLOPS effective = ~3%
of theoretical peak. This sounds low but is typical: tensor core efficiency is
limited by SMEM bandwidth, register pressure, and memory latency, not raw TFLOP count.

Our target: a BF16 kernel that approaches or exceeds cuBLAS TC performance
(≤4.1ms at 4096×4096).

### 4.2 Why TC Kernels Are Architecturally Different

The fundamental shift from CUDA-core kernels to tensor core kernels:

**CUDA cores**: Each thread performs 1 FMA per cycle. Control is per-thread.
Accumulators are plain float registers. SMEM layout is programmer-controlled for
bank-conflict avoidance.

**Tensor cores**: A full warp (32 threads) collectively performs one MMA operation:
a 16×16×16 matrix multiply-accumulate consuming 8 registers of A, 8 of B, and
8 of C per thread, producing 8 output registers. The hardware parallelises across
the warp internally. The programmer does not control which thread holds which
element of the fragment.

Consequences:
1. **Fragment opacity**: `wmma::fragment` has implementation-defined internal layout.
   You cannot index into a fragment by (row, col) position. You must load from SMEM
   and store back to SMEM at output.
2. **SMEM layout is constrained**: `wmma::load_matrix_sync` expects A in row-major
   and B in col-major (or transposed variants). Storing A transposed in SMEM (as we
   do in r2z2 for bank-conflict avoidance) is wrong — TC loads expect standard layout.
3. **Bank conflict strategy changes**: Since the TC load pattern is opaque, use
   **SMEM padding** instead of transposition:
   ```cpp
   __shared__ float As[BM][BK + 8];   // +8 floats padding to avoid 32-bank stride conflicts
   __shared__ __nv_bfloat16 Bs[BK][BN + 8];
   ```
4. **All warps must execute mma_sync**: no divergence within a warp during MMA.

### 4.3 SMEM Tile Sizing for TC

For BF16 (`__nv_bfloat16`, 2 bytes):

Block tile BM=128, BN=128, BK=32 (larger BK beneficial for TC due to higher
arithmetic intensity):
```
As: BM × BK × 2 bytes = 128 × 32 × 2 = 8 192 bytes = 8 KB
Bs: BK × BN × 2 bytes = 32 × 128 × 2 = 8 192 bytes = 8 KB
Total single-buffer: 16 KB
Double-buffer: 32 KB  (fits in 48 KB limit)
```

With BK=32 (vs BK=16 in r2z2), the MMA inner loop runs 32 K-iterations per
SMEM tile load, doubling the FLOPs per global memory byte:
```
AI_SMEM(BK=16) = 128×128×16×2 / (128×16 + 16×128) / 2 = 64 FLOPs/byte
AI_SMEM(BK=32) = 128×128×32×2 / (128×32 + 32×128) / 2 = 128 FLOPs/byte
```
Higher BK is only feasible for TC kernels because WMMA loads handle the BK
loop efficiently via 16-element accumulators.

For double-buffered BK=32: 32 KB SMEM — fits. For BK=64 double-buffer: 64 KB —
exceeds default; would require `cudaFuncSetAttribute(...MaxDynamicSharedMemorySize, 65536)`.

### 4.4 Kernel Structure — WMMA API

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

// Fragments per warp: WMMA shapes for BF16
// m=16, n=16, k=16 is the standard BF16 WMMA size
fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float>                  c_frag;

fill_fragment(c_frag, 0.0f);  // Zero accumulators

// K-loop (over SMEM tiles)
for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // Load tile from global → SMEM (double-buffered, cp.async)
    // ...

    // Inner loop over MMA tiles within the SMEM tile
    for (int inner_k = 0; inner_k < BK; inner_k += 16) {
        int smem_a_row = warp_m_tile * WM;
        int smem_b_col = warp_n_tile * WN;

        load_matrix_sync(a_frag, &As[smem_a_row][inner_k], BK + 8);  // +8 padding
        load_matrix_sync(b_frag, &Bs[inner_k][smem_b_col], BN + 8);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
}

// Store result to SMEM then to global
store_matrix_sync(&Cs[warp_m_tile * WM][warp_n_tile * WN], c_frag, BN, mem_col_major);
```

### 4.5 Warp-Level Tile Structure for TC

With WMMA 16×16 output tiles per warp, the block tile 128×128 decomposes as:

```
Block tile: 128 × 128
Warp tile:  64 × 64   (WM=64, WN=64, 4 warps in 2×2 arrangement)
MMA tiles per warp: (64/16) × (64/16) = 4 × 4 = 16 mma_sync calls per K-block
```

With BK=32 inner loop: 32/16 = 2 MMA steps in K per SMEM tile.  
Total MMA calls per block per K-tile: 4 warps × 16 tiles × 2 steps = 128.  
Each mma_sync: 16×16×16 BF16 = 8192 FLOPs.  
Total FLOPs per K-tile: 128 × 8192 = 1 048 576 FLOPs.

This is the compute density per block per SMEM load — compare to r2z2's
128×128×16 = 262144 FLOPs/tile, 4× less. The TC kernel computes 4× more per
global memory byte at equal BK (and 8× more with BK=32).

### 4.6 BF16 Global Memory Loading

Global memory holds BF16 data (`__nv_bfloat16`, 2 bytes each). Using `bfloat162`
(2-element packed) for vectorized loads:
```cpp
// Load 4 × bfloat16 pairs = 8 BF16 elements in one 128-bit transaction
__nv_bfloat162 tmp = *(const __nv_bfloat162*)(&A[row * K + k_col]);
// Or use nv_bfloat16x4 if available on the target
```

With cp.async for BF16:
```cpp
// 16-byte async copy = 8 BF16 elements
__pipeline_memcpy_async(&As_smem[...], &A_global[...], 16);
```

The BF16 SMEM layout must be row-major for A (to match `wmma::row_major`)
and col-major for B (to match `wmma::col_major`). This means A can be loaded
straight from global (stored row-major) into SMEM row-major — no transpose needed.
B needs a transpose from its global row-major layout to SMEM col-major, or
an explicit `wmma::col_major` + transposed load.

**Simpler alternative**: Store B transposed in global memory (col-major B) for
the kernel. The host passes B already transposed. For training workloads where
B is a weight matrix, this is a one-time layout conversion acceptable at
initialisation time.

### 4.7 C Output Handling

`wmma::store_matrix_sync` writes the 16×16 accumulator fragment to SMEM.
Then threads read from SMEM and apply alpha/beta scaling before writing to global C:
```cpp
// Store to SMEM scratch
__shared__ float C_smem[BM][BN];
store_matrix_sync(&C_smem[warp_m * WM + mma_m * 16][warp_n * WN + mma_n * 16],
                  c_frag, BN, mem_row_major);
__syncthreads();

// Write from SMEM to global (float32 output) with vectorized stores
for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
    int row = i / BN, col = i % BN;
    int global_row = cRow * BM + row;
    int global_col = cCol * BN + col;
    C[global_row * N + global_col] = alpha * C_smem[row][col]
                                   + beta  * C[global_row * N + global_col];
}
```

The C output scratch adds BM×BN×4 = 64 KB of SMEM — which exceeds limits if
added to the double-buffered A/B (32 KB). Solutions:
1. **Accumulate directly in registers** via WMMA fragments, write C at the end
   without SMEM scratch (only works when alpha=1, beta=0; or with separate kernel
   for scaling)
2. **Dynamic SMEM**: set carveout to 100 KB and allocate C scratch + A/B together
3. **Write C tile-by-tile**: after each 16×16 warp MMA tile is complete, write
   that sub-tile to global immediately (no SMEM for C at all)

Option 3 is simplest: warp calls `store_matrix_sync` to a 16×16 SMEM scratch
(only 1 KB, negligible), applies alpha/beta, writes float4 to global C.

### 4.8 Expected Performance and Milestones

Realistic milestones for the BF16 TC kernel:

| Milestone | Expected time at 4096 | vs cuBLAS TC | Notes |
|-----------|-----------------------|--------------|-------|
| WMMA naive (no SMEM opt) | ~30–50ms | ~8–14% | Correct but slow |
| WMMA + SMEM tiling (single-buffer) | ~10–15ms | ~27–41% | Similar to r2x era |
| WMMA + double-buffer + cp.async | ~5–8ms | ~51–82% | Target for initial version |
| Fully optimised (BK=64, larger tiles) | ~4.1–4.5ms | **91–100%** | Matches cuBLAS TC |

The last milestone requires either dynamic SMEM (BK=64 double-buffer = 128 KB)
or extremely tight register/pipeline management. This is a multi-iteration effort.

---

## Part 5: TF32 Kernel for FP32

### 5.1 What TF32 Is

TF32 (TensorFloat-32) is NVIDIA's internal precision format for FP32 GEMM on
tensor cores (Ampere+, SM 8.0+):
- **Storage**: 32-bit (same as FP32, no conversion overhead)
- **Precision**: 10-bit mantissa (same as FP16) + 8-bit exponent (same as FP32)
- **Accuracy**: Within 2–3× of FP32 for typical ML workloads; not suitable for
  scientific computing where exact FP32 semantics are required
- **Throughput**: Same tensor core path as BF16 (but k=8 per MMA tile vs k=16 for BF16)

cuBLAS `sgemm` already uses TF32 internally on Ampere+, which is why our custom
pure FP32 kernel (r2z2 at 98%) matches it — we're both hitting the same compute
limit, cuBLAS just has better pipeline efficiency.

A custom TF32 kernel could be 1.5–2× faster than r2z2 by using tensor cores for
FP32 accumulation, at the cost of reduced numerical precision.

### 5.2 WMMA API for TF32

```cpp
// TF32 WMMA: m=16, n=8, k=8 tile shape (note: different from BF16's 16×16×16)
wmma::fragment<wmma::matrix_a, 16, 16, 8,
               wmma::precision::tf32, wmma::row_major>  a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 8,
               wmma::precision::tf32, wmma::col_major>  b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 8, float>     c_frag;

// Load from FP32 SMEM — WMMA automatically converts FP32 → TF32 on load
load_matrix_sync(a_frag, smem_a_fp32_ptr, lda);
load_matrix_sync(b_frag, smem_b_fp32_ptr, ldb);
mma_sync(c_frag, a_frag, b_frag, c_frag);  // Accumulates in FP32
```

The key: TF32 tensor cores accept FP32 input and accumulate to FP32 output.
The precision loss happens silently during the mantissa truncation on load.
For most neural network workloads, this is acceptable and matches cuBLAS default
behaviour.

### 5.3 TF32 Global Data Format

Since TF32 uses the same 32-bit storage as FP32, the kernel accepts `const float*`
inputs identically to r2z2. No data format conversion on the host side. The
truncation from FP32 → TF32 happens implicitly inside `load_matrix_sync`.

This makes a TF32 kernel a direct drop-in for the current FP32 kernel_master,
with a precision tradeoff that must be disclosed.

### 5.4 TF32 SMEM Sizing

TF32 uses 4-byte elements (same as FP32). For BM=128, BN=128, BK=8 (TF32 k=8):
```
As: 128 × 8 × 4 = 4 KB
Bs: 8 × 128 × 4 = 4 KB
Double-buffer: 16 KB — very comfortably fits in 48 KB
```

With BK=8, global load efficiency is lower (fewer elements per SMEM load than
BK=16 or BK=32). Compensate by using a wider KT (K-iterations per outer loop):
process 4 MMA steps per SMEM tile → effective BK = 32 elements but loaded in
4 rounds of BK=8.

Alternatively: increase BK directly. BK=32 for TF32:
```
Double-buffer As[2][128×32] + Bs[2][32×128]: 4×128×32×4 = 65 536 bytes = 64 KB
```
This requires dynamic SMEM carveout. On RTX 5070, can set to 100 KB. On H100
up to 228 KB. Use `cudaFuncSetAttribute` with `cudaFuncAttributeMaxDynamicSharedMemorySize`.

### 5.5 TF32 vs BF16 — Prioritisation

Both TF32 and BF16 kernels use tensor cores. The key differences:

| Property | TF32 | BF16 |
|----------|------|------|
| Input format | FP32 (auto-truncated) | BF16 (already truncated) |
| MMA k-step | 8 | 16 |
| Accumulator | FP32 | FP32 |
| BK efficiency | Lower (2× fewer K per MMA) | Higher |
| Drop-in for FP32? | Yes (same pointer types) | No (requires BF16 cast) |
| Accuracy loss | Low (10-bit mantissa) | Same as BF16 (10-bit mantissa) |
| Use case | FP32 API, transparent speedup | BF16 training/inference |

**Prioritisation recommendation**: Implement BF16 TC first. Reasons:
1. BF16 is the standard training dtype for modern ML — it provides the highest
   value for the stated goal of matching cuBLAS TC
2. BF16 WMMA uses k=16 (vs TF32's k=8), meaning the inner loop is twice as
   compute-dense per SMEM load — easier to saturate the pipeline
3. The BF16 kernel also provides a foundation for the TF32 kernel (same warp
   tiling structure, different fragment type)
4. cuBLAS TC with BF16 (measured: 4.1ms) is already providing the baseline for
   comparison

---

## Part 6: Recommended Implementation Sequence

```
Phase 1: Close the medium-size gap (FP32)
  1a. NCU profile r2z2 at 1024×1024: confirm warp occupancy is bottleneck
  1b. Implement r3x: BM=64, BN=64, BK=16, double-buffer, same cp.async structure
  1c. Benchmark r3x at 1024, 2048, 4096 — add to kernel_master below new threshold
  1d. (Optional) implement StreamK for 1024 if r3x gain < 15%

Phase 2: BF16 Tensor Core kernel (bf16_tc)
  2a. Implement WMMA naive baseline: correct output, verify L2 error < 0.01
  2b. Add SMEM double-buffering + cp.async (BK=32, 32 KB SMEM)
  2c. Benchmark at 512, 1024, 2048, 4096 vs cuBLAS TC (target: ≥ 82% by end)
  2d. Tune: BK, thread count, SMEM carveout, swizzle

Phase 3: TF32 kernel for FP32 (optional)
  3a. Adapt bf16_tc kernel: change fragment type to wmma::precision::tf32, k=8
  3b. Keep FP32 input interface (no host changes)
  3c. Add to kernel_master as optional precision mode
```

---

## Appendix: Key Hardware Limits for Kernel Sizing (RTX 5070)

| Resource | Limit | Binding constraint for r2z2 (128t, 32KB) |
|----------|-------|------------------------------------------|
| SMEM/SM | 100 KB | 3 blocks/SM → 12 warps/SM (25% occupancy) |
| SMEM/block (default) | 48 KB | r2z2 uses 32 KB — fits; headroom for BK=32 single-buf |
| Threads/SM | 1536 | 3×128=384 active — well under; not binding |
| Regs/SM | 65536 | r2z2 uses ~100 regs/thread × 384 = 38 400 — 59% |
| Warps/SM | 48 | r2z2: 12/48 = 25% — the binding occupancy limit |

For WMMA BF16 kernel (target: BM=128, BN=128, BK=32, 256t, double-buffer):
| Resource | Usage | Fits? |
|----------|-------|-------|
| SMEM (double-buf) | 32 KB (BF16) | Yes, within 48 KB |
| Threads/block | 256 | 2 blocks/SM (SMEM) = 16 warps/SM = 33% |
| SMEM with dynamic | 100 KB usable | Can fit BK=64 double-buffer (64 KB) |
