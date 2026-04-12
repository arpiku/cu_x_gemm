# Bank Conflict Analysis - Test Bed

## Concept Overview

### Shared Memory Banks
- NVIDIA GPUs have 32 banks of shared memory
- Each bank can service one transaction per cycle
- Bank = (row × num_cols + col) % 32

### Bank Conflicts
- **Conflict**: Multiple threads access same bank (different addresses)
- **No conflict**: Multiple threads access same address (broadcast)
- **Perfect access**: Each thread accesses unique bank

## Kernel Variants

| Kernel | SMEM Layout | Expected Behavior |
|--------|-------------|-------------------|
| 01_baseline_r1y | Bs[BK][BN]=64×64 | Low conflicts |
| 02_transposed_layout | Bs[BN][BK]=64×64 | **High conflicts** |
| 03_32x32_tiles | Bs[32][32] | Different pattern |
| 04_128x64_tiles | Bs[64][64] | Larger tiles |
| 05_warp_tiled | Bs[64][64] | Warp-coalesced |

## What to Look For

### 1. Bank Access Distribution
The counter-based output shows:
- Total accesses per bank
- Max/Min ratio (imbalance)
- Visual bar chart

**Ideal**: All banks roughly equal (ratio ~1.0-1.1)
**Bad**: Some banks much higher (ratio >2.0)

### 2. Kernel Performance
Compare execution times:
```
baseline_r1y:     ~X ms
transposed:       ~Y ms (expected: 2-4x slower due to conflicts)
32x32:            ~Z ms
128x64:           ~W ms
warp_tiled:       ~V ms
```

### 3. NCU Bank Conflict Counter
```
lsu__shared_bank_conflicts.sum
```
- Non-zero = actual bank conflicts at hardware level
- Compare across variants

## Expected Observations

### Transposed Layout (02)
- Bank calculation: `bank_id = (load_row * BK + load_col) % 32`
- With Bs[BN][BK]=64×64:
  - load_row ranges 0-63 (col 0-63), BK=64
  - For load_col = 0, all rows map to banks 0-31 (period 32)
  - This creates systematic conflicts vs Bs[BK][BN]

### Thread Tile Patterns
- 2×2 thread tiles means adjacent threads access nearby SMEM
- Bank conflicts more likely with certain (TM, TN) combinations

## Running Analysis

```bash
# Run all kernels
cd test-bed
./scripts/run_all.sh

# Profile with Nsight Systems
./scripts/profile_nsys.sh 01_baseline_r1y 256

# Profile with Nsight Compute (bank conflict metrics)
./scripts/profile_ncu.sh 01_baseline_r1y 256
```

## Metrics to Compare

| Metric | Good | Bad |
|--------|------|-----|
| Bank access imbalance | <20% | >50% |
| lsu__shared_bank_conflicts | 0 | >1000 |
| Kernel time (transposed/baseline) | ~1x | >2x |