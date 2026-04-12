# Test Bed - Bank Conflict Analysis

A separate test environment for exploring shared memory bank conflicts in GEMM kernels.

## Purpose

1. **Visualize bank conflicts** through counters and print statements
2. **Profile with Nsight** systems and compute for hardware-level analysis
3. **Compare variants** to understand impact of SMEM layouts and tile sizes

## Structure

```
test-bed/
├── kernels/           # 5 kernel variants
│   ├── 01_baseline_r1y.cu      # Reference (Bs[BK][BN])
│   ├── 02_transposed_layout.cu  # Bs[BN][BK] - high conflicts
│   ├── 03_32x32_tiles.cu        # Smaller tiles
│   ├── 04_128x64_tiles.cu       # Larger tiles
│   └── 05_warp_tiled.cu         # Warp-level access
├── scripts/
│   ├── run_all.sh      # Build and run all kernels
│   ├── profile_nsys.sh # Nsight Systems profiling
│   └── profile_ncu.sh  # Nsight Compute profiling
└── docs/
    └── explanation.md  # Bank conflict theory
```

## Quick Start

```bash
# Build and run all kernels
cd /home/arpiku/cu_x_gemm
cmake -B test-bed/build -S test-bed
cmake --build test-bed/build

# Run with matrix sizes 64, 128, 256, 512
./test-bed/build/01_baseline_r1y 256

# Or use the script
./test-bed/scripts/run_all.sh

# Profile with Nsight Systems
./test-bed/scripts/profile_nsys.sh 01_baseline_r1y 256

# Profile with Nsight Compute (bank conflict metrics)
./test-bed/scripts/profile_ncu.sh 01_baseline_r1y 256
```

## Kernels

| Kernel | Config | Expected Bank Behavior |
|--------|--------|------------------------|
| 01_baseline_r1y | 64×64×64, Bs[BK][BN] | Low conflicts (reference) |
| 02_transposed_layout | 64×64×64, Bs[BN][BK] | **High conflicts** (2-4x slower) |
| 03_32x32_tiles | 32×32×32, Bs[BK][BN] | Different pattern |
| 04_128x64_tiles | 128×64×64, Bs[BK][BN] | Larger tiles |
| 05_warp_tiled | 64×64, warp-tiled | Warp-coalesced |

## Output Interpretation

Each kernel prints:
- **Execution time** in ms
- **Bank access distribution** - bar chart of accesses per bank (32 banks)
- **Summary stats**: total accesses, max/min, imbalance %

Key metrics:
- `Max/Min ratio`: Closer to 1.0 = better balance
- `Imbalance %`: Lower = more even distribution

## Profiling

### Nsight Systems
- Timeline view: kernel overlap, SM utilization
- Command: `./scripts/profile_nsys.sh <kernel> <size>`

### Nsight Compute
- Bank conflict counter: `lsu__shared_bank_conflicts.sum`
- Command: `./scripts/profile_ncu.sh <kernel> <size>`

## Integration

After analysis, update `scratch/SCRATCH_INDEX.md` with:
- Best performing variant
- Bank conflict patterns discovered
- Recommendations for main project