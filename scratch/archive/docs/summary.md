# GEMM Benchmark Summary

## Latest Results (2026-04-12)

### Performance Comparison by Size

| Size | Master Selected | Custom (ms) | Sgemm (ms) | Ratio % | Notes |
|------|----------------|-------------|------------|--------|-------|
| 32   | naive          | 0.0020     | 0.0047   | 235.6% | Beats cuBLAS |
| 64   | naive          | 0.0033     | 0.0048   | 145.9% | Beats cuBLAS |
| 128  | r1y            | 0.0083     | 0.0046   | 55.5%  | Gap: 44.5% |
| 256  | r1y            | 0.0153     | 0.0074   | 48.3%  | **Gap: 51.7%** |
| 512  | r2x            | 0.0557     | 0.0253   | 45.4%  | **Gap: 54.6%** |
| 1024 | r2z2           | 0.1628     | 0.1140   | 70.0%  | Gap: 30.0% |
| 2048 | r2z2           | 0.8571     | 0.7342   | 85.7%  | Gap: 14.3% |
| 4096 | r2z2           | 6.6330     | 6.5238   | 98.4%  | **Parity!** |

### Individual Kernel Performance (FP32_VARIANT tests)

#### r3x (64x64 tiles, 64 threads, double-buffer + cp.async)
| Size | Custom (ms) | Sgemm (ms) | Ratio % | Notes |
|------|-------------|------------|--------|-------|
| 32   | 0.0042      | 0.0048    | 116.6% | Beats cuBLAS |
| 64   | 0.0064      | 0.0047    | 74.2%  | Gap: 25.8% |
| 128  | 0.0107      | 0.0045    | 42.6%  | Gap: 57.4% |
| 256  | 0.0190      | 0.0074    | 38.9%  | Gap: 61.1% |
| 512  | 0.0380      | 0.0251    | 66.0%  | Gap: 34.0% |
| 1024 | 0.1964      | 0.1236    | 62.9%  | Gap: 37.1% |
| 2048 | 1.2124      | 0.8109    | 66.9%  | Gap: 33.1% |
| 4096 | 9.6997      | 7.0511    | 72.7%  | Worse than r2z2 |

#### r2z2 (128x128 tiles, 128 threads, double-buffer + cp.async)
| Size | Custom (ms) | Sgemm (ms) | Ratio % | Notes |
|------|-------------|------------|--------|-------|
| 128  | 0.0188      | 0.0047    | 24.8%  | Gap: 75.2% |
| 256  | 0.0322      | 0.0074    | 22.9%  | Gap: 77.1% |
| 512  | 0.0629      | 0.0251    | 39.8%  | Gap: 60.2% |
| 1024 | 0.1700      | 0.1088    | 64.0%  | Gap: 36.0% |
| 2048 | 0.8576      | 0.7518    | 87.7%  | Gap: 12.3% |
| 4096 | 6.6903      | 6.5188    | 97.4%  | **Parity!** |

## Key Findings

### Medium Sizes Gap (Primary Target)
- **256**: r1y at 48.3% — gap 51.7%
- **512**: r2x at 45.4% — gap 54.6%

**Root Cause**: Warp occupancy problem
- 128x128 tiles = too few blocks at medium sizes
- r3x (64x64 tiles) provides 4x more blocks but still slower overall

### Small Sizes Performance
- naive beats cuBLAS at 32, 64 (overhead wins)
- r2z2 actually regresses at small sizes

### Large Sizes (Target Achieved)
- r2z2 at 4096: **98.4% parity** with cuBLAS ✓

## Optimization Strategy

### Opportunity: Tile Size Selection per Matrix Size
Different matrix sizes benefit from different tile configurations:

| Size Range | Best Config | BM | BN | Threads | Expected Benefit |
|-----------|------------|----|----|---------|------------------|
| ≤64       | naive      | - | - | -       | Minimal overhead |
| 128-256  | ?          | 64 | 64 | 64     | More blocks |
| 512      | r2x        | 128 | 128 | 256   | Current best |
| 1024+    | r2z2       | 128 | 128 | 128   | Double-buffer |

### Test: r3x at medium sizes
r3x (64x64 tiles) shows promise at medium sizes:
- 512: r3x 66.0% vs r2z2 39.8% (gap closes from 60.2% → 34.0%)
- 1024: r3x 62.9% vs r2z2 64.0% (competitive)

However, r3x underperforms at 4096 (72.7% vs r2z2 97.4%).

### Next Steps
1. Create optimizer script to test different tile configs per size
2. Identify optimal (BM, BN) for each matrix size
3. Update master dispatcher with size-specific tile configurations