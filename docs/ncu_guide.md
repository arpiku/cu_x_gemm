# NVIDIA Nsight Compute (NCU) Profiling Guide

## Overview

NVIDIA Nsight Compute (NCU) is a kernel profiler for CUDA applications. It provides detailed metrics about GPU execution, including memory access patterns, compute utilization, and performance bottlenecks.

**Location**: `/usr/local/NVIDIA-Nsight-Compute-2026.1/ncu`

## Quick Start

### Basic Invocation

```bash
# Basic profile (full kernel)
ncu --set base ./build/cu_x_gemm

# With output to file
ncu --set full --report-section all ./build/cu_x_gemm > ncu_output.txt
```

### Output Formats

```bash
# CSV output (for parsing)
ncu --set base --csv ./build/cu_x_gemm

# JSON output (for detailed analysis)
ncu --set base --json ./build/cu_x_gemm > results.json
```

## Key Metrics for GEMM Analysis

### Memory Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | L1 cache global memory loads | High = good cache utilization |
| `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` | Memory transaction efficiency (%) | >80% = well coalesced |
| `l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum` | Shared memory loads | Should be high (SMEM is fast) |
| `lts__t_sectors_lookup_hit.sum` | L2 cache hits | Higher is better |

### Compute Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| `sm__throughput.fma.ops.sum` | FMA operations executed | Close to peak |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM utilization (%) | >50% = good |
| `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.sum` | Global memory store efficiency | >80% |

### Warp Stall Metrics

| Metric | Description | Indicates |
|--------|-------------|-----------|
| `smsp__average_warps_active_stall_mio.sum` | MIO (memory I/O) stalls | SMEM contention |
| `smsp__average_warps_active_stall_long_synthesized.sum` | Long stall cycles | Memory latency |
| `smsp__average_warps_active_stall_exec_wait.sum` | Execution dependency stalls | Compute bound |

## GEMM-Specific Profiling Commands

### 1. Memory Efficiency Survey

```bash
ncu --set base \
  --metrics "l1tex__t_sectors_pipe_lsu_mem_global_op_ld,\
              smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
              l1tex__t_sectors_pipe_lsu_mem_global_op_st,\
              smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct" \
  ./build/cu_x_gemm
```

**Interpretation**:
- Transaction efficiency <50%: Uncoalesced memory access
- Transaction efficiency >80%: Well-coalesced access

### 2. Warp Stall Analysis

```bash
ncu --set base \
  --metrics "smsp__average_warps_active_stall_mio,\
              smsp__average_warps_active_stall_long_synthesized,\
              smsp__average_warps_active_stall_exec_wait,\
              smsp__average_warps_active_stall_short_synthesized" \
  ./build/cu_x_gemm
```

**Interpretation**:
- High MIO stalls: Shared memory bottleneck
- High long stalls: Global memory latency
- High exec_wait stalls: Compute-bound (usually good for GEMM)

### 3. Cache Performance

```bash
ncu --set base \
  --metrics "l1tex__t_sectors_pipe_lsu_mem_shared_op_ld,\
              lts__t_sectors_lookup_hit,\
              lts__t_sectors_lookup_miss" \
  ./build/cu_x_gemm
```

### 4. Compute Throughput

```bash
ncu --set base \
  --metrics "sm__throughput.fma.ops,\
              sm__throughput.sfu.ops,\
              sm__throughput.avg.pct_of_peak_sustained_elapsed" \
  ./build/cu_x_gemm
```

## Profiling Our GEMM Kernels

### Recommended Profile Command

```bash
#!/bin/bash
# Profile GEMM kernel with key metrics

KERNEL="./build/cu_x_gemm"
OUTPUT_DIR="results/ncu"
mkdir -p "$OUTPUT_DIR"

ncu --set base \
  --metrics "l1tex__t_sectors_pipe_lsu_mem_global_op_ld,\
              smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
              smsp__average_warps_active_stall_mio,\
              smsp__average_warps_active_stall_long_synthesized,\
              sm__throughput.avg.pct_of_peak_sustained_elapsed" \
  --json \
  "$KERNEL" > "$OUTPUT_DIR/profile.json"
```

### Quick Survey Command

```bash
ncu --set base ./build/cu_x_gemm 2>&1 | head -100
```

## Interpreting Results

### Memory Bottleneck Indicators

| Symptom | Metric | Value |
|---------|--------|-------|
| Uncoalesced loads | `smsp__sass_average_data_bytes...pct` | <50% |
| L1 thrashing | `l1tex__t_sectors_pipe_lsu...` ratio | Low hit rate |
| Global memory bound | warp stall pattern | High long stalls |

### Compute Bottleneck Indicators

| Symptom | Metric | Value |
|---------|--------|-------|
| FMA bound | `sm__throughput.fma.ops` | Near peak |
| Low utilization | `sm__throughput.avg.pct...` | <30% |
| Execution stalls | `smsp__average_warps_active_stall_exec_wait` | High |

### Shared Memory Bottleneck Indicators

| Symptom | Metric | Value |
|---------|--------|-------|
| SMEM contention | `smsp__average_warps_active_stall_mio` | High |
| Bank conflicts | SMEM transaction pattern | Inefficient |

## GEMM Roofline Analysis

For GEMM, we can estimate performance position using:

```
Arithmetic Intensity (AI) = (BM × BN) / (BM + BN) FLOPs/byte

RTX 5070 Balance Point ≈ 30 FLOPs/byte (estimated)
H100 Balance Point ≈ 20 FLOPs/byte

If AI > Balance Point: Compute-bound (good)
If AI < Balance Point: Memory-bound (needs optimization)
```

## Common Bottleneck Patterns

### Pattern 1: High MIO Stalls
```
Problem: Shared memory access contention
Solution: Reduce shared memory per block, increase occupancy
         or optimize shared memory access pattern
```

### Pattern 2: Low Transaction Efficiency
```
Problem: Uncoalesced global memory access
Solution: Reorder memory accesses for sequential access,
         use vectorized loads (float4)
```

### Pattern 3: High Long Stalls
```
Problem: Global memory latency
Solution: Increase arithmetic intensity,
         use larger tiles, add caching
```

### Pattern 4: High Exec Wait Stalls
```
Problem: Compute-bound (usually good for GEMM)
Solution: Kernel is doing well, focus on other bottlenecks
```

## Profiling Different Matrix Sizes

```bash
# Profile at different sizes
for size in 1024 2048 4096; do
  ncu --set base ./build/cu_x_gemm --json > "ncu_size_${size}.json"
done
```

## Appendix: Full Metric List

### Memory Metrics
```
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum
l1tex__t_sectors_pipe_lsu_mem_shared_op_st.sum
lts__t_sectors_lookup_hit.sum
lts__t_sectors_lookup_miss.sum
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct
smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct
```

### Compute Metrics
```
sm__throughput.fma.ops.sum
sm__throughput.sfu.ops.sum
sm__throughput.avg.pct_of_peak_sustained_elapsed
smsp__sass_average_data_bytes_per_warp.pct
```

### Stall Metrics
```
smsp__average_warps_active_stall_mio.sum
smsp__average_warps_active_stall_long_synthesized.sum
smsp__average_warps_active_stall_exec_wait.sum
smsp__average_warps_active_stall_short_synthesized.sum
smsp__average_warps_active_stall_not_selected.sum
```

## References

- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUDA Profiling Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)
- [GPU Performance Analysis](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
