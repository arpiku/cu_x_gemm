#!/bin/bash
# NCU Profiling Script for GEMM Kernels
# Usage: ./scripts/ncu_profile.sh <variant> <size>

set -e

VARIANT=${1:-6}  # Default to r1x2 (variant 6)
SIZES=${2:-"1024 2048 4096"}
OUTPUT_DIR="results/ncu"

mkdir -p "$OUTPUT_DIR"

# Metrics for memory efficiency analysis
MEMORY_METRICS="l1tex__t_sectors_pipe_lsu_mem_global_op_ld,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,\
l1tex__t_sectors_pipe_lsu_mem_shared_op_ld"

# Metrics for warp stalls
STALL_METRICS="smsp__average_warps_active_stall_mio,\
smsp__average_warps_active_stall_long_synthesized,\
smsp__average_warps_active_stall_exec_wait,\
smsp__average_warps_active_stall_short_synthesized"

# Metrics for compute throughput
COMPUTE_METRICS="sm__throughput.fma.ops,\
sm__throughput.avg.pct_of_peak_sustained_elapsed"

# Metrics for cache performance
CACHE_METRICS="lts__t_sectors_lookup_hit,\
lts__t_sectors_lookup_miss"

echo "=============================================="
echo "NCU Profiling for GEMM Kernel"
echo "Variant: $VARIANT"
echo "Sizes: $SIZES"
echo "=============================================="

# Function to run profile and save results
run_profile() {
    local size=$1
    local metric_type=$2
    local metrics=$3
    local suffix=$4
    
    local output_file="$OUTPUT_DIR/v${VARIANT}_${size}${suffix}.json"
    
    echo "Profiling size=$size, metrics=$metric_type..."
    
    # Run NCU with metrics
    ncu --set base \
        --metrics "$metrics" \
        --csv \
        --json \
        ./build/cu_x_gemm 2>/dev/null | \
        grep -A 1000 "Device" | \
        tail -n +2 > "$output_file"
    
    echo "  -> Saved to $output_file"
}

# We need to test each size by modifying main.cu's DIMENSIONS array
# For now, just run the default (which tests all sizes)
# In a full setup, we'd compile multiple versions

echo ""
echo "Running full kernel profile..."
ncu --set base \
    --csv \
    --json \
    ./build/cu_x_gemm > "$OUTPUT_DIR/v${VARIANT}_full.json" 2>&1 || true

echo ""
echo "Profile complete. Results in $OUTPUT_DIR/"
echo ""

# Summary
echo "=============================================="
echo "Quick Summary:"
echo "=============================================="
if [ -f "$OUTPUT_DIR/v${VARIANT}_full.json" ]; then
    echo "Full profile saved to: $OUTPUT_DIR/v${VARIANT}_full.json"
    # Extract key metrics if available
    grep -E "(Time|gpu_time|sm__throughput)" "$OUTPUT_DIR/v${VARIANT}_full.json" | head -20 || true
fi
