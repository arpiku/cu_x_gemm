#!/bin/bash
# Profile GEMM kernels using Nsight Compute (ncu)
# Profiles the benchmark run and captures kernel-level metrics

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
BUILD_DIR="${ROOT_DIR}/build"
RESULTS_DIR="${ROOT_DIR}/results"

mkdir -p "${RESULTS_DIR}"

# Add NCU to PATH if not found
if ! command -v ncu &> /dev/null; then
    export PATH="/usr/local/NVIDIA-Nsight-Compute-2026.1:$PATH"
fi

if ! command -v ncu &> /dev/null; then
    echo "Error: ncu not found in PATH"
    echo "Please run: export PATH=\"/usr/local/NVIDIA-Nsight-Compute-2026.1:\$PATH\""
    exit 1
fi

echo "=== Nsight Compute Profiling ==="
echo ""

# Ensure binary is built
if [ ! -f "${BUILD_DIR}/cu_x_gemm" ]; then
    echo "Building cu_x_gemm..."
    cd "${ROOT_DIR}" && cmake -B build -S . > /dev/null && cmake --build build --parallel > /dev/null
fi

# Key metrics for GEMM performance analysis
# - Duration: kernel execution time
# - SM throughput: compute utilization
# - DRAM throughput: memory bandwidth utilization
# - L2 transactions: cache efficiency
# - Instructions: compute intensity

METRICS="gpu__time_duration.sum,\
sm__throughput_pct,\
dram__throughput_pct,\
lts__t_sectors_op_read.sum,\
lts__t_sectors_op_write.sum,\
smsp__inst_executed.sum"

OUTPUT_BASE="${RESULTS_DIR}/ncu_profile"

echo "Running NCU profile..."
echo "Output will be saved to ${RESULTS_DIR}/"
echo ""

# Profile with summary section (human-readable)
ncu --set full \
    --metrics "${METRICS}" \
    --launch-skip 10 \
    --launch-count 1 \
    --export "${OUTPUT_BASE}" \
    "${BUILD_DIR}/cu_x_gemm" 2>&1 | tee "${OUTPUT_BASE}.log" || true

echo ""
echo "=== Profile Complete ==="
echo ""
echo "Generated files:"
ls -la "${RESULTS_DIR}"/ncu_profile* 2>/dev/null || echo "No profile files generated"
echo ""
echo "To view detailed results:"
echo "  ncu-ui ${OUTPUT_BASE}.ncu-rep"
