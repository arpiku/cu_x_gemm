#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
BUILD_DIR="${ROOT_DIR}/test-bed/build"
RESULTS_DIR="${ROOT_DIR}/test-bed/results"

mkdir -p "${RESULTS_DIR}"

if ! command -v ncu &> /dev/null; then
    export PATH="/usr/local/NVIDIA-Nsight-Compute-2026.1:$PATH"
fi

if ! command -v ncu &> /dev/null; then
    echo "Error: ncu not found in PATH"
    echo "Please run: export PATH=\"/usr/local/NVIDIA-Nsight-Compute-2026.1:\$PATH\""
    exit 1
fi

KERNEL=${1:-01_baseline_r1y}
SIZE=${2:-256}

if [ ! -f "${BUILD_DIR}/${KERNEL}" ]; then
    echo "Error: Kernel ${KERNEL} not built"
    echo "Run scripts/run_all.sh first"
    exit 1
fi

OUTPUT="${RESULTS_DIR}/${KERNEL}_${SIZE}_ncu"

METRICS="gpu__time_duration.sum,\
lsu__shared_bank_conflicts.sum,\
smsp__average_shared_accesses.pct,\
sm__throughput_pct,\
dram__throughput_pct"

echo "=== Nsight Compute Profiling ==="
echo "Kernel: ${KERNEL}"
echo "Size: ${SIZE}"
echo "Output: ${OUTPUT}"
echo ""

ncu --set full \
    --metrics "${METRICS}" \
    --launch-skip 5 \
    --launch-count 1 \
    --export "${OUTPUT}" \
    "${BUILD_DIR}/${KERNEL}" $SIZE 2>&1 | tee "${OUTPUT}.log"

echo ""
echo "=== Generated files ==="
ls -la "${OUTPUT}".*

echo ""
echo "To view in GUI:"
echo "  ncu-ui ${OUTPUT}.ncu-rep"