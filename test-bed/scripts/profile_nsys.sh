#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
BUILD_DIR="${ROOT_DIR}/test-bed/build"
RESULTS_DIR="${ROOT_DIR}/test-bed/results"

mkdir -p "${RESULTS_DIR}"

if ! command -v nsys &> /dev/null; then
    export PATH="/usr/local/NVIDIA-Nsight-Systems-cli-2026.1:$PATH"
fi

if ! command -v nsys &> /dev/null; then
    echo "Error: nsys not found in PATH"
    echo "Please run: export PATH=\"/usr/local/NVIDIA-Nsight-Systems-cli-2026.1:\$PATH\""
    exit 1
fi

KERNEL=${1:-01_baseline_r1y}
SIZE=${2:-256}

if [ ! -f "${BUILD_DIR}/${KERNEL}" ]; then
    echo "Error: Kernel ${KERNEL} not built"
    echo "Run scripts/run_all.sh first"
    exit 1
fi

OUTPUT="${RESULTS_DIR}/${KERNEL}_${SIZE}"

echo "=== Nsight Systems Profiling ==="
echo "Kernel: ${KERNEL}"
echo "Size: ${SIZE}"
echo "Output: ${OUTPUT}"
echo ""

nsys profile -o "${OUTPUT}" \
    --trace=cuda,nvtx \
    --gpu-metrics-device=all \
    "${BUILD_DIR}/${KERNEL}" $SIZE

echo ""
echo "=== Generated files ==="
ls -la "${OUTPUT}".*

echo ""
echo "To view in GUI:"
echo "  nsys-ui ${OUTPUT}.nsys-rep"