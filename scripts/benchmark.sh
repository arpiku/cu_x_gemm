#!/bin/bash
# Build and run GEMM benchmark, then generate plots

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
BUILD_DIR="${ROOT_DIR}/build"
RESULTS_DIR="${ROOT_DIR}/results"

mkdir -p "${RESULTS_DIR}"

echo "=== Building cu_x_gemm ==="
cd "${ROOT_DIR}"
cmake -B build -S . > /dev/null
cmake --build build --parallel > /dev/null

echo ""
echo "=== Running benchmark ==="
"${BUILD_DIR}/cu_x_gemm"

echo ""
echo "=== Generating plots ==="
python3 "${SCRIPT_DIR}/plot_results.py" "${RESULTS_DIR}/benchmark_results.csv" -o "${RESULTS_DIR}"

echo ""
echo "=== Done ==="
echo "Results saved to ${RESULTS_DIR}/"
ls -la "${RESULTS_DIR}/"
