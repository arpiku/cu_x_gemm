#!/bin/bash
# Build and run GEMM benchmark

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
BUILD_DIR="${ROOT_DIR}/build"
ARCH_DIR="unknown"

for arg in "$@"; do
    case "${arg}" in
        -h100)
            ARCH_DIR="h100"
            ;;
        -rtx5070)
            ARCH_DIR="rtx5070"
            ;;
    esac
done

RESULTS_DIR="${ROOT_DIR}/results/${ARCH_DIR}"

mkdir -p "${RESULTS_DIR}"

echo "=== Building cu_x_gemm ==="
cd "${ROOT_DIR}"
cmake -B build -S . > /dev/null
cmake --build build --parallel > /dev/null

echo ""
echo "=== Running benchmark ==="
"${BUILD_DIR}/cu_x_gemm" "$@" | tee "${RESULTS_DIR}/benchmark.log"

echo ""
echo "=== Done ==="
echo "Results saved to ${RESULTS_DIR}/"
ls -la "${RESULTS_DIR}/"
