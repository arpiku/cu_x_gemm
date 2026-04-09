#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build"
RESULTS_DIR="${SCRIPT_DIR}/../results"
NCU_PATH="/usr/local/NVIDIA-Nsight-Compute-2026.1/ncu"

mkdir -p "${RESULTS_DIR}"

if [ ! -f "${BUILD_DIR}/cu_x_gemm_benchmark" ]; then
    echo "Building cu_x_gemm_benchmark..."
    cd "${BUILD_DIR}" && make -j$(nproc)
fi

echo "Profiling with Nsight Compute..."
echo ""

DIM=${1:-1024}
DTYPE=${2:-BF16}

echo "Dimension: ${DIM}, Type: ${DTYPE}"
echo ""

"${NCU_PATH}" \
    --set full \
    --target-processes all \
    --export "${RESULTS_DIR}/profile_${DTYPE}_${DIM}" \
    "${BUILD_DIR}/cu_x_gemm_benchmark" \
    --single ${DIM} \
    --types ${DTYPE} \
    --warmup 5 \
    --measure 10

echo ""
echo "Profile saved to ${RESULTS_DIR}/profile_${DTYPE}_${DIM}.ncu-rep"
echo "Open with: ncu-ui ${RESULTS_DIR}/profile_${DTYPE}_${DIM}.ncu-rep"
