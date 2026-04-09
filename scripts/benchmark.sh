#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build"
RESULTS_DIR="${SCRIPT_DIR}/../results"

mkdir -p "${RESULTS_DIR}"

echo "Building cu_x_gemm_benchmark..."
cd "${BUILD_DIR}" && make -j$(nproc)

echo ""
echo "Running benchmarks..."
echo ""

DIMENSIONS="32,64,128,256,512,1024,2048,4096"
WARMUP=10
MEASURE=50

for DTYPE in BF16 FP32; do
    echo "=== Benchmarking ${DTYPE} ==="
    "${BUILD_DIR}/cu_x_gemm_benchmark" \
        --dimensions "${DIMENSIONS}" \
        --types "${DTYPE}" \
        --warmup ${WARMUP} \
        --measure ${MEASURE} \
        --output "${RESULTS_DIR}/benchmark_${DTYPE}.csv"
    echo ""
done

echo "Merging results..."
echo "dim,dtype,custom_time_ms,custom_tflops,cublas_time_ms,cublas_tflops,ratio,correctness_l2" > "${RESULTS_DIR}/benchmark_all.csv"
cat "${RESULTS_DIR}/benchmark_BF16.csv" "${RESULTS_DIR}/benchmark_FP32.csv" | grep -v "^dim," >> "${RESULTS_DIR}/benchmark_all.csv"

echo ""
echo "Results saved to ${RESULTS_DIR}/"
echo "Generate plots with: python ${SCRIPT_DIR}/plot_results.py ${RESULTS_DIR}/benchmark_all.csv"
