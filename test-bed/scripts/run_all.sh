#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
BUILD_DIR="${ROOT_DIR}/test-bed/build"

SIZES="64 128 256 512"
KERNELS=(
    "01_baseline_r1y"
    "02_transposed_layout"
    "03_32x32_tiles"
    "04_128x64_tiles"
    "05_warp_tiled"
)

echo "=== Building test-bed kernels ==="
cd "${ROOT_DIR}"
rm -rf test-bed/build
cmake -B test-bed/build -S test-bed
cmake --build test-bed/build --parallel

echo ""
echo "=== Running bank conflict analysis ==="
echo ""

for kernel in "${KERNELS[@]}"; do
    echo "=== $kernel ==="
    for size in $SIZES; do
        echo "--- Size: $size ---"
        if [ -f "${BUILD_DIR}/${kernel}" ]; then
            "${BUILD_DIR}/${kernel}" $size
            echo ""
        fi
    done
    echo ""
done

echo "=== Done ==="