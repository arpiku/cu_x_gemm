#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

echo "=== Building R3X Tuner ==="
cd "$BUILD_DIR"
make cu_x_gemm_r3x_tuner -j$(nproc)

echo ""
echo "=== Running R3X Tuner ==="
./cu_x_gemm_r3x_tuner

echo ""
echo "=== Results ==="
if [ -f "$PROJECT_DIR/results/r3x_tuner.csv" ]; then
    echo "CSV: $PROJECT_DIR/results/r3x_tuner.csv"
    echo ""
    head -20 "$PROJECT_DIR/results/r3x_tuner.csv"
fi
