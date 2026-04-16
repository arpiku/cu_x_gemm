#!/bin/bash
# Build and run the single-variant fp32 pedantic benchmark.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
BUILD_DIR="${ROOT_DIR}/build"

detect_arch_dir() {
    local compute_cap=""
    local major=""

    if command -v nvidia-smi >/dev/null 2>&1; then
        compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '\r ' || true)"
    fi

    major="${compute_cap%%.*}"
    case "${major}" in
        9) echo "h100" ;;
        12) echo "rtx5070" ;;
        *) echo "rtx5070" ;;
    esac
}

ARCH_DIR=""
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

if [[ -z "${ARCH_DIR}" ]]; then
    ARCH_DIR="$(detect_arch_dir)"
fi

RESULTS_DIR="${ROOT_DIR}/results/${ARCH_DIR}"
ARCHIVE_DIR="${ROOT_DIR}/results/archive/${ARCH_DIR}"

archive_previous_run() {
    local timestamp=""
    local snapshot_dir=""
    local archived_any=0
    local files=("${RESULTS_DIR}/benchmark_results.csv" "${RESULTS_DIR}/benchmark.log")

    for file in "${files[@]}"; do
        if [[ -e "${file}" ]]; then
            if [[ ${archived_any} -eq 0 ]]; then
                timestamp="$(date +%Y%m%d_%H%M%S)_$$"
                snapshot_dir="${ARCHIVE_DIR}/${timestamp}"
                mkdir -p "${snapshot_dir}"
            fi
            mv "${file}" "${snapshot_dir}/"
            archived_any=1
        fi
    done

    if [[ ${archived_any} -eq 1 ]]; then
        echo "Archived previous run to ${snapshot_dir}/"
    fi
}

prune_archives() {
    [[ -d "${ARCHIVE_DIR}" ]] || return 0

    shopt -s nullglob
    local -a archives=("${ARCHIVE_DIR}"/*/)
    shopt -u nullglob

    local count=${#archives[@]}
    if [[ ${count} -le 10 ]]; then
        return 0
    fi

    local remove_count=$((count - 10))
    local -a to_remove=("${archives[@]:0:${remove_count}}")
    rm -rf -- "${to_remove[@]}"
}

mkdir -p "${RESULTS_DIR}"
mkdir -p "${ARCHIVE_DIR}"

archive_previous_run

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

prune_archives
