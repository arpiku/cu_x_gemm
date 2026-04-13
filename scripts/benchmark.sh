#!/bin/bash
# Build, run, or export PTX for GEMM benchmark

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
BUILD_DIR="${ROOT_DIR}/build"
ARCH_DIR="unknown"
EXPORT_PTX=0
ARCHIVE_LIMIT=10

KERNEL_SOURCES=(
    "src/gemm_bf16.cu"
    "src/gemm_fp32.cu"
    "src/gemm_fp32_r2z.cu"
    "src/gemm_fp32_r2z_tc0.cu"
    "src/gemm_fp32_r2z_tc1.cu"
)

PTX_TARGETS=()

for arg in "$@"; do
    case "${arg}" in
        -h100)
            ARCH_DIR="h100"
            PTX_TARGETS=("h100")
            ;;
        -rtx5070)
            ARCH_DIR="rtx5070"
            PTX_TARGETS=("rtx5070")
            ;;
        -ptx)
            EXPORT_PTX=1
            ;;
    esac
done

if [[ ${EXPORT_PTX} -eq 1 && ${#PTX_TARGETS[@]} -eq 0 ]]; then
    PTX_TARGETS=("h100" "rtx5070")
fi

ptx_compute_for_arch() {
    case "$1" in
        h100) echo "compute_90" ;;
        rtx5070) echo "compute_120" ;;
        *) return 1 ;;
    esac
}

export_ptx() {
    local ptx_root="${ROOT_DIR}/ptx"
    local nvcc
    nvcc="$(command -v nvcc)"
    if [[ -z "${nvcc}" ]]; then
        echo "nvcc not found in PATH" >&2
        return 1
    fi

    mkdir -p "${ptx_root}"

    local failed=0
    local arch compute out_dir src out_file
    for arch in "${PTX_TARGETS[@]}"; do
        compute="$(ptx_compute_for_arch "${arch}")"
        out_dir="${ptx_root}/${arch}/${compute}"
        mkdir -p "${out_dir}"

        echo "=== Exporting PTX for ${arch} (${compute}) ==="
        for src in "${KERNEL_SOURCES[@]}"; do
            out_file="${out_dir}/$(basename "${src}" .cu).ptx"
            if ! "${nvcc}" -ptx -std=c++17 -O3 -arch="${compute}" \
                -I"${ROOT_DIR}/src" -I"${ROOT_DIR}" \
                "${ROOT_DIR}/${src}" -o "${out_file}"; then
                echo "PTX export failed for ${src} -> ${out_file}" >&2
                failed=1
            fi
        done
        echo "PTX saved to ${out_dir}/"
    done

    return ${failed}
}

if [[ ${EXPORT_PTX} -eq 1 ]]; then
    export_ptx
    exit $?
fi

RESULTS_DIR="${ROOT_DIR}/results/${ARCH_DIR}"
ARCHIVE_DIR="${ROOT_DIR}/results/archive/${ARCH_DIR}"

archive_previous_run() {
    local timestamp snapshot_dir archived_any=0
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
    if [[ ${count} -le ${ARCHIVE_LIMIT} ]]; then
        return 0
    fi

    local remove_count=$((count - ARCHIVE_LIMIT))
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
