#!/bin/bash
# Pull H100 results and PTX artifacts from a Runpod host.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/pull_runpod_artifacts.sh HOST PORT [--key PATH] [--remote-path PATH]

Defaults:
  --key         ~/.ssh/id_ed25519
  --remote-path ~/cu_x_gemm

Copies remote:
  results/h100/
  ptx/

Archives local copies before overwrite:
  results/h100/ -> results/archive/h100/<timestamp>/
  ptx/          -> ptx.archive/<timestamp>/
EOF
}

if [[ $# -lt 2 ]]; then
    usage >&2
    exit 1
fi

HOST="$1"
PORT="$2"
shift 2

KEY_PATH="${HOME}/.ssh/id_ed25519"
REMOTE_PATH="~/cu_x_gemm"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --key)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --key" >&2
                exit 1
            fi
            KEY_PATH="$2"
            shift 2
            ;;
        --remote-path)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --remote-path" >&2
                exit 1
            fi
            REMOTE_PATH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! -f "${KEY_PATH}" ]]; then
    echo "SSH key not found: ${KEY_PATH}" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."

timestamp="$(date +%Y%m%d_%H%M%S)"

archive_dir() {
    local src="$1"
    local archive_root="$2"
    local name="$3"

    if [[ -e "${src}" ]]; then
        mkdir -p "${archive_root}"
        mv "${src}" "${archive_root}/${timestamp}"
        echo "Archived ${name} -> ${archive_root}/${timestamp}"
    fi
}

echo "=== Archiving local artifacts ==="
archive_dir "${ROOT_DIR}/results/h100" "${ROOT_DIR}/results/archive/h100" "results/h100"
archive_dir "${ROOT_DIR}/ptx" "${ROOT_DIR}/ptx.archive" "ptx"

mkdir -p "${ROOT_DIR}/results"

echo "=== Pulling remote artifacts ==="
scp -r -i "${KEY_PATH}" -P "${PORT}" \
    "${HOST}:${REMOTE_PATH}/results/h100" \
    "${ROOT_DIR}/results/"

scp -r -i "${KEY_PATH}" -P "${PORT}" \
    "${HOST}:${REMOTE_PATH}/ptx" \
    "${ROOT_DIR}/"

echo "=== Done ==="
echo "Local results: ${ROOT_DIR}/results/h100"
echo "Local PTX: ${ROOT_DIR}/ptx"
