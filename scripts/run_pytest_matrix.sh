#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_VERSIONS=("3.8" "3.9" "3.10" "3.11" "3.12")

usage() {
  echo "Usage:"
  echo "  ./scripts/run_pytest_matrix.sh"
  echo "  ./scripts/run_pytest_matrix.sh all"
  echo "  ./scripts/run_pytest_matrix.sh single <python-version>"
}

is_supported_version() {
  local candidate="$1"
  local version
  for version in "${PYTHON_VERSIONS[@]}"; do
    if [[ "${version}" == "${candidate}" ]]; then
      return 0
    fi
  done
  return 1
}

run_for_version() {
  local version="$1"
  echo "==> Running tests on Python ${version}"
  uv python install "${version}"
  uv run \
    --python "${version}" \
    --no-project \
    --with-requirements requirements.txt \
    --with-requirements requirements-dev.txt \
    pytest -q
}

MODE="${1:-all}"

case "${MODE}" in
  all)
    if [[ $# -gt 1 ]]; then
      usage
      exit 2
    fi
    for version in "${PYTHON_VERSIONS[@]}"; do
      run_for_version "${version}"
    done
    ;;
  single)
    if [[ $# -ne 2 ]]; then
      usage
      exit 2
    fi
    VERSION="$2"
    if ! is_supported_version "${VERSION}"; then
      echo "Unsupported Python version: ${VERSION}"
      echo "Supported versions: ${PYTHON_VERSIONS[*]}"
      exit 2
    fi
    run_for_version "${VERSION}"
    ;;
  *)
    usage
    exit 2
    ;;
esac
