#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${JUNO_HARNESS_BUILD_DIR:-${ROOT_DIR}/build}"
BUILD_TYPE="${JUNO_HARNESS_BUILD_TYPE:-Release}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"
CTEST_BIN="${CTEST_BIN:-ctest}"

"${CMAKE_BIN}" -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DJUNO_HARNESS_BUILD_TESTS=ON \
  -DJUNO_HARNESS_BUILD_EXAMPLES=ON \
  -DJUNO_HARNESS_ENABLE_LLAMA_CPP="${JUNO_HARNESS_ENABLE_LLAMA_CPP:-ON}" \
  -DFETCHCONTENT_QUIET=OFF \
  "$@"

"${CMAKE_BIN}" --build "${BUILD_DIR}" --parallel

if [[ "${JUNO_HARNESS_SKIP_TESTS:-0}" != "1" ]]; then
  "${CTEST_BIN}" --test-dir "${BUILD_DIR}" --output-on-failure
fi
