#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
BUILD_DIR=${TMPDIR:-/tmp}/llama-capture-store-tsan
COMPILER=${CXX:-g++}
OUTPUT=${BUILD_DIR}/test-server-capture-store-tsan
SHA_SOURCE=${PROJECT_ROOT}/tools/server/server-capture-sha256.cpp

# Keep this source list synchronized with test-server-capture-store's CMake
# target: capture-store now calls the shared standard SHA-256 implementation.
if [ ! -f "${SHA_SOURCE}" ]; then
    echo "missing capture-store SHA-256 source: ${SHA_SOURCE}" >&2
    exit 1
fi

mkdir -p "${BUILD_DIR}"
"${COMPILER}" -std=c++17 -g -O1 -fsanitize=thread -fno-omit-frame-pointer -pthread \
    -I"${PROJECT_ROOT}/tools/server" -I"${PROJECT_ROOT}/ggml/include" \
    "${PROJECT_ROOT}/tools/server/server-capture.cpp" \
    "${SHA_SOURCE}" \
    "${PROJECT_ROOT}/tools/server/server-capture-store.cpp" \
    "${PROJECT_ROOT}/tests/test-server-capture-store.cpp" \
    -o "${OUTPUT}"

TSAN_OPTIONS=${TSAN_OPTIONS:-halt_on_error=1} "${OUTPUT}"
