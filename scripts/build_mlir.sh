#!/usr/bin/env bash
# Build and install the pinned LLVM/MLIR that the Coupling Graph dialect needs.
#
#   scripts/build_mlir.sh <install-prefix> [source-dir] [build-dir]
#
# Installs a relocatable package; see docs/BUILD_MLIR.md for why a build tree
# must not be consumed directly.
set -euo pipefail

PREFIX=${1:?usage: build_mlir.sh <install-prefix> [source-dir] [build-dir]}
SRC=${2:-${TMPDIR:-/tmp}/llvm-project}
BUILD=${3:-${TMPDIR:-/tmp}/llvm-build}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
COMMIT=$(cat "${REPO_ROOT}/third_party/cutlass/cutlass_compiler/LLVM_COMMIT")

# A full shallow clone of llvm-project through a proxy has failed near completion
# before (docs/experiments/V_F/raw/fetch_attempts.txt); fetching the exact commit
# is what worked.
if [ ! -d "${SRC}/.git" ]; then
  mkdir -p "${SRC}"
  git -C "${SRC}" init -q
  git -C "${SRC}" remote add origin https://github.com/llvm/llvm-project.git
fi
git -C "${SRC}" fetch --depth 1 origin "${COMMIT}"
git -C "${SRC}" checkout -q FETCH_HEAD

cmake -S "${SRC}/llvm" -B "${BUILD}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX" \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}"

ninja -C "${BUILD}"
ninja -C "${BUILD}" install

echo
echo "MLIR installed. Configure TileMega with:"
echo "  cmake -S . -B build -G Ninja -DMLIR_DIR=${PREFIX}/lib/cmake/mlir"
