#!/usr/bin/env bash
# Build isl + barvinok into third_party/.install/barvinok.
#
# Why a standalone script instead of the CMake build graph (see
# cmake/TileMegaBarvinok.cmake): barvinok is an autotools project, and wiring it
# into ninja through ExternalProject creates ugly ordering dependencies, while
# it is an almost unchanging leaf dependency that is built once and reused.
#
# Dependencies (Ubuntu): libgmp-dev libntl-dev automake libtool autoconf
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/third_party/barvinok"
PREFIX="${1:-$ROOT/third_party/.install/barvinok}"
JOBS="${JOBS:-$(nproc)}"

for sub in isl polylib; do
  if [ ! -f "$SRC/$sub/configure.ac" ] && [ ! -f "$SRC/$sub/configure.in" ]; then
    echo "错误：$SRC/$sub 未初始化。执行：" >&2
    echo "  git submodule update --init --recursive third_party/barvinok" >&2
    exit 1
  fi
done

echo "==> barvinok 源码 : $SRC"
echo "==> 安装前缀      : $PREFIX"
echo "==> 并行度        : $JOBS"

# autogen must run inside the source tree (it generates configure).
cd "$SRC"
[ -f configure ] || ./autogen.sh

# But BUILD OUT OF TREE: barvinok is a submodule, and an in-source build leaves
# .libs / .deps / generated executables behind, keeping `git status` permanently
# dirty and inviting accidental commits.
BUILD_DIR="${TILEMEGA_BARVINOK_BUILD_DIR:-$ROOT/build-barvinok}"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# --with-pet=no  : pet is a C source parser; TileMega gets its IR from MLIR and
#                  does not need it.
# --enable-shared: likely needed later for the Python side.
"$SRC/configure" \
  --prefix="$PREFIX" \
  --with-isl=bundled \
  --with-polylib=bundled \
  --with-pet=no \
  --enable-shared \
  --disable-static

make -j"$JOBS"
make install

echo
echo "==> 完成。校验："
ls "$PREFIX/lib/"libisl.* "$PREFIX/lib/"libbarvinok.* 2>/dev/null || {
  echo "警告：预期的库文件没找到" >&2; exit 1; }
echo "现在可以带 -DTILEMEGA_ENABLE_ISL=ON 重新 configure。"
