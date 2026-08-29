#!/usr/bin/env bash
# 构建 isl + barvinok 到 third_party/.install/barvinok。
#
# 为什么单独一个脚本而不进 CMake 构建图（见 cmake/TileMegaBarvinok.cmake）：
# barvinok 是 autotools 工程，用 ExternalProject 塞进 ninja 会带来难看的顺序
# 依赖，而它是个几乎不变的叶子依赖，一次构建长期复用。
#
# 依赖（Ubuntu）：libgmp-dev libntl-dev automake libtool autoconf
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

# autogen 必须在源码树里跑（它生成 configure）。
cd "$SRC"
[ -f configure ] || ./autogen.sh

# 但**构建放在源码树之外**：barvinok 是 submodule，就地构建会在里面留下
# .libs/.deps/生成的可执行文件，让 `git status` 常年脏，也容易误提交。
BUILD_DIR="${TILEMEGA_BARVINOK_BUILD_DIR:-$ROOT/build-barvinok}"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# --with-pet=no  : pet 是 C 源码解析器，TileMega 从 MLIR 拿 IR，用不到。
# --enable-shared: Python 侧后续可能需要。
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
