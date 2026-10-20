#!/usr/bin/env bash
# Build one requested component; no aggregate target, install or test matrix.
set -eo pipefail
source /home/d50064638/junhuip/TileMega/scripts/as1_env.sh
set -u
component=${1:-}
case "$component" in
  m4|gmp|isl|ntl|polylib|barvinok|mlir-tblgen|mlir-ir|mlir-pass|core) ;;
  *) printf '%s\n' 'usage: as1_build_host.sh m4|gmp|isl|ntl|polylib|barvinok|mlir-tblgen|mlir-ir|mlir-pass|core' >&2; exit 2 ;;
esac
python3 "$REPO/scripts/as1_disk_status.py" > "$AS1_WORK/reports/disk_before_host_build.json"
prefix="$AS1_WORK/envs/host"
export PKG_CONFIG_PATH="$prefix/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
log="$AS1_WORK/logs/host-$component-$(date +%Y%m%dT%H%M%S%N).log"
printf 'Single-component build, parallel=1; log: %s\n' "$log"
exec > "$log" 2>&1
printf 'component=%s max_compile_parallel=1\n' "$component"
build_dir="$AS1_WORK/build/$component"
case "$component" in
  polylib) build_dir="$AS1_WORK/build/polylib5" ;;
  barvinok) build_dir="$AS1_WORK/build/barvinok-poly5" ;;
  mlir-*) build_dir="$AS1_WORK/build/llvm" ;;
  core) build_dir="$AS1_WORK/build/tilemega" ;;
esac
mkdir -p "$build_dir"
trap 'status=$?; python3 "$REPO/scripts/as1_register.py" --path "$build_dir" --stage AS-B --category build --completed --pid "$$" --build-id "host-$component-$(basename "$log")"; exit "$status"' EXIT
if [[ "$component" == ntl && ! -f "$build_dir/src/configure" ]]; then
  python3 - <<'PY'
import os, shutil
from pathlib import Path
w=Path(os.environ['AS1_WORK']); d=w/'build/ntl'
if list(d.iterdir()): raise RuntimeError('Refusing to populate an unfamiliar NTL build directory')
shutil.copytree(w/'third_party/ntl-11.5.1',d,dirs_exist_ok=True)
PY
fi
cd "$build_dir"
case "$component" in
  m4)
    "$AS1_WORK/third_party/m4-1.4.19/configure" --prefix="$prefix" --disable-nls
    nice -n 10 /usr/bin/time -v make -j1
    make -j1 install ;;
  gmp)
    "$AS1_WORK/third_party/gmp-6.2.1/configure" --prefix="$prefix" --disable-shared --enable-static --with-pic --disable-assembly --enable-cxx
    nice -n 10 /usr/bin/time -v make -j1
    make -j1 install ;;
  isl)
    "$AS1_WORK/third_party/isl-0.28/configure" --prefix="$prefix" --with-gmp-prefix="$prefix" --disable-shared --enable-static --with-pic --with-int=gmp
    make -j1 gitversion.h
    nice -n 10 /usr/bin/time -v make -j1 libisl.la
    make -j1 install-libLTLIBRARIES install-pkgincludeHEADERS install-nodist_pkgincludeHEADERS install-pkgconfigDATA ;;
  ntl)
    cd src
    ./configure PREFIX="$prefix" GMP_PREFIX="$prefix" CXXFLAGS='-O2 -fPIC' NATIVE=off NTL_THREADS=off NTL_THREAD_BOOST=off TUNE=generic
    nice -n 10 /usr/bin/time -v make -j1
    make -j1 install ;;
  polylib)
    "$AS1_WORK/third_party/polylib-5.22.8/configure" --prefix="$prefix" --with-libgmp="$prefix" --disable-shared --enable-static --with-pic --disable-int-lib --disable-longint-lib --disable-longlongint-lib
    nice -n 10 /usr/bin/time -v make -j1 libpolylibgmp.la
    make -j1 install-libLTLIBRARIES
    make -C include -j1 install ;;
  barvinok)
    # Debian's source repack omits zsolve. Build the cardinality library only.
    CONFIG_FILES='Makefile doc/Makefile' CONFIG_HEADERS=config.h CPPFLAGS="-I$prefix/include" LDFLAGS="-L$prefix/lib" \
      "$AS1_WORK/third_party/barvinok-0.41.9/configure" --prefix="$prefix" --disable-shared --enable-static --with-pic \
      --with-gmp=system --with-gmp-prefix="$prefix" --with-polylib=system --with-polylib-prefix="$prefix" \
      --with-isl=system --with-isl-prefix="$prefix" --with-ntl-prefix="$prefix" --with-pet=no --without-glpk --without-cddlib --without-zsolve
    nice -n 10 /usr/bin/time -v make -j1 libbarvinok.la
    make -j1 install-libLTLIBRARIES install-pkgincludeHEADERS install-nodist_pkgincludeHEADERS ;;
  mlir-*)
    [[ $(git -C "$AS1_WORK/third_party/llvm-project" rev-parse HEAD) == 23a60f15f2fcafcf67b95b0a035053579958b732 ]]
    cmake -S "$AS1_WORK/third_party/llvm-project/llvm" -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD=Native -DLLVM_ENABLE_ASSERTIONS=OFF \
      -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF \
      -DMLIR_INCLUDE_TESTS=OFF -DMLIR_INCLUDE_INTEGRATION_TESTS=OFF -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
      -DLLVM_BUILD_TOOLS=OFF -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_LIBXML2=OFF \
      -DLLVM_ENABLE_TERMINFO=OFF -DCMAKE_INSTALL_PREFIX="$AS1_WORK/envs/llvm"
    case "$component" in mlir-tblgen) target=mlir-tblgen;; mlir-ir) target=MLIRIR;; mlir-pass) target=MLIRPass;; esac
    nice -n 10 /usr/bin/time -v cmake --build "$build_dir" --target "$target" --parallel 1 ;;
  core)
    cmake -S "$REPO" -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS_DEBUG='-O0 -g0' \
      -DMLIR_DIR="$AS1_WORK/build/llvm/lib/cmake/mlir" -DLLVM_DIR="$AS1_WORK/build/llvm/lib/cmake/llvm"
    nice -n 10 /usr/bin/time -v cmake --build "$build_dir" --target tilemega_core --parallel 1 ;;
esac
