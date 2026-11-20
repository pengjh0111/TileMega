#!/bin/bash
# Build inside the CUDA 12.8 Apptainer login environment. All inputs are staged offline.
set -euo pipefail
src=/work/software/mpk-63af900
tools=/work/software/mpk-tools
stage=/work/.incoming/mpk
venv=/work/envs/mpk
cd "$src"
[[ $(git rev-parse HEAD) == 63af900000d8928a885f7af7889a3448621e18a9 ]]
export CUDA_VISIBLE_DEVICES="" CUDAARCHS=80 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
export PATH="$venv/bin:$tools/rust/bin:$PATH"
export CARGO_HOME="$tools/cargo" CARGO_BUILD_JOBS=2 PYO3_PYTHON="$venv/bin/python"
export LD_LIBRARY_PATH="/work/.cpu-stub:${LD_LIBRARY_PATH:-}"
mkdir -p "$tools/cache" "$CARGO_HOME"
for part in rustc cargo rust-std; do
    sha=$(awk 'NR==1 {print $1}' "$stage/$part.sha256")
    printf '%s  %s\n' "$sha" "$stage/$part.tar.xz" | sha256sum -c -
    if [[ ! -f $tools/rust/.installed-$part ]]; then
        dir="$tools/cache/$part-1.88.0-x86_64-unknown-linux-gnu"
        tar xf "$stage/$part.tar.xz" -C "$tools/cache"
        bash "$dir/install.sh" --prefix="$tools/rust" --disable-ldconfig >/dev/null
        touch "$tools/rust/.installed-$part"
    fi
done
if [[ ! -f $tools/.vendor-ready ]]; then
    tar xzf "$stage/mpk-vendor.tar.gz" -C "$tools"
    touch "$tools/.vendor-ready"
fi
cat >"$CARGO_HOME/config.toml" <<EOF
[source.crates-io]
replace-with = "pinned"
[source.pinned]
directory = "$tools/vendor"
EOF
rustc --version
for spec in abstract_subexpr:abstract_expr/abstract_subexpr formal_verifier:verification/formal_verifier_equiv; do
    name=${spec%%:*}
    cargo build --locked --offline -j 2 --release --manifest-path "src/search/${spec#*:}/Cargo.toml" --target-dir "$src/build/$name"
done
z3=$(python -c 'from pathlib import Path; import z3; print(Path(z3.__file__).parent)')
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DZ3_CXX_INCLUDE_DIRS="$z3/include" -DZ3_LIBRARIES="$z3/lib/libz3.so" \
    -DABSTRACT_SUBEXPR_LIB="$src/build/abstract_subexpr/release" \
    -DFORMAL_VERIFIER_LIB="$src/build/formal_verifier/release"
cmake --build build --target mirage_runtime --parallel 2
MIRAGE_SKIP_NATIVE_BUILD=1 python setup.py build_ext --inplace -j 2
python -c 'import mirage; from mirage.mpk.persistent_kernel import PersistentKernel; print(mirage.__file__)'
git diff --binary | sha256sum >"$tools/source-diff.sha256"
sha256sum python/mirage/core*.so >"$tools/extension.sha256"
