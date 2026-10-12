# TileMega

**TileMega** is a compiler framework that automatically constructs
tile-level execution graphs and generates persistent megakernels for deep
learning inference, including decoder LLMs, mixture-of-experts models and
conventional DNNs. At its core is a **parameterized Coupling Graph (CG)**
that captures task spaces, inter-operator dependencies, synchronization
requirements, and execution placement while preserving their dependence on
model shapes and task granularity. TileMega uses **ISL-based symbolic**
dependence analysis over parameterized task spaces and tensor access
relations to derive coupling properties, combines hardware-aware models and
search to explore execution configurations, and uses CuTe/CUTLASS to realize
high-performance task implementations, ultimately lowering the scheduled CG
to a CUDA persistent megakernel.

## Build

Dependency versions and optional environments are listed in
[DEPENDENCIES.md](docs/DEPENDENCIES.md). Use an MLIR install tree;
[BUILD_MLIR.md](docs/BUILD_MLIR.md) describes its construction.

On Ubuntu or Debian, install the host-side prerequisites first:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git cmake ninja-build \
  python3 autoconf automake libtool libtool-bin pkg-config m4 \
  libgmp-dev libntl-dev
```

Initialize only the submodules used by the build. The nested barvinok URLs use
the `git://` protocol upstream; overriding them with HTTPS avoids failures on
networks that block port 9418. `pet` is intentionally not initialized.

```bash
git submodule update --init third_party/cutlass third_party/barvinok
git -C third_party/barvinok config \
  submodule.isl.url https://repo.or.cz/isl.git
git -C third_party/barvinok config \
  submodule.polylib.url https://repo.or.cz/polylib.git
git -C third_party/barvinok submodule update --init isl polylib
```

Build the pinned polyhedral stack out of tree. The `autogen.sh` steps must run
in their source directories; they may create untracked autotools helper files
inside the polylib submodule, but those generated files must not be committed.

```bash
(cd third_party/barvinok/isl && ./autogen.sh)
mkdir -p build-isl
(cd build-isl && ../third_party/barvinok/isl/configure \
  --with-int=gmp --enable-shared=no --enable-static=yes --with-pic \
  CFLAGS="-O2 -fPIC" CXXFLAGS="-O2 -fPIC" && make -j"$(nproc)")

(cd third_party/barvinok/polylib && ./autogen.sh)
mkdir -p build-polylib
(cd build-polylib && ../third_party/barvinok/polylib/configure \
  --with-libgmp=/usr --enable-shared=no --enable-static=yes --with-pic \
  CFLAGS="-O2 -fPIC" && make -j"$(nproc)")

(cd third_party/barvinok && ./autogen.sh)
mkdir -p build-barvinok
(cd build-barvinok && ../third_party/barvinok/configure \
  --with-isl=build --with-isl-builddir=../build-isl \
  --with-polylib=build --with-polylib-builddir=../build-polylib \
  --with-pet=no --with-gmp=system \
  --enable-shared=no --enable-static=yes --with-pic \
  CFLAGS="-O2 -fPIC" CXXFLAGS="-O2 -fPIC" && make -j"$(nproc)")
```

Build and install the pinned LLVM/MLIR once, then configure TileMega against
that relocatable install tree. Replace `/path/to/prefix` with a persistent
absolute path. Select the deployment architecture explicitly for a portable or
cross-compiled build; leave it as `auto` only for a native build.

```bash
# Build and install the pinned LLVM/MLIR (once).
scripts/build_mlir.sh /path/to/prefix

cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/path/to/prefix/lib/cmake/mlir \
  -DTILEMEGA_TARGET_ARCH=auto \
  -DTILEMEGA_ENABLE_MLIR=ON \
  -DTILEMEGA_ENABLE_ISL=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The three polyhedral build directories default to `build-isl`,
`build-polylib`, and `build-barvinok`. Relocated builds can be supplied with
`-DTILEMEGA_ISL_BUILD_DIR=...`, `-DTILEMEGA_POLYLIB_BUILD_DIR=...`, and
`-DTILEMEGA_BARVINOK_BUILD_DIR=...`. See `docs/DEPENDENCIES.md` for the
versions and environment pins.

CMake options:

| Option | Default | Meaning |
| --- | --- | --- |
| `TILEMEGA_BUILD_TESTS` | `ON` | unit tests and the CG dialect `lit` suite |
| `TILEMEGA_BUILD_VERIFY` | `ON` | verification experiments, including `crosscompile-matrix` |
| `TILEMEGA_ENABLE_MLIR` | `ON` | required; `OFF` is rejected with an error |
| `TILEMEGA_ENABLE_ISL` | `ON` | required; `OFF` is rejected with an error |
| `TILEMEGA_TARGET_ARCH` | `auto` | `auto`, `sm_80`, `sm_89`, `sm_90`, `sm_100`, `sm_120` |
| `TILEMEGA_TARGET_CONFIG` | — | path to a target JSON, overriding `TILEMEGA_TARGET_ARCH` |
| `TILEMEGA_LIT_DRIVER` | derived | path to `lit.py`; needed only when `LLVMConfig` does not export `LLVM_BUILD_MAIN_SRC_DIR` |

Additional targets:

```bash
ninja -C build check-cg-lit           # CG dialect parse/verify/print tests
ninja -C build check-policy           # rejects __CUDA_ARCH__ outside
                                      # ArchDispatch.h, sm-version comparisons
                                      # and hardware resource literals
ninja -C build crosscompile-matrix    # compile one architecture-neutral source
                                      # for supported architectures and record the
                                      # ptxas resource matrix
```

`crosscompile-matrix` requires `TILEMEGA_BUILD_VERIFY=ON`, which is the default;
it will not exist as a target if verification was configured off.

## Usage

Install the serving environment from `requirements/serving.txt` with a CUDA
wheel appropriate to the deployment system. Model repositories and revisions
are pinned in `configs/models/`. For Llama-3.2-1B:

```bash
export PYTHONPATH="$PWD/python"
python -m tilemega.serving.download_models --model llama3_2_1b --model-root /root/models
python -m tilemega doctor --config configs/e2e/llama3_2_1b.json
python -m tilemega run --config configs/e2e/llama3_2_1b.json
```

Set model paths and ISL, OSL and batches in the configuration. Multiple request
lengths use `workload.points`; `configs/e2e/sweep.json` is an example. These are
search configurations; accepted defaults require the correctness and guarded
performance gates. Raw output belongs under ignored `runs/`.

DNN forward uses a separate reference environment from `requirements/dnn-*`:

```bash
python -m tilemega dnn build --config configs/dnn/resnet18.json
python -m tilemega dnn run --config configs/dnn/resnet18.json
```

Set the checkpoint and dataset paths in the DNN configuration before running.
The full MoE script needs an 80 GB GPU; on a smaller device, dry-only generates
plans and records the memory preflight without executing the model:

```bash
TILEMEGA_DRY_ONLY=1 scripts/run_qwen3_moe_full.sh
# On an 80 GB device:
scripts/run_qwen3_moe_full.sh
```

The lower-level flow is export → import → compile:

```bash
python python/tilemega/export_bridge.py model.pt2 --out export.json
build/tools/tilemega inspect import export.json > cg.mlir
build/tools/tilemega-opt cg.mlir
build/tools/tilemega compile cg.mlir out.cu
```

A `.cu` output generates source; `.so` also invokes NVCC. See
[TOOLS.md](docs/TOOLS.md) for build identity, selection, measurement and cache
interfaces. On shared GPUs, use one `TILEMEGA_GPU_LOCK` for builds and tests;
all timing runs through the guarded benchmark scheduler.
