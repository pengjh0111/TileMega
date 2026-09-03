# TileMega

**TileMega** is a compiler framework for automatically constructing tile-level execution graphs and generating persistent megakernels for LLM inference. At its core is a **parameterized Coupling Graph (CG)** that captures task spaces, inter-operator dependencies, synchronization requirements, and execution placement while preserving their dependence on model shapes and task granularity. TileMega uses **ISL-based symbolic** dependence analysis over parameterized task spaces and tensor access relations to derive coupling properties, combines hardware-aware models and search to explore execution configurations, and uses CuTe/CUTLASS to realize high-performance task implementations, ultimately lowering the scheduled CG to a CUDA persistent megakernel.

## Architecture

<p align="center">
  <img src="docs/TileMega_arch.svg" width="100%">
</p>

## Dependencies

| Dependency | Version | Notes |
| --- | --- | --- |
| C++ compiler | C++17 | |
| CMake | 3.22 or newer | |
| Ninja | any | generator used throughout |
| CUDA Toolkit | provides `nvcc` | required; `sm_80/89/90/120` targets |
| LLVM/MLIR | pinned commit `23a60f15f2fcafcf67b95b0a035053579958b732` (`23.0.0git`) | **required** — the CG dialect is the mandatory L4-to-L1 contract, so `TILEMEGA_ENABLE_MLIR=OFF` is rejected |
| CUTLASS | `third_party/cutlass` submodule | task interiors and collectives |
| Python | 3.8 or newer | `lit` driver, export bridge |
| PyTorch | 2.x | only for the `torch.export` bridge, not for building the compiler |
| barvinok | `third_party/barvinok` submodule | recorded for Phase 3 polyhedral work; **not built** |
| ISL | — | disabled until Phase 3 (`TILEMEGA_ENABLE_ISL=OFF`) |

MLIR must come from an **install tree**. A build tree's CMake package bakes
absolute paths to both the build and source directories and breaks as soon as
either moves; an install tree computes its paths relative to the config file and
survives being relocated. `scripts/build_mlir.sh` builds the pinned commit and
installs it; `docs/BUILD_MLIR.md` covers the details, including how to recover a
build tree that has already been moved.

## Build

```bash
git submodule update --init --recursive

# Build and install the pinned LLVM/MLIR (once).
scripts/build_mlir.sh /path/to/prefix

cmake -S . -B build -G Ninja -DMLIR_DIR=/path/to/prefix/lib/cmake/mlir
ninja -C build
ctest --test-dir build --output-on-failure
```

CMake options:

| Option | Default | Meaning |
| --- | --- | --- |
| `TILEMEGA_BUILD_TESTS` | `ON` | unit tests and the CG dialect `lit` suite |
| `TILEMEGA_BUILD_VERIFY` | `ON` | verification experiments, including `crosscompile-matrix` |
| `TILEMEGA_ENABLE_MLIR` | `ON` | required; `OFF` is rejected with an error |
| `TILEMEGA_ENABLE_ISL` | `OFF` | Phase 3 |
| `TILEMEGA_TARGET_ARCH` | `auto` | `auto`, `sm_80`, `sm_89`, `sm_90`, `sm_120` |
| `TILEMEGA_TARGET_CONFIG` | — | path to a target JSON, overriding `TILEMEGA_TARGET_ARCH` |
| `TILEMEGA_LIT_DRIVER` | derived | path to `lit.py`; needed only when `LLVMConfig` does not export `LLVM_BUILD_MAIN_SRC_DIR` |

Additional targets:

```bash
ninja -C build check-cg-lit           # CG dialect parse/verify/print tests
ninja -C build check-policy           # rejects __CUDA_ARCH__ outside
                                      # ArchDispatch.h, sm-version comparisons
                                      # and hardware resource literals
ninja -C build crosscompile-matrix    # compile one architecture-neutral source
                                      # for sm_80/89/90/120 and record the
                                      # ptxas resource matrix
```

`crosscompile-matrix` requires `TILEMEGA_BUILD_VERIFY=ON`, which is the default;
it will not exist as a target if verification was configured off.

Generating a megakernel from a Coupling Graph:

```bash
# torch.export archive -> stable export JSON
python3 python/tilemega/export_bridge.py model.pt2 --out export.json

# stable export JSON -> Coupling Graph dialect (printed on stdout)
build/tools/tilemega-import export.json > cg.mlir

# parse, verify and round-trip the dialect (a standard mlir-opt driver)
build/tools/tilemega-opt cg.mlir

# Coupling Graph -> generated CUDA -> shared object
build/tools/tilemega-compile cg.mlir out.so
```

`tilemega-compile` accepts either a `.mlir` Coupling Graph or the stable export
JSON directly. Passing a `.cu` output writes only the generated CUDA; passing a
`.so` writes the CUDA alongside it as `out.so.cu` and then invokes `nvcc`
(honouring `CUDACXX`, defaulting to `/usr/local/cuda/bin/nvcc`).
