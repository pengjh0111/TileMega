# TileMega

Turning megakernel task partitioning, event generation and granularity selection
from hand-supplied configuration into a solvable compilation problem over affine
dependence relations, on top of the NVIDIA CUDA Tile IR / TensorIR stack.

**See [`Tilemega_skeleton.md`](Tilemega_skeleton.md) for the development guide** —
that is the long-lived main document, covering design rationale, platform
constraints, the phased TODO list and measured findings. This README only covers
how to build.

## Build

```bash
git clone --recursive <repo> tilemega && cd tilemega
# if already cloned: git submodule update --init --recursive

# 1) system dependencies
sudo apt-get install -y libgmp-dev libntl-dev automake libtool autoconf ccache

# 2) isl + barvinok (autotools, built separately, ~3 minutes)
./scripts/build_barvinok.sh

# 3) main build (first run compiles LLVM/MLIR from source, ~1-2 hours)
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT=/data/cuda-13.3.1
ninja -C build

# 4) tests
ctest --test-dir build --output-on-failure
```

> **Disk**: the LLVM source tree plus its build needs roughly 15 GB. Make sure
> the partition holding `build/` has room.

## Dependency pins

The three dependencies do not each track `main`. They form a set of pins that
**must be updated together**; `third_party/tensor-ir/cmake/TensorIRDependencyPins.cmake`
is the source of truth:

| Dependency | Commit | Notes |
|---|---|---|
| tensor-ir | `63692d79` | GitHub main |
| cuda-tile | `af241704` | = CUDA Tile IR 13.3.3 release |
| LLVM | `57109bef` | required by both of the above |
| barvinok | `dd7e6d83` | 0.41.9, bundles isl 0.28 |

`cmake/TileMegaVersionGuard.cmake` asserts these relationships at configure time
and fails the build if they do not hold. See P0.1 in the skeleton document.

## Tools

| Tool | Purpose |
|---|---|
| `build/tools/tilemega-opt/tilemega-opt` | opt driver registering both `nv_tensor_ir` and `cuda_tile` (upstream `tensor_ir-opt` does not register `cuda_tile`) |
| `build/tools/tilemega-occupancy/tilemega-occupancy` | occupancy query that reads the required block size from the kernel instead of guessing it |
| `scripts/tilemega-compile` | MLIR to cubin in one step, emitting a toolchain environment snapshot alongside each artifact |
| `scripts/gpu_stat_run.sh` | statistical GPU test runner (N>=50 runs, per-run timeout, exclusive GPU access) |
| `scripts/sass_report.sh` | SASS structure report (barriers, `CCTL.IVALL`, loop back edges) |
| `scripts/hang_probe.sh` | hang forensics (deadlock vs livelock discrimination) |
