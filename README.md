# TileMega

## What it is

TileMega is a compiler framework for lowering a Coupling Graph into a GPU
megakernel. The graph makes cross-operator tile dependencies explicit, while
TaskBody implementations encapsulate architecture-specific tile execution.

## Architecture

The implementation follows the layering in `TileMega_skeleton.md` section
4.1:

```text
torch.export graph + symbolic shapes
                 |
                 v
          Frontend importer
                 |
                 v
        Coupling Graph dialect
                 |
       +---------+---------+
       |                   |
       v                   v
  dependence analysis   target/resource queries
       |                   |
       +---------+---------+
                 v
       candidate generation and solving
                 |
                 v
       CUDA megakernel + TaskBody codegen
                 |
                 v
          runtime launcher
```

Architecture differences are confined to
`include/tilemega/Target/ArchDispatch.h`; runtime resource budgets come from
`TargetSpec`.

## Core abstraction

The Coupling Graph uses five definitions from the design document:

1. A task type describes one tiled operator and its legal tile variants.
2. A task instance is identified by a symbolic coordinate in that task type's
   iteration domain.
3. A coupling edge is an exact producer-to-consumer relation between task
   instances, not merely an operator-level edge.
4. A placement maps task instances to megakernel CTA slots and execution
   order.
5. A synchronization realization maps each coupling edge to an implementation
   tier such as intra-CTA, cluster, or global event synchronization.

The required invariants are single ownership of every produced tile, complete
coverage of consumer predecessors, acyclic realized execution or a proven
streaming wait-for schedule, release/acquire visibility at every cross-CTA
edge, and resource feasibility under the selected `TargetSpec`.

## Build

The default build requires a C++17 compiler, CMake 3.22 or newer, Ninja, the
CUDA Toolkit, the CUTLASS submodule, and an MLIR development build. The CG
dialect is the mandatory L4-to-L1 contract, so MLIR is enabled by default;
ISL remains disabled until Phase 3.
Barvinok is recorded as a submodule for later polyhedral work and is not built.

```bash
git submodule update --init --recursive
cmake -S . -B build -G Ninja -DMLIR_DIR=/path/to/lib/cmake/mlir
ninja -C build
ctest --test-dir build --output-on-failure
```

Relevant CMake options are:

- `TILEMEGA_BUILD_TESTS=ON`
- `TILEMEGA_BUILD_VERIFY=ON`
- `TILEMEGA_ENABLE_MLIR=ON` (required; `OFF` is rejected)
- `TILEMEGA_ENABLE_ISL=OFF`
- `TILEMEGA_TARGET_ARCH=auto|sm_80|sm_89|sm_90|sm_120`
- `TILEMEGA_TARGET_CONFIG=/path/to/target.json`

Run `ninja -C build crosscompile-matrix` to compile the verification skeleton
for all four supported architecture tags.

## Target portability

`TargetSpec::Probe()` reads the installed device at runtime.
`TargetSpec::FromJson()` loads a target without requiring that GPU to be
present, which enables cross compilation. The checked-in
`configs/targets/*.json` files hold resource budgets and explicit capability
switches; `TILEMEGA_TARGET_ARCH` selects one target at configure time.

Migration checklist:

```text
1. Does configs/targets/sm_XX.json exist?
2. Does ArchDispatch.h contain an sm_XX specialization?
3. cmake -DTILEMEGA_TARGET_ARCH=sm_XX
4. ./tools/tilemega-calibrate --out configs/targets/sm_XX.json
5. Rerun docs/experiments/.

No application code changes are involved.
```

The calibration command is intentionally a Phase 4 skeleton today: it probes
the device and writes the schema, but latency and bandwidth fields remain
uncalibrated.

## Repository layout

```text
include/tilemega/       public compiler, codegen, target, and runtime APIs
lib/                    current implementations and phase stubs
configs/targets/        portable target descriptions
tools/                  occupancy and calibration utilities
scripts/                experiment and cubin inspection helpers
test/                   unit-test and experiment harnesses
docs/experiments/       reproducible pre-construction verification
third_party/cutlass/    CUTLASS submodule
third_party/barvinok/   future polyhedral dependency; not built
```

## Status

Phase 2 is implemented for the fixed two-layer Llama validation configuration.
A versioned Python bridge serializes `ExportedProgram`; the C++ importer emits
the verified CG dialect; and `CouplingGraphToCUDA` generates L0.5/L1 CUDA using
the TaskBody library. E2E_GEN matches the handwritten reference bitwise and
passes 50/50 fresh processes. Coupling derivation (Phase 3), the production
solver (Phase 4), fine-grained events, and serving remain explicit future work.

## Verification

See [docs/VERIFICATION_PLAN.md](docs/VERIFICATION_PLAN.md) for status, evidence
labels, commands, and links to every experiment.
