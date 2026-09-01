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
CUDA Toolkit, and the CUTLASS submodule. MLIR and ISL are disabled by default.
Barvinok is recorded as a submodule for later polyhedral work and is not built.

```bash
git submodule update --init --recursive
cmake -S . -B build -G Ninja
ninja -C build
ctest --test-dir build --output-on-failure
```

Relevant CMake options are:

- `TILEMEGA_BUILD_TESTS=ON`
- `TILEMEGA_BUILD_VERIFY=ON`
- `TILEMEGA_ENABLE_MLIR=OFF`
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

This repository is at the pre-construction verification stage. `TargetSpec`,
architecture dispatch, TaskBody contracts, CUDA synchronization paths, and the
verification targets are implemented. Frontend importing, Coupling Graph
lowering, the production solver, most TaskBody operators, and the final runtime
launcher remain explicit phase stubs.

## Verification

See [docs/VERIFICATION_PLAN.md](docs/VERIFICATION_PLAN.md) for status, evidence
labels, commands, and links to every experiment.
