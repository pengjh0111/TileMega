# DNN / SM90R6 integration into tilemega

Verified source snapshots (2026-11-16):

| Source | Commit | Scope |
| --- | --- | --- |
| tilemega | `3b0797b41dfbf9b5a086b37b2176f42e212bcda8` | Existing SM90X1 delivery retained |
| dnn4 | `afcab4a1913aaafc04e57e13a3f82767c7e18a58` | Includes DNN1–3, production F3 and admitted BERT follow-ups |
| sm90r6 | `d47297f9b60b27da8a7a5995e2fa5813c8a21b59` | Compiler, validation tools and audited result tables |

Verified: all source heads above and DNN1/DNN2/DNN3 are ancestors of this
integration. The DNN, R6 and original X1 result files remain byte-identical to
their source branches; no performance samples or targets were replaced.

## Delivered code and configurations

Verified: production `Sm90F3*` headers and `ForwardF3Codegen.cpp` are identical to
dnn4. Dense native bodies/runtime and native solver sources are identical to
sm90r6. Shared CLI code retains DNN `--abi`/`--dump-plan` and R6 native probe
options. Conflict resolution preserves X1's explicit scheduler override, R6's
bounded preflight scheduler, both trace-output namespaces, and the CPU build-lock
override together with R6 lock pools. Original requested locks remain in receipts.

Verified: [BERT profiles](../../configs/dnn/dnn4/README.md) now include all five
admitted coordinate/environment recipes. Actual merged-compiler emission from
these committed files matches the validated native rebuild. They require the
complete environment profile together with the coordinate file and `--abi f3`.

Verified exclusions: uncommitted private DSM, statistics-TMA and reader-policy
prototypes are absent. `sm90r6-ab4-hostlaunch` is not an ancestor; its
performance-only runtime option is absent. Existing disabled diagnostic options
keep their source defaults. Source branches and private worktrees are retained.
Raw measurements, binaries, generated CUDA and dependency overlays remain under
ignored `runs/`.

## Validation

Verified: fresh CMake configuration and complete selected host/CLI/core builds
succeed, followed by the `tilemega-unit` build. The original MLIR installation
lacks imported CAPI objects; a private prefix reconstructs 94 ELF objects
byte-for-byte from the matching installed archives and validates 508 imported
files. The original installation is unchanged. The first configuration failure
and the explicit pinned lit-driver follow-up remain in raw logs.

Verified: **23/23 selected ctests pass**: ten coordinate/cost/native pipeline,
flow/CTG/MoE/sweep checks and thirteen DNN/frontend/semantic/memory checks.
Two additional actual native-generation tests pass across 14 dense/MoE CUDA
programs, checking recorded execution graphs and typed operand contracts.
This is not a claim that the complete CPU/CUDA ctest suite passed.

Verified: all five admitted BERT profiles rebuild with zero spills and exactly
identical encoded GPU instruction streams. B2 differs from historical CUDA only
in the order of independent numeric prefix macros; the full remaining source and
bindings are identical. The first literal-source rejection is retained; the
follow-up compiles the actual new source, with no binary substitution. Existing
B1 CPU build evidence is explicitly reused. These are codegen/compilation checks,
not new GPU correctness or timing measurements.

Verified: the unfrozen-baseline test now uses a temporary authorization fixture
and passes 7/7 CPU cases, including paused and unauthorized-device rejection.
The frozen production collection worker is unchanged.

Verified broad Python audit: **83/89 modules pass** after the two explicit follow-ups;
476 testcase executions are recorded. Six modules retain historical/environment issues:

| Module | Retained issue |
| --- | --- |
| dnn1_aitemplate_profiler_devices | Archived AIT source absent in a clean worktree; source branch with that archive passes |
| dnn1_collection_gpu_pool | Historical fixtures disagree with the later seven-card collection policy |
| dnn1_device | Historical device-query/floor fixtures disagree with the later policy |
| dnn1_owned_controller | Existing CPU-waiter cancellation fixture fails on both source and integration |
| dnn1_tvm_infra_driver | Historical pinned-driver hash rejects the later baseline driver |
| dnn1_tvm_reuse_driver | Same historical pinned-driver mismatch |

Verified: test/helper content is identical to dnn4 for all six. Five failures are
reproduced on dnn4; the AIT archive dependency is isolated to environment. No
resource guard, provenance pin, numerical tolerance or historical reference was
weakened to obtain a pass. Standalone scripts not covered by unittest are listed
separately and are not counted as passing tests.

## Latest admitted BERT performance

Verified historical H800/BF16/sequence128 latency in ms. Each median uses three
fresh 50-warmup/200-event processes; historical cards, fixed targets and the
container-PID qualification remain in [the authoritative CSV](DNN4_BERT_current.csv).

| Batch | Median | P90 | Fixed DNN4 target |
| ---: | ---: | ---: | ---: |
| 1 | 0.464496 | 0.477696 | 0.485840 |
| 2 | 0.508096 | 0.511104 | 0.517104 |
| 4 | 0.572032 | 0.575680 | 0.590496 |
| 8 | 0.753536 | 0.762560 | 0.745712 |
| 16 | 1.145920 | 1.152448 | 1.086864 |

Verified: B1/B2/B4 beat their targets; B8/B16 remain slower by 1.049%/5.434%.
The merge does not claim DNN4 completion, a P-MODEL pass, or full R6 acceptance.
R6's DT-P gap and output/protocol exclusions remain as published in
[SM90R6_h800.md](SM90R6_h800.md). No hardware performance was recollected here.

Raw integration evidence: `runs/integration_dnn_r6_20261116/integration_audit.json`,
`build.log`, `unit_build.log`, `ctest_cpu.log`, `ctest_dnn.log`, `python_v3/`,
`parent_failures/`, `native_generated_features/`, `bert_rebuild/`, `profile_pack.json`.
