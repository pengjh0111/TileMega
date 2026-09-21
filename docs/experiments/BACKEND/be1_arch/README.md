# BE-1 — the TaskBody ABI is parameterized on the Plan's architecture

Before R8 the harness bound every TaskBody to one architecture in a header:
`ModelHarness.cuh` read `using HarnessArch = cutlass::arch::Sm80;`, so a Plan
solved for sm_89 compiled its bodies against the sm_80 capability table and
nothing anywhere compared the two. `TargetSpec` was already capability-driven
and `Target/ArchDispatch.h` already carried `Caps<Arch>` for Sm80/89/90/100/120;
what was missing was the path from the Plan to the instantiation.

## The path, end to end

| step | where | what it carries |
|---|---|---|
| solve | `PlacementSolvePass.h`, beside `solved_placement` | `tilemega.solved_arch = "sm_89"` from `options.target.arch_tag` |
| codegen | `lib/Codegen/Codegen.cpp` | `TILEMEGA_ARCH_FROM_PLAN`, `TILEMEGA_ARCH_ID 890`, `TILEMEGA_ARCH_TAG "sm_89"`, behind the same "compile option disagrees" guard the other solved parameters use |
| instantiation | `ModelHarness.cuh` | `HarnessArch = arch::ArchFromId<TILEMEGA_ARCH_ID>::type` |
| device pass | `ModelHarness.cuh` | `static_assert(TILEMEGA_ARCH_ID == __CUDA_ARCH__)`, only in a generated source |
| run | `RunModel` | the device's own `major*100+minor*10` against the Plan's, hard failure on disagreement |

`arch::ArchId<Arch>` and `arch::ArchFromId<Id>` are the two directions of one
table in `Target/ArchDispatch.h`; `ArchIdForTag` is the host-side spelling used
by codegen. The identifier is `__CUDA_ARCH__`'s own numbering, so the device
pass compares one integer rather than two spellings.

A source generated before this attribute existed emits none of it, keeps the
header's `TILEMEGA_ARCH_ID 800` default and runs no device check — which is
every pre-generated reference source under `docs/experiments/`, unchanged.

## Verified

✅ **The Plan's architecture reaches the binary.** `generated_macros.txt` is the
macro block of a gqa2 solve at `--seq 4 --past 3`; the run prints
`E2E_ARCH plan=sm_89 plan_id=890 device=sm_89 device_id=890` and
`RESULT status=PASS` with all three levels bit-identical
(`plan_arch_match.log`).

✅ **A disagreement is a hard failure, not a fallback.** The same harness built
`-arch=sm_80 -DTILEMEGA_ARCH_ID=800` and run on this sm_89 device — which CUDA
happily JITs — prints
`E2E_ARCH plan=sm_80 plan_id=800 device=sm_89 device_id=890`, writes
`the Plan was solved for sm_80 and this device is sm_89` to stderr and exits
**2** before any launch (`plan_arch_mismatch.log`). §4.1 asks for exactly this:
no degradation.

⚠️ **Inferred: no behaviour changed on this machine.** `Caps<Sm89>` derives from
`Caps<Sm80>` in `ArchDispatch.h`, so binding the bodies to Sm89 instead of Sm80
selects the same capability set; what changed is that the binding is now the
Plan's statement rather than a header's assumption. The reference cell passes
and ctest is unchanged.

## Per-role traits

`TaskTraits<Threads, SharedBytes>` keeps its meaning and gains `kRoles = 1`.
A warp-specialized body declares `using Roles = TaskRoles<TaskRole<128,4096>,
TaskRole<128,2048>>;` instead, and `TaskRolesOf<Body>` hands every body a role
list — a single role covering the whole CTA when it declares none, which is
what each body meant before R8. The sums are computed from the parts, so a
split that does not partition the CTA is a compile error rather than a launch
that hangs. Nothing declares more than one role yet; BE-5's litmus decides
whether the harness can synchronize them (§8.3).
