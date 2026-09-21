# Porting the reworked backend to A100 / H100 / B200

What each target needs, which paths this round compiled, and which it did not.
The point of BE-1's capability plumbing is that this table is short.

## What travels with the Plan

A generated source carries `TILEMEGA_ARCH_ID` and `TILEMEGA_ARCH_TAG` from
`tmexec.solved_arch`. The TaskBodies instantiate on
`arch::ArchFromId<TILEMEGA_ARCH_ID>::type`, the device pass asserts the tag
against `__CUDA_ARCH__`, and `RunModel` refuses a device whose compute
capability disagrees — exit 2, before any launch. Nothing below is selected by
comparing a version number; every branch reads one member of
`arch::Caps<Arch>`.

| target | arch tag | `cluster` | `tma` | `warp_specialized` | `tcgen05` | `cp_async` | `kBf16CollectiveBuilder` |
|---|---|---|---|---|---|---|---|
| A100 | `sm_80` | no | no | no | no | yes | no |
| Ada (this machine) | `sm_89` | no | no | no | no | yes | no |
| H100 | `sm_90` | yes | yes | yes | no | yes | **yes** |
| B200 | `sm_100` | yes | yes | yes | yes | yes | **yes** |
| consumer Blackwell | `sm_120` | yes | yes | yes | no | yes | no (see below) |

## A100 — `sm_80`

Nothing to do. `Caps<Sm89>` derives from `Caps<Sm80>`, so A100 and this
machine take the same path: the cp.async multistage `CollectiveMma` with the
`SM80_16x8x16_F32BF16BF16F32_TN` atom and an FP32 accumulator.

✅ Compiled and CPU-checked this round (`be2_collective/arch_check.tsv`).
⚠️ Not run: no A100 here. The resource budgets differ (A100 has 164 KB of
shared memory per SM against Ada's 100 KB), so `TargetSpec::ComputeStages`
will pick a different stage count; that is a JSON target away, not a code
change.

## H100 — `sm_90`

The `CollectiveBuilder` path is selected automatically:
`Caps<Sm90>::kBf16CollectiveBuilder` is true, so `ArchTensorCollective` picks
the TMA warp-specialized collective. At 64x128x64 the builder returns a
mainloop with **9 stages and 221440 bytes** of shared storage.

✅ Compiled and CPU-checked.
❌ **Not priced.** The solver's `TensorBF16SmemBytes` closed form describes the
multistage collective (73728 bytes at the same shape), so a candidate built
from the builder does not satisfy the compile-time contract that lets the host
enumerate candidates without compiling them. `TypedGemmCandidate::kPricedCollective`
marks this, and the assertions are scoped to the priced path. **To run on
H100 the cost model needs a second shared-memory and thread form for the
builder path** — this is the first thing to do, and it is a solver change, not
a backend one.
❌ **Not executed.** The megakernel invokes its mainloop directly; a TMA
warp-specialized collective expects `mbarrier`-based producer/consumer roles.
That needs BE-5's role path, which this round did not deliver — the litmus's
barrier control never failed, so §8.5 was left alone (`BARRIER/README.md`).
On H100 the roles are not optional, so **BE-5 is a hard prerequisite for
H100 execution**, in the order: litmus on an sm_90 part → §8.5 → role-aware
harness barriers → warp-specialized bodies.

## B200 — `sm_100`

Same as H100 plus `tcgen05` and `l1_5`, neither of which any body reads yet.
The builder returns **8 stages and 196736 bytes** at 64x128x64.

✅ Compiled and CPU-checked. ❌ Not priced, not executed — same two reasons.
⚠️ `tcgen05` implies a different accumulator residency (TMEM). Nothing in the
current bodies models that; treat the sm_100 path as "compiles and selects the
right collective", not as a tuned backend.

## Consumer Blackwell — `sm_120`

⚠️ **CUTLASS 4.8's sm_120 builder refuses BF16.**
`CollectiveBuilder<Sm120, OpClassTensorOp, bfloat16_t, ...>` fails with
"SM120 TmaWarpSpecialized builder currently only supports F8F6F4" and
"No MMA matches SM120_16x8x32_TN for given data types". So
`kBf16CollectiveBuilder` is false for sm_120 and it takes the cp.async
multistage path, which compiles and is priced like sm_89.

✅ Compiled and CPU-checked on that path. The `run_sm120.sh` runners carry the
correctness and timing matrix for a real device.

## Summary of what blocks each target

| target | blocked by |
|---|---|
| A100 | nothing; needs a target JSON and a run |
| sm_120 | nothing in the backend; BF16 stays on the multistage collective until CUTLASS ships an sm_120 BF16 builder |
| H100 | (1) cost model must price the builder collective; (2) BE-5's role path, which needs a litmus that can fail its barrier control — an sm_90 part is where that experiment belongs |
| B200 | the two above, plus TMEM-aware accumulator placement if `tcgen05` is to be used at all |
