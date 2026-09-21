# BE-2 — the GEMM collective is selected by capability

## What was already there, and what was not

§1 counts CUTLASS references per file and concludes the GEMM family barely
touches CUTLASS. Reading the code says something more specific:

| body | reachable from the harness? | what it is |
|---|---|---|
| `GemmStageTaskBody` | yes (`T_Gemm`) | builds `backend::GemmCandidate`, which is a CUTLASS `CollectiveMma` with `MainloopSm80CpAsync` and, on the BF16 profile, the `SM80_16x8x16_F32BF16BF16F32_TN` tensor-core atom with **F32 accumulate** |
| `GemmCombineTaskBody` | yes (`T_GemmCombine`) | the split-K reduction, not a GEMM -- treated under BE-4 |
| `FusedGemmTaskBody` | yes, under `TILEMEGA_FUSED_*` | reuses the same candidate |
| `GemmTaskBody` | **no** | a placeholder that writes `context.iteration` into the output; `ModelHarness.cuh` never instantiates it |
| `GemmSplitKTaskBody` | **no** | `: GemmTaskBody<Arch, Stages>` with nothing added |

So the GEMM the anchored models run was already a CUTLASS collective with
tensor cores and an FP32 accumulator. What was missing is the thing BE-2 is
actually about: **the collective was not chosen by the architecture.**
`TypedGemmCandidate` picked by dtype alone, and every instantiation landed on
the sm_80 cp.async multistage schedule no matter what the Plan targeted.

## The CUTLASS that is pinned here

`third_party/cutlass` is at `dc45f979`, which is **CUTLASS 4.8.0**.
`include/cutlass/gemm/collective/builders/` holds specializations for
**sm90, sm100, sm103, sm120** and *no SM80 or SM89 builder at all* — the
sm_80-class collective exists only as `collective/sm80_mma_multistage.hpp`.

§3 anticipated the opposite (an old submodule with only SM80) and asked for an
upgrade in that case. The upgrade does not apply: 4.8.0 is current, and the
gap is on the development machine's own architecture rather than on the new
ones. `CollectiveBuilder<Sm89, ...>` cannot be instantiated by any version of
CUTLASS, so the sm_89 path stays on `CollectiveMma` — which is a CUTLASS
collective, not a hand-written mainloop.

⚠️ **Verified: sm_120's builder refuses BF16.** Probing
`CollectiveBuilder<Sm120, OpClassTensorOp, bfloat16_t, ...>` directly gives

```
sm120_mma_builder.inl(80): error: static assertion failed with
  "SM120 TmaWarpSpecialized builder currently only supports F8F6F4..."
cute/arch/mma_sm120.hpp(47): error: static assertion failed with
  "No MMA matches SM120_16x8x32_TN for given data types."
```

That is §8.3's case exactly: the combination is recorded and falls back to an
available schedule, and BE-2 is not stopped.

## How the choice is made

`arch::Caps<Arch>::kBf16CollectiveBuilder` is a new capability meaning *CUTLASS
ships a BF16 tensor-op CollectiveBuilder for this architecture* — true for
Sm90 and Sm100, false for Sm80, Sm89 and Sm120, each value established by the
probe above rather than assumed. `ArchTensorCollective<Arch, Info>` reads that
one boolean. No code compares an architecture version, and the epilogue is
instantiated before the mainloop so the builder can carve its stage count out
of what the epilogue leaves (§4.2).

The Plan's architecture reaches the file-scope GEMM variants through the same
`TILEMEGA_ARCH_ID` macro BE-1 introduced.

## Verified, all five architectures

`run_arch_check.sh` compiles `arch_check.cu` once per `-arch` against this
repository's headers and runs each binary on the CPU (`arch_check.tsv`):

| arch | builder | path | threads | mainloop smem | priced by the solver |
|---|---|---|---|---|---|
| sm_80 | 0 | cp.async multistage `CollectiveMma` | 128 | 73728 | yes |
| sm_89 | 0 | cp.async multistage `CollectiveMma` | 128 | 73728 | yes |
| sm_90 | 1 | TMA warp-specialized `CollectiveBuilder` | 128 | 221440 | **no** |
| sm_100 | 1 | TMA warp-specialized `CollectiveBuilder` | 128 | 196736 | **no** |
| sm_120 | 0 | cp.async multistage `CollectiveMma` | 128 | 73728 | yes |

✅ **Verified: every architecture instantiates and self-checks on the CPU**
(A-f), including the two with no device on this machine.

⚠️ **Not done, and declared: the builder path is compiled, not priced.** The
solver's `TensorBF16SmemBytes` closed form describes the multistage
collective; the builder's shared storage is 3.0x larger on sm_90 and 2.7x on
sm_100. `TypedGemmCandidate::kPricedCollective` marks which of the two a
candidate is, the compile-time contract is asserted for the priced path, and
the other is reported instead of being asserted away. Enumerating candidates
the cost model cannot cost would be a cost-model change; it belongs with BE-6
and R10, and `porting.md` carries it as a named gap.

## Reproducing

```
bash docs/experiments/BACKEND/be2_collective/run_arch_check.sh
```
