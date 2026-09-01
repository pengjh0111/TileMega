# V-I — four-target cross compilation

Evidence labels: ✅ compiled locally; ⚠️ documented but not run; ❌ inference.

## Matrix

The same TaskBody contract, megakernel skeleton, event/fence path, and target
dispatch were compiled with CUDA 12.8 and the checked-in CUTLASS `main`.

| Target | Stages | TaskBody SHM | Collective selected | REG | Main | Cluster/DSMEM path |
|---|---:|---:|---|---:|---|---|
| sm_80 | 8 | 131,072 B | cp.async multistage | 10 | ✅ PASS | n/a |
| sm_89 | 5 | 81,920 B | cp.async multistage | 10 | ✅ PASS | n/a |
| sm_90 | 8 | 131,072 B | TMA warp-specialized | 10 | ✅ PASS | ✅ PASS, 10 REG |
| sm_120 | 5 | 81,920 B | TMA warp-specialized, SM120 MMA | 12 | ✅ PASS | ✅ PASS, 10 REG |

The stage count is computed from each JSON target as
`min(8, floor((max_dynamic_smem_per_cta - 4096) / 16384))`. The approximately
two-times target budget difference changes the same TaskBody from 5 to 8
stages, proving that `Stages` must be a `TargetSpec` input.

## Answers

1. ✅ All four main paths compiled; the matrix above includes ptxas registers,
   static TaskBody storage, and the selected collective family.
2. ✅ The same logical body needs 131,072 / 81,920 / 131,072 / 81,920 bytes for
   sm_80 / sm_89 / sm_90 / sm_120 under the target-driven stage policy.
3. ✅ `Caps<Sm120>::kTcgen05` is false. The sm_120 cubin selected the legacy
   SM120 MMA/TMA path and compiled; no rejected `tcgen05` instruction was used.
4. ✅ No business or TaskBody code needed a target-name/version branch.
   Capability definitions and compile-time selection remain confined to
   `ArchDispatch.h`; JSON supplies runtime resource budgets.

⚠️ Cross compilation proves toolchain expression, not runtime correctness on
A100, H100, or RTX 5090. Rerun all experiments after migration.

## Reproduction

```bash
cmake -S . -B build -G Ninja
ninja -C build crosscompile-matrix
cat docs/experiments/V_I/raw/matrix.tsv
```
