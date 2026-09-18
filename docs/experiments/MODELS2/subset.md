# MODELS2 — numerical admission of the maximal connected Llama graph (R7 A3, A-e)

## What is held fixed

The graph, its export, the fixture, the seed, the residual edges, the 66 checked
outputs and the 0.0231875014 tolerance are R6's, byte for byte: every arm below
compiles `MODELS/covered_llama_admitted2/auto.cu` (the solver-selected geometry
whose admission failed) against the same fixture `MODELS/covered_llama/fixture`.
Nothing here re-exports, re-solves, re-seeds or re-tolerances. The only moving
part is the set of `-DTILEMEGA_*` switches.

## A-e — separating the three changes

`ablation/admitted2/ablation.tsv`, one build and one fresh process per arm.

| arm | switches | result | `V[0,463]` | failing outputs |
|---|---|---|---|---|
| base | none | MISMATCH | -0.44921875 | 1 / 66 |
| a1 | `NORM_EPSILON=1e-5f` | MISMATCH | -0.44921875 | 1 / 66 |
| a2 | `ROPE_FP32_PHASE=1` | MISMATCH | -0.44921875 | 1 / 66 |
| a1a2 | both | MISMATCH | -0.44921875 | 1 / 66 |
| refine | both + `MIDPOINT_REFINE=1` | PASS | -0.451171875 | 0 / 66 |

✅ **Verified: A1 and A2 have no effect on this graph, separately or together.**
This is a structural fact, not a measurement accident. `MODELS/export_covered.py`
cuts the token embedding, both per-layer RMSNorms, RoPE and the final RMSNorm out
of the covered region -- their outputs enter as graph inputs -- so the generated
source contains no `kRMSNorm` and no `kRoPE` stage at all (113 `kGemm`, 32
`kAdd`, 32 `kKVAppend`, 16 `kAttention`, 16 `kElementwise`). An epsilon that no
stage reads and a rotation no stage performs cannot move any output. The three
arms differ only in their binary hash, from the macro definition itself.

R7 §1(三) states that the two precision defects explain the A1-subset failure.
For this graph that cannot hold, and the ablation is the evidence. Both defects
are real and both are fixed (R7 steps 1 and 2); they are simply not the cause of
F-203. Their effect becomes measurable only once A4/A5/A6 put those operators
inside the graph.

## What the failure actually was

`V[0,463]` of the first layer is `v_proj` applied to a graph *input*: one
2048-term dot product in FP32, with no normalization and no rotation upstream.
F-203 localized it correctly; what it did not say is that this leaves accumulation
as the only candidate. The FP64 accumulator is -0.450195362966042 and the BF16
rounding midpoint between -0.44921875 and -0.451171875 is -0.4501953125. The
distance is 5.05e-8, about 1.69 FP32 ulp at that magnitude, against a realistic
FP32 K-loop error some 30x larger. No association of a 2048-term FP32 sum decides
that rounding reliably; the GEMM was a coin flip on one element.

`MIDPOINT_REFINE` recomputes exactly those elements in FP64 -- a BF16xBF16 product
is exact in FP64, so only the FP64 summation error remains, around 2^-40 of the
FP32 one. The guard is relative: `|(|sum-rounded|) - ulp/2| < 0.015625 * ulp`,
which at this magnitude is 3.05e-5, comfortably covering the 5.05e-8 gap. After
it, the whole first-layer V matches its FP64 rounding in every element
(`v_rows_differing_from_fp64 = 0`), not merely at column 463.

## A-a — 50 fresh processes

`admission/admitted2/correctness/r{0..49}.{log,json}`: 50/50 `RESULT status=PASS`,
one binary (`fabb59f368bf...`), 3300 `E2E_OUTPUT_DIFF` lines and not one with a
non-zero mismatch, `l05_vs_l0`, `l1_vs_l05` and `l2_vs_l1` all zero. The
tolerance, output set, seed and residual edges are unchanged from the run that
failed.

## Reproducing

```
python3 docs/experiments/MODELS2/ablate.py --geometry admitted2 --arms base,a1,a2,a1a2,refine
python3 docs/experiments/MODELS2/run_admission.py --geometry admitted2 --arm refine --rounds 50
```

Each arm's `run.json` records the build command, both hashes, the head commit and
the recomputed element. `dump/` keeps only the three buffers the recomputation
reads (the token rows, the V weight and the GPU V), not the full 2.4 GB snapshot.
