# A-b — the maximal connected Qwen3 graph (§4.5)

Qwen3-1.7B, public config (`MODELS/sources/qwen_config.json`: 28 layers, hidden
2048, 16 heads over 8 KV heads, head_dim 128, intermediate 6144, vocab 151936),
seeded weights, seq 4, past 3. One command from the export to the rounds; the
solve chose everything below the export.

## Why this graph, and what it costs

The full decoder does not compile. Qwen3's per-head Q/K RMSNorm is the one
operator the task-work derivation refuses:

```
tilemega-compile: local reduction requires an exact unit indexing axis:
  l0.s03.qknorm semantic l0.s03.qknorm operand 0 axis 1 dim r
  index 128*floordiv(c, 128) + 1*r
```

`lib/Analysis/TaskWork.cpp:279-283` requires a reduction axis to be indexed by
exactly one unit term. The per-head normalization reduces over `r` inside a head
while the flattened channel axis carries `128*floordiv(c, 128)` -- an offset that
is invariant in `r` but not a unit term, and 128 is `head_dim`. The check is a
`throw`, so it aborts the whole solve rather than skipping one candidate.

Two ways out were tried this round:

1. **Hoist exactly that operator** (`MODELS2/export_full.py --hoist-qk-norm`):
   Q/K enter the layer already normalized, everything else stays. The graph
   imports (`{"tasks": 2899, "couplings": 3738, "guards": 307}`) and then fails
   in the plan builder -- `tilemega-compile: explicit plan requires stages and
   observable outputs` (`lib/Frontend/Frontend.cpp:819`) -- because
   `DecoderLayerPattern` does not match a layer whose Q and K arrive as graph
   inputs, so `BuildModelPlan` falls through to the covered-region path and
   returns no stages. Reproduced at 1, 2 and 28 layers.
2. **The A-a cut list** (`MODELS2/export_covered_qwen3.py`): cut the token
   embedding, both per-layer RMSNorms, RoPE and the final RMSNorm, exactly as
   `MODELS/export_covered.py` does for Llama -- and because RoPE sits *after*
   the per-head normalization, cutting RoPE cuts that normalization with it. The
   layer module is imported from `MODELS/export_covered.py` rather than
   restated, so the two models are the same code on two configurations. This is
   the graph below.

What remains connected is the residual chain, the V/cache/attention/O chain and
the SwiGLU chain of all 28 layers, plus the vocabulary projection: 114 checked
outputs (2 model outputs and 4 per layer).

## Solve

`solve.json`, `solve.log`. Import `{"tasks": 1096, "couplings": 1233, "guards":
196}`; `SOLVE_SUMMARY evaluated=12 deferred=75 residency_scope=compiled
hop_calibrated=1`; codegen `tasks=562 couplings=504 stages=365`. 2235 s wall
(02:03:04 to 02:40:19; the process was started from a shell, so that window is
file times, not a clock the process kept). Winner
`64x128x16s2split2kappa1r3` with placement `chain`, floor 3.07725e+06 ns,
predicted 3.39889e+06 ns.

## 50 fresh processes

`correctness/r*.log`, `TILEMEGA_WARMUP=0 TILEMEGA_REPEAT=1`, one process each,
one binary.

✅ **Verified: the megakernel reproduces the reference implementation exactly,
50/50.** Every round prints the same line, and all three levels share one hash:

```
E2E_DIFF l05_vs_l0_mismatch=190 max_abs=0.09375 max_rel=10
         l1_vs_l05_mismatch=0 max_abs=0 max_rel=0
         l2_vs_l1_mismatch=0 max_abs=0 max_rel=0
E2E_HASH l05=40f7e79310c4b01c l1=40f7e79310c4b01c l2=40f7e79310c4b01c
```

L1 against L0.5 and L2 against L1 are bit-identical in all 50 rounds -- no
mismatching element, no differing hash, no round-to-round variation.

❌ **Not met: the gate is 50/50 against the CPU golden, and that is 0/50.** 113
of the 114 outputs agree exactly in every round. The one that does not is output
0, the hidden state after 28 residual additions (`buffer=728`): 190 of its 8192
elements land outside `Compare()`'s `1.6e-2 + 1.6e-2*|expected|`, identically in
all 50 rounds.

## What the difference is

`residual_cancellation.py`, on the buffers the harness dumps with
`TILEMEGA_DUMP_BUFFERS` (`residual_cancellation.txt`, `dump_run.log`):

```
ELEMENTS total=8192 outside_tolerance=190 tolerance=0.016+0.016*|expected|
MAGNITUDE all       |expected| median=1.25     p90=3.0625 max=7.53125
MAGNITUDE offending |expected| median=0.399414 p90=1.65625 max=4.4375
```

⚠️ **Inferred: chained bf16 rounding, not a wrong computation.** The worst
absolute differences are 0.09375 on values of 2.5 to 4.4 -- 2.1% to 3.8%
relative, five to ten bf16 ulps (one ulp is ~0.4% relative) -- and the offending
elements are the *smaller* ones (median `|expected|` 0.399 against 1.25 over all
8192), which is what a tolerance with a relative term does to a value that has
cancelled. Both sides are legitimate bf16 evaluations of the same graph with
fp32 accumulation inside each GEMM; they round at 28 layer boundaries in
different orders. The harness tolerance is one relative constant, not a depth
budget, so it prices a 28-deep chain the same as a single layer.

Nothing here moves the tolerance or the expected values: the difference is
recorded (`CLAUDE.md`, evidence rules). The same exporter's Llama graph at 16
layers passes 50/50 (A-a), which is why the depth sweep below asks where the
crossing is rather than assuming it.

## Depth sweep

The same covered export at 4, 8 and 16 layers, exported, solved, built and run
the same way, 5 fresh processes each (`depth.tsv`; the per-depth roots are under
`/root/r7_work/qwen3_cov_l<N>/`). The hidden state is 8192 elements at every
depth, so the counts are directly comparable.

| layers | checked outputs | rounds | passing | elements outside tolerance | `max_abs` | levels identical |
|---|---|---|---|---|---|---|
| 4 | 18 | 5 | 5 | 0 | 0.03125 | yes |
| 8 | 34 | 5 | 0 | 2 | 0.046875 | yes |
| 16 | 66 | 5 | 0 | 44 | 0.0625 | yes |
| 28 | 114 | 50 | 0 | 190 | 0.09375 | yes |

✅ **Verified: the graph passes at depth 4 and the divergence grows with
depth.** Same exporter, same solve flags, same tolerance, same
`MIDPOINT_REFINE=1`: 0, 2, 44 and 190 elements outside tolerance at 4, 8, 16 and
28 layers, with `max_abs` 0.031, 0.047, 0.063, 0.094 -- roughly the square root
of the depth ratio, which is what an accumulation of independent rounding steps
does. Every depth has L0.5, L1 and L2 bit-identical and one distinct `E2E_DIFF`
line over all its rounds.

⚠️ **Inferred, and bounded: depth is not the whole story across models.** A-a's
Llama graph at 16 layers passes 50/50 under this same tolerance formula and the
same refinement, while this model's 16-layer cut has 44 elements outside it.
The two differ in head geometry (16x128 over 8 KV heads against 32x64 over 8),
in the MLP width, and in their fixture draws, so what the sweep establishes is
the trend *within* one model, not a universal depth limit.

## Files

| file | what it is |
|---|---|
| `solve.json`, `solve.log` | the invocation and its output |
| `auto.cu`, `auto.cu.search.tsv`, `auto.cu.top3.tsv`, `auto.cu.bounds.tsv` | the generated module and the search's own tables |
| `build/` | the nvcc command, both hashes, the head |
| `correctness/r*.log`, `r*.json` | the 50 fresh processes |
| `residual_cancellation.txt`, `dump_run.log` | the difference, priced element by element |
| `fixture_manifest.json` | the export's manifest: config hash, dtype, output count |
| `depth.tsv` | mismatch count against depth |
