# Part 4 — production width and depth

## c8be09e continuation (supersedes pending items, preserves old data)

✅ [Controlled depth/noise experiment](depth_result.md): 300 fresh processes,
50 at each of six fixed-input/weight-prefix depths. Fixed-criterion passes
are 50/50 at2/4/6 and 0/50 at8/12/16. k_L2=.9913284393 at16 supports the
observed BF16 noise-floor attribution, not criterion acceptance. CPU golden
thread count alone explains the historical198 versus controlled162
mismatches; TileMega's hash did not change.

✅ [Original condition9 rerun](condition9_result.md): FP32 partials do not
fully repair the original 4×4096 leader. split8 fails with one mismatched
element, max_abs=.046875. Stopped at0/1; split16 compiled but not run.
Condition9 remains open. Do not interpret the historical descriptions of
old split failures below as the new FP32-partial process counts.

✅ Exact component-wise isl endpoint enumeration avoids a >1089s unfinished
wide split8 codegen and finishes that same input in17.1849s. The16-layer
control is byte-identical but 321.8086→322.9925s, **no observed benefit**;
[evidence and switch](../ISL_LIFETIME/enumeration_result.md). No GPU timing
claim follows from these individual compiler runs.

Reproduce with `run.sh`. The default is the Llama-3.2-1B decoder shape; the
width control is:

```sh
LAYERS=4 HIDDEN=4096 INTERMEDIATE=14336 HEADS=32 KV_HEADS=8 \
  bash docs/experiments/REALMODEL/run.sh
```

`VARIANT_PLAN` selects an exact checked-in candidate. For example, the valid
real-width candidate is reproduced with:

```sh
LAYERS=4 HIDDEN=4096 INTERMEDIATE=14336 HEADS=32 KV_HEADS=8 \
  VARIANT_PLAN=docs/experiments/REALMODEL/plan_nosplit.json \
  LABEL=l4h4096_nosplit RUNS=50 \
  bash docs/experiments/REALMODEL/run.sh
```

`plan_split8.json` and `plan_split16.json` retain the two rejected model-ranked
arms. Their failures are deterministic for this fixture; `RUNS=1` reproduces
the recorded diagnostic without presenting it as a 50-process experiment.

Evidence status: ✅ generated, compiled, and measured on an RTX 4090 (sm_89),
BF16, 2026-09-06. A failed acceptance is kept as a failed acceptance; the BF16
comparison bound was not widened.

## What matched without a new pattern

The existing declarative decoder-layer `GraphPattern` matched both production
shapes directly. No slot, target-name branch, or TaskBody was added. The bridge
and CG scale exactly by layer: one layer contributes 90 FX tasks after the
first, 17 CG tasks, 15 stages, and 22 inter-layer-adjusted couplings. The full
16-layer graph is 1443 FX tasks / 1805 couplings and lowers to 272 CG tasks /
350 couplings / 240 stages.

## Compiler and table scaling

`raw/scale.tsv` and `raw/size.tsv` are the measurements:

| shape | parameters | CG | codegen | nvcc | generated CUDA | dependency rows |
|---|---:|---:|---:|---:|---:|---:|
| 16 x 2048 x 8192 | 973,144,576 | 272 tasks / 240 stages | 1353 s | 26 s | 97,516 B | 318 |
| 4 x 4096 x 14336 | 872,448,256 | 68 tasks / 60 stages | 252 s | 23 s | 25,573 B | 78 |

At hidden=2048, generated source and stage tables are linear through 16 layers.
Codegen is also approximately linear (71, 135, 267, 499, 1353 seconds for
1/2/4/8/16 layers), but its constant is unacceptable for Phase 5: the isolated
one-layer measurement in `raw/codegen_isolation.tsv` is 3 seconds without tile
ownership and 63 seconds with it. Re-fitting integer wait windows at three
sequence probes, not FX import or CUDA emission, is the dominant cost.

### Symbolic window derivation

✅ That three-point re-fit is no longer the default.  The importer now obtains
the lexicographic minimum and maximum of the isl coupling, linearizes the task
coordinates, and proves both subset directions between the resulting symbolic
interval and the original relation.  Only an edge whose form cannot be proved
falls back to the old three-point path; fallback is per edge, not per model.

| depth | CG tasks | couplings | symbolic | fallback | codegen | generated CUDA |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17 | 20 | 20 | **0** | **0.861 s** | — |
| 16 | 272 | 350 | 350 | **0** | **362.035 s** | 102,192 B |
| 32 | 544 | 702 | 702 | **0** | **10,872.781 s** | 203,632 B |

The one-layer ownership case falls from 63 s to 0.861 s (**73.2x**).  The two
reference models use 42/42 and 86/86 symbolic windows with zero fallback, and
their generated CUDA is byte-for-byte identical to the pre-change source.
The 16-layer source is also the source compiled for the task-queue real-model
check.  Raw values are in `symbolic_codegen.tsv`.

⚠️ The 32-layer total is not a success hidden by the one-layer result: although
all 702 windows are symbolic, end-to-end codegen still takes 3 h 1 min.  The
remaining superlinear cost is whole-graph coupling construction/proof and isl
object lifetime, not window re-fitting.  The required 32-layer number is
therefore reported as measured, not extrapolated, and is a new scaling debt.

Both shapes compile to a 24,576-byte shared-memory union, 212 registers for the
persistent kernels, two resident CTAs/SM, and a 256-CTA grid. Depth grows the
host/device tables, but it does not grow the shared-memory union because stages
reuse it.

## Correctness: depth fails before width

⚠️ T0–T4 update (baseline e305a9f): the results below are historical, not a
rerun with FP32 split partials. The new reference-model split sweep passes
500/500 but does not establish acceptance of either 973M or the 4×4096
cost-optimal configurations. The golden is computed by the exported PyTorch
CPU BF16 program, with explicit FP32 normalization/softmax and BF16 graph
materialization boundaries; its GEMM reduction tree is not specified in the
export. See `../BF16/result.md`, T2.b, for the three evaluated routes and
their unimplemented measurements. No threshold or golden has been replaced.

The requested 16-layer 1B shape runs, and all three TileMega paths are
bit-identical to each other in every process. It nevertheless fails the fixed
PyTorch comparison in **0/50** processes:

| shape | L0.5/L1/L2 agreement | PyTorch acceptance | max absolute error |
|---|---|---:|---:|
| 4 x 2048 | bit-identical | 1/1 | 0.03125 |
| 8 x 2048 | bit-identical | 0/1 (3 elements) | 0.0546875 |
| 16 x 2048 | bit-identical | **0/50 (198 elements)** | 0.078125 |
| 4 x 4096 | bit-identical | **50/50** | 0.046875 |

The failure begins with depth, not width. Per-layer K/V outputs remain inside
the bound until later layers; the final hidden accounts for 190 of the 198
16-layer mismatches. This is accumulated BF16 association error between the
TaskBodies and PyTorch, not an L1/L2 synchronization defect: L0.5, L1 and L2
have the same hash `621738651f623f5b` in all 50 failed processes. The acceptance
bound remains `0.016 + 0.016 * abs(reference)`.

Part 4.3 therefore applies exactly as written. The full 1B depth is recorded as
a correctness failure, and the verified fallback is real width with fewer
layers: hidden=4096 / intermediate=14336 / head_dim=128 at four layers passes
50/50 with one hash (`451f7525006a532e`). Its medians are L0.5 5.843968 ms,
L1 5.802976 ms and L2 6.132224 ms.

## Cost-model transfer

✅ `raw/cost_subset.tsv` is a small-subset transfer check, not a new 1077-point
oracle. The BF16 model was evaluated with the generated production tables and
their live footprints (1.865 GiB for 16x2048 and 1.668 GiB for 4x4096). These
are **uniform** configurations from the existing candidate set; a per-operator
DP plan was not compiled and is not claimed here.

| shape | candidate | predicted L1 | measured L1 | fixed-bound result |
|---|---|---:|---:|---:|
| 16x2048 | control `128x128x16s3k1` | 11.9872 ms | 12.5237 ms | 0/50 (depth) |
| 16x2048 | model top `32x128x32s3k8` | 2.10657 ms | 4.31104 ms | 0/1 (1391 elements) |
| 4x4096 | control `128x128x16s3k1` | 5.46625 ms | 5.80422 ms | 50/50 |
| 4x4096 | model top `32x256x16s3k16` | 1.33868 ms | 2.95142 ms | 0/1 (24 elements) |
| 4x4096 | second `32x128x32s3k8` | 1.37867 ms | 2.63904 ms | 0/1 (26 elements) |
| 4x4096 | best split=1 `32x32x16s2k1` | 2.13917 ms | 4.17382 ms | **50/50** |

❌ The unconstrained winner is not admissible: both leading real-width
split-K candidates violate the unchanged BF16 comparison bound, while L0.5,
L1 and L2 remain bit-identical. On the 16-layer graph, the model winner makes
the already-failing depth error substantially worse. A Phase-5 optimizer must
therefore gate candidates by a numerical-feasibility contract; predicted cost
alone cannot select split-K.

✅ The fastest candidate in the tested, correctness-preserving subset is the
split=1 configuration. It passes 50/50 fresh processes with the same hash as
the control. In 25 interleaved paired rounds it changes L1 by **-28.05%**
(bootstrap 95% CI [-28.15%, -28.03%], Wilcoxon p=1.31e-5) and L2 by
**-34.41%** ([-34.52%, -34.30%], p=1.31e-5). The raw rounds and deterministic
statistics are in `raw/cost_paired.tsv` and `raw/cost_paired_stats.tsv`.
Regenerate the latter with `python3 docs/experiments/REALMODEL/summarize_cost.py`.

⚠️ This does not satisfy "DP-selected configuration versus measured optimum":
only four uniform points were compiled, and the two predicted leaders are
numerically invalid. It establishes transfer failure and one valid improvement,
not the rank of the optimum over the production-size search space.
# T1–T5 continuation (c8be09e)

⚠️ No new depth/noise-floor/4×4096 GPU run has completed in this continuation.
The historical 16×2048 mismatch below must not be relabeled a verified BF16
noise-floor result. Fixed weights/input across depth, PyTorch BF16 and
TileMega BF16 against the same FP32 golden, and the original best split=8/16
4×4096 scene remain required. Phase 5 conditions 7 and 9 remain open;
the numerical tolerance is unchanged. Historical measurements follow.

---
