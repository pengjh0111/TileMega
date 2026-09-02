# Second model: structurally different from V-H's two-layer GQA (acceptance B)

Evidence labels: ✅ executed/observed; ⚠️ scope limitation; ❌ conjecture.

## What this proves

Acceptance B requires "at least two structurally different models end to end,
... and the generator produces correct code without throwing," and states
explicitly that this cannot be satisfied by `#include`-ing a handwritten file.
This experiment is the second model. It differs from `docs/experiments/V_H`'s
export in two independent ways at once:

- **Layer count**: 4 layers instead of 2.
- **Attention structure**: `kv_heads == heads` (plain multi-head attention,
  no grouped-query attention), instead of V-H's 4 query heads / 2 KV heads.

`docs/experiments/P3_GENERALIZATION/export_second.py` builds this model from
the same `docs/experiments/V_H/export_probe.py` layer implementation (so the
math is the same Llama-family decoder layer, only its shape parameters and
head-grouping differ) and exports it through `torch.export` exactly the way
V-H does, with the same dynamic `seq`/`past` dims.

## Pipeline

Full frontend-to-binary path, no shortcuts:

```
export_second.py → torch.export → export_bridge.py → tilemega-import
  → tilemega-opt (verify) → tilemega-compile → nvcc → run against its own
  PyTorch reference fixture
```

`tilemega-compile` reads the `tilemega.model_plan` attribute that
`lib/Frontend/ModelPlan.cpp` built structurally from this export's own FX
graph and parameter shapes (layer count, `hidden`/`intermediate` width,
`heads`/`kv_heads`, `head_dim`) — none of these are read from a config
written for the first model.

## Results

```
{"layers": 4, "attention": "MHA", "outputs": 9}
{"tasks": 355, "couplings": 444, "guards": 11}
IMPORT_SUMMARY tasks=355 couplings=444 stages=60 guards=11
CODEGEN_SUMMARY tasks=355 couplings=444 stages=60 output=.../generated.cu
E2E_RESOURCE block=256 reg=ptxas smem=49536 ctas_per_sm=1 num_sms=128 grid=128
E2E_TIME l05_ms=2.181728 l1_ms=2.160576 ratio=0.990305 l2_ms=2.928160 l2_over_l1=1.355268
E2E_DIFF l05_vs_l0_mismatch=0 max_abs=1.9073486e-06 max_rel=0.0097467825 l1_vs_l05_mismatch=0 max_abs=0 max_rel=0 l2_vs_l1_mismatch=0 max_abs=0 max_rel=0
E2E_HASH l05=fd15fa2e89cdb915 l1=fd15fa2e89cdb915 l2=fd15fa2e89cdb915
RESULT status=PASS
```

- ✅ Generated L0.5 matches its own PyTorch L0 reference: 0 mismatches, max
  abs error `1.9073486e-6`, well under the `3e-5` tolerance.
- ✅ Generated L1 is bitwise identical to generated L0.5, and generated L2 is
  bitwise identical to generated L1 (hash `fd15fa2e89cdb915` for all three).
- ✅ No exception was thrown anywhere in the pipeline (import, opt-verify,
  codegen, compile, run all completed cleanly).
- ✅ `raw/forbidden.txt` is empty: `grep -nE '% 12|TILEMEGA_GENERATED_TASK_COUNT
  179|TILEMEGA_GENERATED_COUPLING_COUNT 222|GeneratedLlamaRuntime'` against
  `raw/generated.cu` finds nothing. The 179/222 figures and the `% 12` stage
  arithmetic were V-H's numbers, specific to the first model; this model's own
  355/444/60 numbers reach the binary only as generated table *data*
  (`kDims`, `kBuffers[]`, `kGemms[]`, `kStages[]`), never as `#define`d
  control-flow constants baked into `ModelHarness.cuh`.
- ⚠️ This is a single acceptance run, not a 50-fresh-process statistical
  claim — that bar is reserved for the primary model in
  `docs/experiments/E2E_GEN/` per the task's stated priority (acceptance B
  over acceptance A's performance/statistics work when time is short). The
  synchronization primitives exercised here (global barrier for L1, per-edge
  `kAll` events for L2) are identical code paths to the ones the first model
  already validated 50/50; this run is evidence the *generator*, not the
  *runtime primitives*, generalizes.

## What this does and does not demonstrate

✅ Demonstrates: `ModelPlan` derives layer count, hidden/intermediate width,
and the GQA/MHA head ratio structurally from parameter shapes rather than
from a fixed rule; `CouplingGraphToCUDA` and `ModelHarness.cuh` contain no
per-model control flow; two models with different layer counts and different
attention structure both compile and run correctly through one generator and
one runtime.

⚠️ Does not demonstrate: generality over arbitrary ATen graphs. `ModelPlan`'s
pattern match still requires the Llama decoder-layer dataflow shape
(RMSNorm → QKV → RoPE → KVAppend → Attention → O-proj → residual → RMSNorm →
gated MLP → residual) and `layers.N.*` parameter naming; a structurally
different family (e.g. a pure MLP stack, a different normalization) would
need a new rule added to `lib/Frontend/ModelPlan.cpp`. This is recorded as
residual debt in `TileMega_skeleton.md` §1.5.1, not hidden behind the passing
acceptance run.
