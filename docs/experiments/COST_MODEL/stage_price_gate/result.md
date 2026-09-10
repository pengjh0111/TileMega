# A6 complete stage-price bit gate

✅ 4308/4308 groups: all 1077 archived configurations × two models ×
BF16/FP32. Every GEMM stage is checked at seq={1,4,128,512,2048}, past3,
residency={1,2}. Both the direct derived-work entry and the actual cached
`CostModel::TaskStageNs` entry are compared with historical `GemmStageNs`
using double object bits: **904680/904680 each, 1809360 comparisons total**.
All four model/dtype status records pass; explicit isl remaining=0.

The frozen executable SHA256 is
`f5f184e02b84216497c1051bc045c8fb901b45d96dcdc9e25997fee3a5c2be5d`.
`verify_gemm_price.py --directory docs/experiments/COST_MODEL/stage_price_gate
--stage-entry` independently checks configuration-set coverage, stage counts,
point counts, both-entry status records and zero references. Its output and
input SHA256 are in `verification.json`. No tolerance was introduced.

Together with [scalar difference evidence](../scalar_work/result.md),
[ranking controls](../unified_rank/result.md), and [complete solver plans](../unified_solver/result.md),
this closes A6's functional/regression gates. BF16 ranking remains poor on
the historical measurable subset; that negative result is not hidden or
promoted to a complete post-repair GPU oracle. Attention chunk, mixed fusion
and symbolic (a) are still separate tasks, not supplied by this gate.

`TILEMEGA_UNIFIED_TASK_COST=1` now selects unified CG prices by default.
`=0` or explicit `CostModelOptions::unified_task_cost=false` keeps the old
price path. `.cu`-based historical audit, migration and finite-parameter
controls explicitly select the latter; missing CG semantics never trigger
a fallback. The command-line costmodel/solve drivers import CG in the new
mode and retain `--legacy-task-cost` as an independent control.
