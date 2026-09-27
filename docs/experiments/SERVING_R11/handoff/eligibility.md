# Access-proved handoff candidates (B=1 audit)

`tilemega-unit handoff_ir <CG.mlir> audit` creates a temporary `tmexec.handoff`
decision for each CG coupling and each non-event kind, calls the production
`VerifyHandoffAccess` legality checker, then erases that decision. The TSV
files contain every accepted `(kind, coupling, producer, consumer)` tuple;
`eligibility_summary.json` pins the input CG hashes.

| Candidate CG | `recompute` | `last_arriver` | `smem_direct` |
|---|---:|---:|---:|
| Llama decode B1, R10 rank 1 | 35 | 49 | 0 |
| Qwen3 decode B1, R10 rank 1 | 59 | 85 | 0 |

These are legal **candidate edges**, not selected handoffs or a running fused
plan. Qwen3 rank 1 is not the saved R10 winner. Geometry and B can change
legality, so these counts do not stand in for the eventual R11 selected-plan
edge table. The zero `smem_direct` counts support the prompt's synthetic-CG
fallback, but runtime lowering for that fallback is still open.
