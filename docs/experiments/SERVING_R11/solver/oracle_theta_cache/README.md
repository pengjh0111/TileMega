# Exact theta-bound release cache

The `General` Oracle now binds `(B, past)` once per relation and reuses the
resulting exact ISL set for the many consumer-coordinate release queries.
It retains the full Presburger fiber, including holes; the cache is invalidated
whenever either theta parameter changes. `symbolic_oracle` includes a two-
parameter 2/5 → 2/8 → 1/8 → 2/5 rebinding test. The isolated build passed
`symbolic_oracle`, `stage_flow`, and `flow_runtime_release` (3/3). The
baseline `flow_runtime_release` took 87.01 s and the isolated change took
78.15 s while a separate Qwen3 search was active, so this is only an
indicative timing comparison.

The attached flow-audit files use the same Qwen3 s1 selected CG, target,
fixture, residency 1 and κ=1. The old prebuilt binary was from `61d19ec59`;
the new isolated binary includes commit `533cc56ea` (cherry-picked as
`11b1fc347`). Both returned exactly 5,627,316.8694249196 ns for Level 1,
the same 3,506,496.2188630085 ns floor, and the same release/makespan
exactness checks. The isolated `STRUCTURE` phase was 218.291 → 204.134 ms,
while total wall time was 33.109 → 33.132 s. CPU contention means these
figures cannot establish a whole-plan speedup. In particular, this edit does
not close the Qwen3 600 s plan-time gate; the dominant relation and task
pricing phases remain.
