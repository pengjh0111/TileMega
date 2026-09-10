# Attention candidate DP

Verified: two production models x seq4/128 x four supplied compiled chunk
plans. All 16 feasible alternatives have the same returned CostBreakdown
bits as direct evaluation, and the four minima are exact. Each selects
chunk1 at residency2, matching the frozen GPU winner. `gqa2_candidates.tsv`
and `mha4_candidates.tsv` retain the actual compiled resource inputs.
All logs end with `remaining=0`.

Implementation: `lib/Solver/ChainDP.cpp:28`, `SolveAttentionPlans`.
This tests a supplied four-plan domain under L1 pricing, not the complete
per-layer product and not L2/fusion/placement joint optimization. See
`../attention_prices_batch/result.md` for full scope, rankings and residuals.
