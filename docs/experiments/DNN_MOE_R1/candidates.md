# DM-1 GEMM candidate contracts

verified: `CI7_gemm_candidates_host.json` records 7/7 host checks and
check-policy for `DmGemmCandidates`. The original LLM candidate domain is
unchanged. Production skeleton search integration remains pending.

The DM domain includes TN=16 and TK=16/32, filters allocated shared storage
against the target's CTA budget, and includes split-K only when complete
iteration blocks can be partitioned. A short final K block remains legal.
Pipeline candidates expose at most eight cp.async groups. Per-filter channel
padding contributes to iteration counts: C'=24, R=S=3, TK=16 needs 18
iterations, rather than ceil(216/16)=14. Small channels must divide TK.

Gate pairs must fit a complete interleaving unit within TN. Expert gate/up
TN divides 2*Ie and expert TK divides Ie. A deployment using one packed
weight allocation constrains N/K tiles to its physical packing; TM remains
independent. The precise constraint is supplied by the deployment planner.

verified: the first expert-domain fixture omitted the target memory budget
and failed. Its original log is retained. TM16/TN256/TK128/stages2 uses
139264 B and exceeds the fixture's unchanged 98304 B target budget. The
independent oracle now includes that budget; candidate logic was unchanged.
No measured speed, fitted cost or model coverage follows from these checks.
