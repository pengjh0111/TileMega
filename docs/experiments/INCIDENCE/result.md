# Round 5 A1 / A11: physical wait domains and incidence identity

✅ A1 corrects both counts to the same physical bipartite graph. The stored
CG relation is clipped as well: fixing only wait correctly failed the
unchanged CG verifier's independent `wait == Card(C)` check. We did not
disable or weaken that verifier.

Code: `lib/Analysis/CouplingDerivation.cpp:546` counts physical wait/fanout;
`:616` persists the physical relation. CMake option/compile macro
`TILEMEGA_PHYSICAL_WAIT_DOMAIN` defaults ON. OFF is an explicit historical
negative control, never a fallback after an isl failure. `Card` and
`FanoutCard` now have scoped reference guards and null-result errors
(`lib/Analysis/CouplingRelation.cpp:247`). No CostModel arithmetic changed.

## Gates and switch matrix

| Scope | Control OFF | Physical ON |
|---|---|---|
| BF16 gqa2,42 edges×5seq×3past | 124/630 unequal incidence sums | 630/630 equal |
| BF16 mha4,86 edges×5seq×3past | 248/1290 unequal incidence sums | 1290/1290 equal |
| Eight analysis reference graphs×same15 parameter points | Nominal sums recorded per edge | 2505/2505 equal |
| CTest | Negative control deliberately not accepted | 26/26 pass |

seq={1,4,128,512,2048}, past={0,3,512}; reference L_s=seq+past.
The production path substitutes semantic role names and aliases, not only
literal `S`. First gqa2 edge atseq4/past3 changes **512→16**, with fanout16
unchanged. Atseq1 it changes512→4. All production fanout/count/volume values
remain unchanged; 372 wait sums change. The all-row comparison is
`production_comparison.tsv`. `reference.tsv` independently reconstructs the
historical nominal relation from access maps and records every old/new sum.

| Reference graph | Edge/parameter checks | Changed wait sums |
|---|---:|---:|
| llama | 315 | 30 |
| llama4 | 1350 | 120 |
| mha | 540 | 60 |
| mlp | 240 | 0 |
| misaligned | 15 | 15 |
| gather | 15 | 0 |
| affine_gather | 15 | 0 |
| unknown | 15 | 0 |

Permanent tests: `test/unit/incidence_test.cpp:14` and
`tools/tilemega-event-cost.cpp:16`, registered as incidence_identity and
imported_incidence_identity in CTest. The historical-record option emits
all failing rows but **still exits2**, not a falsified PASS.

## Failed attempts and corrected test assumptions

1. Changing wait alone yielded `wait ... does not match ... fiber cardinality`
   because CG still serialized nominal C. Persisting the same clipped C
   repaired the actual inconsistency. That failed import printed remaining=0.
2. An old unit test evaluated wait4096 with S unbound. The physical result
   is4096 only for a full128-row tile, and128 for S=4 with32 heads. The old
   numeric anchor remains tested atS512; a separate short-seq check tests128.
   Attempting to collapse the now-variable function caused barvinok's
   bounded-solution diagnostics during max/min. No erroneous scalar was used.
3. isl retained redundant range inequalities, so string equality of the
   RoPE→append relation and Coarsen identity/composition laws failed. Tests
   now prove both subset directions against the **same original relation**;
   they do not replace it with current output text.
4. KV→attention physical waits vary at both past and S boundaries; their
   countability is correctly piecewise_quasipoly, not constant. The test's
   expected classification changed for that stated mathematical reason.

✅ The real ComputeMetrics error path is exercised by leaving a tile divisor
unbound: rejected=1, before=0, after=0, final ISL_CONTEXT remaining=0.
Reference/production gate logs both end at zero references. This covers A1's
new error exit, not A2/A3 paths that have not yet been implemented.

## A11: limited historical audit

✅ `audit.py` reruns six tilemega-derive modes, four frontend wiring dumps,
table27/semantic-lifting/wiring tests and four SEMANTIC plain/core JSON
codegens. Original reports and GPU logs are preserved; new data live in
`history_audit/`. Both normalized stage tables are unchanged between
plain/core. The **44-edge/440-cell comparison still has exactly4 naming
differences** (`wiring-test.txt`). All14 grouped coupling rows remain
automatically derived, including the omitted residual edge; boundary wait
values no longer mean a universal full nominal tile width. Numeric changes
are enumerated above, not hidden by the unchanged edge count.

OWNERSHIP's41.40% is computed from runtime `TILEMEGA_WAIT_PROFILE` output
(`OWNERSHIP/run.sh:71`), not these stored wait metrics. The labeling/reach
probe's edge weight is `metrics.volume * metrics.count`
(`CLUSTER/cluster_probe.cpp:31`), not wait. A1 leaves those quantities and
topology unchanged, so neither historical experiment is rerun or rebranded
as a new performance claim. CostModel/ChainDP/AlignmentPropagation did not
consume coupling_metrics at baseline; no old solver ranking is attributed
to this wait bug.

⚠️ Coverage is the listed fixtures and parameter grids, not every future
graph or an all-parameter proof. No new GPU synchronization/performance
acceptance follows from these CPU gates. A2 projection and event pricing
remain separate pending gates.

Reproduce ON gates with `cmake -S . -B build-portable
-DTILEMEGA_PHYSICAL_WAIT_DOMAIN=ON`, build, then CTest. OFF control uses the
same tool with `--record-historical-mismatch`. `audit.py` needs both recorded
tables before computing the comparison. Build ends restored to ON.
