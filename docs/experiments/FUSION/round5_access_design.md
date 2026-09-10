# B1.1 exact access composition

The A6 and A9 entry gates passed before this implementation began.
`analysis::ComposeFusionAccesses` is an L-task analysis primitive, not a
completed fusion pass or a claim of GPU fusion.

It takes tensor-keyed physical R/W maps and identifies the internal tensor
edges. `R_c.ApplyRange(W_p.Reverse())` gives consumer-to-producer task C.
Every internal read must be supplied by that relation, and C must cover
the full consumer task domain. A consumer spanning multiple producer tasks
is rejected as an unsatisfied granularity constraint: the solver must first
enumerate a producer tile covering that read footprint. No operator-name
switch guesses an RMSNorm or elementwise tile rule.

The fused reads are `C.ApplyRange(R_p)` plus the remaining consumer reads;
same-tensor overlaps are unioned before counting. Consumer writes remain.
Externally visible producer writes cannot disappear: retention is accepted
only when C covers all producer writers and gives each one a unique fused
owner. Replicated external stores are rejected, not treated as a benign race.
This restriction does not forbid internal fanout greater than one: those
producers are recomputed, with exact symbolic replication count
`sum(fanout) - |image(C)|`. The per-producer fanout polynomial is also retained
for nonuniform producer prices; multiplying an average price is not implied.

I1 is used literally: existing relation composition/union preserve theta and
granularity parameters. No point sampling or second hand-written isl algebra
replaces `CouplingRelation`. The relation is checked against the A1 identity
`sum(wait) == sum(fanout)` before return.

✅ `fusion_access_test`: symbolic tensor-set equality, fanout=2 with exact
replication at S=1/4/128, external retention with fanout=1, and five actual
rejection branches pass; the process reports `ISL_CONTEXT remaining=0`.
Every entry/exception exit is covered by `IslReferenceAudit`.
Code: `include/tilemega/Analysis/FusionAccess.h:17`,
`lib/Analysis/FusionAccess.cpp:8`, `test/unit/fusion_access_test.cpp:9`.

⚠️ Still required: production CG adaptation; the four cost terms (including
max scratch plus cross-boundary tile and global pinned residency); mixed
arithmetic audit; interval DP; L-task writeback/reverification; lowering;
two BF16 GPU calibration edges and their 50-process validations. This
primitive does not discharge those items or the fusion event-price gate.
