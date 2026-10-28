# MX-5 local LDS synchronization checkpoint

Authority: original MX5 prompt SHA256
`75431f98a6286063cc73effe5ece2ed3d462304de4c4192dbd4023efe2bd6f73`.
User amendment: permit already-probed LDS-only primitives for pure local LDS
synchronization when pursuing TPOT, with the difference recorded. Global DT-P
remains P-B. Other original protocols, numerical rules and gates remain.

## Purpose and boundary

Inferred purpose: avoid unnecessarily retiring global prefetch loads when a
CTA only needs to publish/reuse LDS. The former restriction was the strict
implementation rule in section0.2 combined with P-B's synchronization scope;
it was not a hardware prohibition. This amendment does not reclassify DT-P
as P-A or remove global publication boundaries.

- `0b74c686c`: replace attention wave-result LDS publication with compiler
  barriers, `arrive_bsmcnt(0)` and `barrier_inst`, joined by every wave.
- `7c9658ac5`: additionally use that primitive for segment-tail LDS workspace
  reuse. Run-tail stdCTA, StageBarrier and all LA/global release/acquire remain.
- Source-level HB review: next segments use different heads; same-head global
  consumers remain after LA release/acquire or retained Run/grid boundaries.
  In particular, the earlier LA release does not publish later ctx writes.

## Actual validation (verified)

The first candidate passes5fresh×56 D128/Q2 P0 attention cases and the actual
helper litmus5fresh×6 cases(512/576×N0/2/8KiB)×2000iterations×104CTAs,0errors.
The second passes the same pilots/litmus independently. Its full required
attention matrix also passes5 distinct fresh processes×336 cases,0skips, plus
15 MX4 transform bitwise cases per process. Each process first checks actual
336 same-source P0 queries,24 selected unit kernels and6 helper kernels;
selected kernels have API0,frame0,static-shared0 andresident1.

Both candidates pass the manual Qwen0.6B B2 C-1/C2 checks at512 decode steps;
maxgap=.03125 andC2mismatches=0. These do not replace final top4 selections.

| Same-card7 P-PAIR | Old median ms | New median ms | Change | Distinguishable |
|---|---:|---:|---:|---|
| Row2678 → initial LDS publication |2.304253|2.307628|+0.146%|No|
| Initial publication → segment-tail reuse |2.309219|2.302293|−0.300%|No|

Each arm has2 fresh processes,warm1/timed2,ISL64/OSL1024,BF16 greedy.
No TPOT gain or default promotion is claimed. Both comparisons retain raw
rounds and their own old arm; do not compare their absolute medians as one pair.
The original MX4 B2 TPOT remains2.095492ms; the manual new result still exceeds it.

Normal source88327 producer is freshly built160/160 and59delivery files are
rehashed. New A19 isdd8abe75a9d5037649d4b7bfa61f9570a4c41c4d8cff8d3e3a467aa19d7fb3dd.
G/H transitive source bytes remain identical to the archived qualified pool.
Native B2 512 WD0/WD1 each has231MT/frame0 andactualresident1. Its576floor
control has168MT/frame360 andis queried only,never numerically executed.
Stagebench rejection does not authorize blanket whole-program512 fallback.
The independent28-kernel comparison confirms both local sites in MIR; steady
MMA loops have no4160/generic access. Some whole-kernel4160 counts remain.

## Cost lineage and remaining work

Verified old-source Row2678/A19cc122 attention400/400 has3fresh measurements,
with25 scoped512 consumer replays from4545G/H/A keys. These costs were never
blocked by local-LDS permission; the earlier192/400 checkpoint reflected guard75.
New-source400 P0/primary resource queries pass independently. New cold costs
complete400/400,3fresh per key and256MiB active rotation. All25 cost scopes
replay successfully from the new4545-key G/H/A pool. Old A rows are
not relabelled as new A. Neither25 cost scopes nor336 unit passes approve all
whole programs,final solver choices or defaults.

Raw roots: `runs/mx5/local_lds_publication_candidate_v1/`,
`local_lds_reuse_tail_candidate_v2/`, `local_lds_tail_required336_root_t512_v1/`,
`local_lds_tail_costs400_v1/`, `row2678_complete400_qualified25_t512_v3/`.

Verified storage maintenance:40 exact duplicate sealed graph exports now
share physical storage after SHA256 checks; all paths/content remain. About
20.687GiB reclaimed. No measurement,failed attempt or source archive removed.

Dense final25,MoE native5,LLM30 acceptance,andDNN native25 remain unfinished;
original phase order and60card-hour budget remain. The measured local-LDS
comparisons do not establish the hoped-for TPOT improvement.

Verified576 context counterexample (same new A19): Llama1B B1 normal
G/H-floor,attentionPK2 program has167MT/frame0 andactualresident1,API0.
The runtime-shape self-prefetch stagebench has PK0=137–138MT/frame0,
PK2=168MT/frame44–48. Actual queries give PK0fit4/4 andPK2reject4/4,
with0numerical launches. Section4.5 choosesPK2 fromPK0 Rnext, so that
576stage-cost item remains blocked; forcing the whole plan to512 or using
PK0 costs would violate the original rule. No single compiler cause is proved.
Locations: experiments/mx5/attention_stage_bench.cu:53 (self-next context),
DecodeAttentionV2.h:432 (common Run);raw local_lds_tail_576_context_actual_queries_v1.

Verified native solver continuation: new-source25 requests each emit an exact
independent512 top4 preview,100 candidates,0device compiles/0GPU calls. These
are predictions, not final selections or512 exceptions. Five development
requests'20 original canonical576 programs are rebuilt normally from their
actual576 P0 profiles:3 haveframe0 and17 retainframe failures. Guarded
zero-launch queries andprescribed history consumption follow;no missing retry
history or D7 evidence is filled in. The source-specific query log in the
frozen tree is added to the physical-card accounting;all prior logs andthe
60h limit are retained.

Verified first new-source exacttop4 E2E selection: Qwen0.6B B1 allfour
candidates passC1/C2512,two fresh timing processes each. Medians(ms) are
[1.976266,1.977535,1.980171,1.974518]. Anchor0 is retained by the1% rule;
pred/meas1.02291. FixedMX4 is1.983552ms(−.367%,indistinguishable). This
passes cell nonregression without claiming a measurable speedup. Current
source completion subsequently reaches2/25dense,0/5MoE,0/25DNN;defaults unchanged.

Verified second new-source selection: Qwen0.6B B16 allfour C-1/C2512
andtwo fresh timings each pass. Candidate medians are
[3.524260,3.523996,3.528153,3.529580]ms;anchor0 remains selected.
Pred/meas1.00856;fixedMX44.537082ms gives−22.323%,fastestbaseline4.0203ms
givesratio.876616. This is an actualsolver result,not the manual LDS P-PAIR.
Its 4selection/4correctness/3serving/8resource rows append without changing
prior bytes. Raw:local_lds_tail_qwen3_0_6b_B16_top4_execution_v1.

Verified development history closure:20canonical576 originals plus bounded
rebuilds have46actual zero-launch queries;16terminal histories support
independent512,D7 reviewed;4Llama1B outcomes retain576. Remaining20cells'
80canonical576 originals are being built onCPU only;no inherited512 approval.
