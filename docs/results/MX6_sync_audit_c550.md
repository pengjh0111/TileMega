# MX-6 synchronization audit

## Evidence and checkpoint scope

Verified CP1 source audit at `bc413139d`, CP2 at `d4e8b5dae`.
CP2 stops on the prescribed E2E performance condition; P3/X1/dense remain pending.
This records publication sites and measurements, not G6 completion.
Primitive runtime evidence: both `barrier_v3_early_arrive` and
`lds_sync_inflight` pass **5/5 fresh processes**, each 2000 iterations,
104 participating CTAs per iteration, zero violations. Every process queries
resident1, local0, static-smem0; MT56/68 respectively. Each wave issues 8 KiB
from a rotating 256 MiB cold allocation. Raw:
`runs/mx6/cp1/SYNC_PASS.json`, `sync_build_bound/BUILD.json`, `sync_runs/`.
No 576-thread MoE V3 program is generated. The generic 576 role wrapper is
source-only preparation; its dense execution qualification belongs to CP6.

## Stage arrival invariants (verified source and DIAG cohort)

1. `lib/Codegen/StageProgram.cpp` emits `arrive(); return X::Prefetch(...)`
   as each V3 next callback and `wait()` after the body returns. Host execution
   tests cover eighteen loop/tail/terminal cases, two steps each, with exact
   body→arrival→prefetch→wait ordering; V2 emission tests still pass.
2. `MoeProgramV3.cuh:205` selects the current stage's 64-byte counter line.
   StageSyncV3's role fences and adds1 without taking the result. V3 Wait
   performs the existing volatile target poll, acquire, token broadcast and
   LDS generation publication. Routing metadata is absent from that broadcast.
3. `MoeProgramV3.cuh:189` places a volatile LDS arrival count after RouteTable.
   The role initializes it, increments on arrival, traps unless it is1 at
   Wait, and resets it for the next stage. The DIAG model cohort passes
   **20/20 fresh processes per development cell**, 80/80 total, each with
   warmup1+timed2 full1024-token requests and104 CTAs. Raw:
   `runs/mx6/cp1/ledger/COMPLETE.json`, `ledger_collection/COMPLETE.json`.
4. Final publication sites below remain CTA-wide and precede next().
   Intermediate last-arriver protocols are preserved at this checkpoint.

| Stage | Final CTA publication site | Last global output |
|---|---|---|
|qkv / o|`DecodeGemmV2.h:562`|epilogue output or stream-K partial/reduced output|
|attention (control V2)|`DecodeAttentionV2.h:459`|KV/ctx/partial merge output|
|router logits|`RouterLogitsV3.h:498`|BF16 logits and DIAG input/logits|
|gate_up|`GroupedGemmV3.h:530`|expert h; RouteTable built before the core|
|down (CP1 completion)|`GroupedGemmV3.h:530`|partials and completed residual x|
|head|`DecodeHeadV2.h:202`|token after vocabulary reduction|
|DIAG skipped body|`StageProgram.cuh`, RunStageV3|incoming prefetch retired before CTA sync|
|DIAG skipped GU arithmetic|`GroupedGemmV3.h:552`|table/debug outputs; table retained for down|

The role's stage atomic in the bound sync-probe MIR has no result definition;
there is no atomic-result dependency before the outgoing payload loads.
All39 frozen V3 plans' stage U64 atomic sites likewise discard the result;
`runs/mx6/cp1/ARRIVAL_MIR_AUDIT.json` binds the final target machine instruction
streams after stack-frame-layout, including waits. The SDK has no binary
disassembler; this is the final-machine-IR method documented by
`scripts/mx_mir_report.py`, not a claim to have independently decoded bytes.
The40th V3 plan is a byte-identical Qwen B1 rebuild with complete attribution.

## LDS exchanges and prefetch lifetime (verified source)

StageSyncV3::LdsSync is exactly compiler memory clobber →
`arrive_bsmcnt(0)` → `barrier_inst` → compiler memory clobber.
Its cross-wave RAW/WAR publication under inflight loads has the **5/5** process
evidence above. Source body changes only select that primitive for LDS exchange;
the accumulation and reduction loop orders remain the V2 orders.

| Region | Policy |
|---|---|
|GEMM Finish (qkv/o)|eight LDS-exchange sites use Shape::Sync::LdsSync|
|head Round reduction|three exchange sites use Shape::Sync::LdsSync|
|router logits Finish|eight exchange sites use LdsSync; no dispatch/completion LA|
|grouped Finish|eight exchange sites use LdsSync|
|RouteTable construction|one logits load/wait, then LDS-only ranking/count/row exchanges|
|grouped Unit / OutputRow|explicit address-space3 table reads, no global metadata/wait|
|all body final publications|CtaSync, before the next callback|
|existing LA internals|CtaSync/fence/atomic/acquire unchanged|

Round primes the following unit before Finish. Its LDS exchanges do not add a
global wait. CP1 down still enters the prescribed old per-unit release protocol;
CP2 will replace that completion site with deferred combine. Attention remains
the specified V2 control until CP3; its merge chains have not been optimized.
No global metadata table, router tile ticket, router completion ticket or
BroadcastRouting exists in the V3 program/runtime.

## V2 control preservation and remaining gates

Verified V2 policy maps LdsSync to CtaSync and shapes select StageSyncV2.
Post-change Qwen B1 V2 decode/prefill device text is byte-identical to the
frozen same-bridge controls; the DIAG-field migration fix also leaves V3 DIAG0
device text unchanged. Raw: `runs/mx6/cp1/BUILD_MATRIX.json`.
This binary statement does not claim new whole-program race qualification.

Pending: P3 attention publication audit, conditional X1's
five-process readiness litmus, and CP6's 576-thread dense qualification.
No G6-SYNC or final reliability acceptance is claimed here.

## CP2 deferred combine — 已取代 (historical verified evidence)

`GroupedGemmV3.h:504` finishes every round before calling Complete at line509.
The old per-unit completion types/calls are absent from V3. At unchanged
geometry, the CTA's contiguous [begin,end) partitions every expert/tile unit
once; summing its `unit % T == tile` counts across104 CTAs gives n_active.
The global counter result is intentionally consumed here, unlike stage arrival.

| Publication step | CP2 source at d4e8b5dae |
|---|---|
|all partial stores complete before release|`MoeCombineV3.h:69`, CTA-wide sync|
|one wave0 release batch|line71 fence; lane-strided tile counts/atomicAdd at72–79|
|last flags visible CTA-wide|line83 CtaSync; flags are explicit address-space3 LDS|
|acquire before partial reads|wave0 fence87, CtaSync88, Combine89|
|unchanged per-output arithmetic|rank0..7 fmul_rn/fadd_rn and BF16 rounding in Combine|
|counter reuse|wave0 resets each last ticket92; final CtaSync94 precedes next512|

Last flags reuse the finished body's workspace; the persistent RouteTable is
outside it and stays untouched. The following stage prefetch starts only after
Complete returns and the early-arrival callback executes. There is no new
global metadata read or per-unit release. Inter-unit Finish exchanges remain
LdsSync. Qkv/o/head, attention V2 control and stage arrival retain CP1 behavior.

`moe_deferred_combine`: **5/5 fresh processes**, each2000 layers ×104 CTAs,
zero violations,208000 participants/process, host-reference x bitwise equal,
MT100, local/static0, resident1. Seeds are the five distinct process PIDs.
n_active cycles1/8/64/128 with randomized routing permutations and inputs;
n_active1 is a synthetic publication case, not a valid distinct top8 route.
Every output and final zeroed ticket is checked. Raw:
`runs/mx6/cp2/unit_bound/sync/`, `unit_gpu/COMPLETE.json`;
all fifteen P0/P2 process rows: `MX6_sync_checks_c550.csv`.

Forty same-geometry grouped cases pass exact V2/V3 h/partials/x comparison and
repeat V3 comparison (B1/16,E64/128; random,skew,spread; empty experts/count16).
This is one guarded process with repeated launches, not five fresh processes.
The five-process publication claim is the independent litmus above.
All21 CP2 V3 native plans pass final-MIR stage-arrival result-discard checks,
MIR/resource checks and identical-primary/auxiliary device-text attribution.
Two rebuilt V2 plans remain byte-identical to the frozen same-bridge controls.
`runs/mx6/cp2/NATIVE_AUDIT_PASS.json`;23 actual queries pass resident1/local0/static0.

Correctness does not enable CP2: OLMoE B16 P-PAIR regresses11.294080→11.568084 ms,
triggering prompt13.4.3/13.4.5. GPU work stops; two other timing processes are
intentionally SIGTERM-aborted, preserved separately from kernel failures.

## CP2′ P1b and fixed owners (verified source and 5/5 litmus)

P0's implementation is unchanged. RouteTable places per-element exponentials
in body LDS, then LdsSync before the original serial summation. Counts precede
32-bit mask construction, followed by LdsSync before popcounts. Expert/row/
weight bits match V2 in all40 combinations ×360 inputs, for each of both,
weights-only and ordinal-only variants (three fresh guarded fixture processes).

M6's owner16 combine uses H/16 counters and fixed owner=floor(tc×104/Tc).
The inverse owner interval is [ceil(cta×Tc/104),ceil((cta+1)×Tc/104)).
All down rounds precede CtaSync; wave0 fences and emits non-returning atomic
counts. Role0 polls only its tiles after its CTA has finished producing,
acquires, and publishes the persistent LDS generation; every thread awaits it.
Generation target=past×layers+layer+1 avoids racing a shared read/increment;
it increases within the launch and resets only on kernel entry. The16-byte
state follows RouteTable, outside both bodies' workspace, and is included in
host/runtime/manifest LDS bounds. Owner-only counter stores precede final
CtaSync→next(); the next grid barrier separates counter reuse. Each item issues
residual+eight partial loads as one batch and preserves V2's rank/rounding order.

`moe_owner_combine`: **5/5 fresh processes**, each2000 layers ×104 CTAs,
zero violations,208000 participants, host x bitwise, MT101/local0/static0,
actual resident1. All16 combinations of TN16/32/64/128 and active1/8/64/128
occur in each process; n_active1 remains a synthetic publication case.
Raw: `runs/mx6/v2/owner_build/litmus`, `owner_diagnostic/litmus_r1..5`;
appended process rows are in MX6_sync_checks_c550.csv.
All four combine modes pass64 same-geometry V2 comparisons each plus repeats,
including B1/16,E64/128, empty experts and count16. The mode fixtures are four
fresh processes, not five-process protocol evidence for the optional modes.
Native final MIR retains non-returning stage U64 and owner I32 adds; full
per-plan source/binary/MIR/occupancy receipts are retained in runs/mx6/v2.

CP2′ whole-program qualification: seven512-step C-1/C-2 cells pass;
Qwen B1/B2/B4 full1024-token V2/V3 comparisons pass two fresh processes/arm.
Both B1 DIAG mask cohorts pass **20/20 fresh processes**,40/40 total, with
per-CTA per-stage arrival-count traps active throughout warmup and timed runs.
All39 new native plans pass actual resident1/frame0/static0 queries and required
MIR reports. This qualifies CP2′ only; final20-process-per-pin reliability and
P3/X1/dense synchronization qualification remain pending.

## CP3 attention publication (verified; nine fresh fixture processes)

`DecodeAttentionV3.h:374–391` assigns each wave one head and writes wave
(m,l,O) slots before LdsSync. Split1 merges only its head's nonempty waves
inside that CTA; it allocates no global partials/tickets. Final ctx/KV stores
precede CtaSync and the unchanged early-arrival next callback. Projection
loads are issued together with the prefix; the original inverse-RMS arithmetic
runs from registers. No new cross-CTA protocol is introduced in split1.

Split2/4 use the existing LastArriverV2 publication/acquire protocol. All m/l/O
partial loads for a thread are issued as one batch before its sole global wait;
there is no CTA sync inside that merge. The unchanged LA protocol's earlier
five-process litmus remains its protocol evidence. The new attention body is
separately exercised in **9/9 fresh guarded processes**,2184 FP64/reference,
cache-history and repeat-bitwise cases; each profile includes legal split1/2/4.
This is body testing, not a claim of2000-iteration split litmus coverage.
Raw: runs/mx6/cp3/unit_gpu_v2/COMPLETE.json. Existing P0/owner litmus evidence
remains5/5 per protocol; P0 and owner code are unchanged at CP3.

All ten primary MoE configurations and seven split candidates pass512-step
C-1/C-2. B1 diagnostic cohorts exercise the DIAG arrival-count trap throughout
warmup1+timed2 full requests: S1 Qwen20/20, S1 OLMoE20/20, and selected OLMoE
split4 20/20 (60/60 new DIAG processes total). Dense's two smoke tests use the V2
program with attention3; dense early-arrival/576 qualification is deferredCP6.
Final twenty-process-per-pin reliability is still required at CP5.


## CP4 optional combine gate — A3 stop (verified failed qualification)

owner_tile: **5/5 fresh processes**, each2000 iterations ×104 CTAs,
208000 participants,0 mismatching words. ResourcesMT101/local0/static0,
resident1; source/probe/process receipts are in MX6_sync_checks_c550.csv.
unit_owner's original scattered-writer fixture fails its first fresh process
with1403349 mismatching32-bit words (launch/sync both report no error).

The corrected fixture at sync_litmus.cu:503 compacts each expert's row ids as
production Epilogue does. It reproduces WN1 Finish's LDS rendezvous *before*
partial stores (GroupedGemmV3.h:434), then preserves the one-lane fence and
non-returning atomic (MoeCombineV3.h:70–73). No after-store CTA rendezvous is
added. Original arithmetic,2000 iterations,104 CTAs and all16 active/TN cases
remain; per-case mismatch counters are diagnostic only.

Its first corrected fresh process still fails: **0/1 pass**,560020 mismatching
32-bit words,208000 participants; resourcesMT90/local0/static0/resident1.
Only the required synthetic n_active1 case fails: TN64=116597,TN128=443423;
other14 combinations0. active1 is not a valid distinct-top8 model route.
This is failed protocol qualification, not evidence of a normal-route illegal
access. No five-process success is claimed or substituted. Raw:
`runs/mx6/cp4/unit_owner_actual_epilogue_qualification/process1/stdout.log`;
source/compiler/failed-fixture hashes are bound in its SOURCE/BUILD records.

Inferred: one issuer's fence does not cover all writer-wave stores in the
synthetic multi-iteration publication case. A draft post-store rendezvous
would strengthen WN1's protocol; its unexecuted CPU build is archived and is
not used as passing evidence. Production code remains unchanged. Per v2
§13.4 A3, work stops2026-10-31 18:33:43 Asia/Shanghai; no GPU job follows.
G6-SYNC is not passed; top-four selection, X1 and CP5/CP6 remain unrun.
