# MX-6 synchronization audit

## Evidence and checkpoint scope

Verified CP1 source audit at `bc413139d`; P2/P3/X1/dense remain pending.
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

Pending: P2 deferred-combine five-process litmus and source audit,
P3 attention publication audit, conditional X1's
five-process readiness litmus, and CP6's 576-thread dense qualification.
No G6-SYNC or final reliability acceptance is claimed here.
