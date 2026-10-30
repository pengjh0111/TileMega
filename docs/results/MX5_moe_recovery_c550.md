# MX-5 C550 MoE recovery

Prompt SHA256: `75431f98a6286063cc73effe5ece2ed3d462304de4c4192dbd4023efe2bd6f73`.
Starting HEAD: `7fb2ddc1f`. This is an ongoing recovery report, not acceptance.

## torch.compile missing baseline cells

Verified by reading all eight original MX-4 logs and guard records: MoE-L24
B2/B4/B8/B16, rounds 1 and 2, all terminate with code 124, `guard timeout`.
Every log contains the Dynamo recompile-limit warning and specialization on
`self_attn.layer_idx`; none contains an out-of-memory exception. The logs also
record failure of fullgraph=True and retry with fullgraph=False. Recompilation
is an observed contributor; the logs do not isolate the entire timeout cost.

| B | Round 1 peak used MiB | Round 2 peak used MiB | Disposition |
|---|---:|---:|---|
| 2 | 31710.09 | 31693.64 | timeout in both rounds |
| 4 | 32043.75 | 31883.54 | timeout in both rounds |
| 8 | 32627.05 | 32843.65 | timeout in both rounds |
| 16 | 34346.00 | 34345.55 | timeout in both rounds |

These are sampled device totals on 65536 MiB cards, not allocator peaks.
Even the smallest sampled free-memory value is 31189.996 MiB. Record these
cells as timeout, not OOM or zero latency. Original fixed baseline CSVs are
unchanged; no external baseline was rerun. Raw CPU audit:
`runs/mx5/moe_recovery_20261030_v1/BASELINE_TIMEOUT_AUDIT_CPU.json`.

Verified additional B1 metadata from both completed MX-4 rounds: grouped_mm
raises `RuntimeError: grouped mm is not supported on your system`; the old
baseline explicitly runs eager experts instead. It records fullgraph=False
and compiled_call_used=True. Its retained fullgraph traceback points to
`for expert_idx in expert_hit` in the HF expert implementation. Thus the
completed B1 result is compiled decode with graph breaks and eager experts,
not a successful grouped_mm/fullgraph baseline. The timed-out B>=2 logs do
not preserve the full exception metadata; no unobserved error is invented.

## TileMega B4/B16 resource block

Verified previous full-program original top-four builds reach MTreg 256 with
positive stack frames: B4 256–288 B, B16 320–368 B. Their prescribed prefetch
reductions to D still fail. This is register spilling, not a device-memory OOM.
The original thresholds remain MTreg <=255 at 512 threads and zero stack.
Previous stage-omission diagnostics implicate expert down/combine integration;
they do not establish that every spill instruction belongs to that body.

The user now prioritizes making B4/B16 executable and testing them, followed
by prompt-directed MoE optimization toward about 1.1x speedup. Supplementary
recovery explores the existing measured legal domain in native k-best order;
original failed candidates and their resource histories remain recorded.
An extended diagnostic preview does not grant execution, select defaults,
change resource/numerical thresholds, or replace the final four-candidate
end-to-end comparison. Any supplementary ranks must be identified as a
selection difference from the prompt's original predicted top-four set.

## Executable development controls (verified)

The original failed programs remain unchanged. B4 changes expert down from
TN128/WN2/D3 to D2 or D1 using measured rows from the same legal domain.
B16 changes down from TN128/WN8/D3 to TN128/WN2/D1. For B16, the remaining
down/head prefetch widths are halved to D, in the prescribed largest-width
order. These controls retain all 24 layers and all six stages per layer.
They are manual development controls, not replacement solver selections.

| Control | Decode WD0 MTreg | Prompt WD0 MTreg | Decode WD1 MTreg | Prompt WD1 MTreg |
|---|---:|---:|---:|---:|
| B4 down D2 | 245 | 249 | 249 | 253 |
| B4 down D1 | 235 | 239 | 239 | 243 |
| B16 down D1, down/head Pw1 | 241 | 245 | 249 | 253 |

Every artifact in this table has zero stack/static shared memory, passes
the MIR checks, and has an actual zero-launch query reporting one resident
CTA and localSizeBytes=0. Their own canonical 576-thread prompt/decode
histories still spill after widths reach D. Previous standalone attention
body-pressure evidence supports the per-plan 512-thread branch; this is not
a blanket waiver of the register limit. All three controls pass 512-step
C-1 (authorized eager HF reference) and bitwise C-2.

Two local orchestration failures are preserved: a missing initial decode
query-spec path stopped B4 after ten zero-launch canonical queries; the
corrected fresh query campaign completed. B16's first controls were rejected
before any model launch because WD1 prompt still spilled; reducing the
remaining head width to D produced the complete fitting pair above.

52 resource rows are appended to `MX5_resources_c550.csv`, retaining its
old byte prefix. They include fitting/rejected recovery artifacts and the
separate unroll4 experiment; resource fit is not model acceptance.
Raw: `runs/mx5/moe_recovery_20261030_v1/`.

Verified recovery timing completes two fresh processes per control, each
warmup1/timed2, ISL64/OSL1024, greedy BF16, L1 prompt and decode, with no
trace/DIAG/watchdog/events in timed binaries. All nine per-round/median rows
are appended to `MX5_serving_c550.csv` as recovery development, preserving
its old prefix and leaving selected_origin_candidate_id blank.

| Control | Round 1 TPOT ms | Round 2 TPOT ms | Median ms | SGLang / TileMega |
|---|---:|---:|---:|---:|
| B4 down D2 | 7.921309 | 7.923681 | 7.922495 | 0.9470x |
| B4 down D1 | 7.961001 | 7.960367 | 7.960684 | 0.9425x |
| B16 down D1, down/head Pw1 | 17.500871 | 17.505814 | 17.503343 | 0.8426x |

The B4 controls differ by less than1%, so this is not a distinguishable
performance win between them. Both missing batches now have tested full
native prompt/decode pairs, but neither is final solver acceptance.
Explicit `*_recovery_development.json` pins retain their measured source,
binary identity and open gates; they are not installed as default pins.
For a 1.1x speedup over fixed SGLang, B4 would require <=6.820727ms and
B16 <=13.408273ms. The recovery measurements do not meet those targets.

## Router experiments (development evidence; final gates remain open)

The rank loop groups four comparisons under each loop iteration, preserving
all 128 comparisons, expert-index tie breaking, weights and dispatch order.
No synchronization or arithmetic contract changes. An isolated fixture
passes all 100 legacy-bitwise router/combine cases and 36 zero-launch
resource queries. New router P0 references retain MTreg107 for the B1
context and129 for B16. Full-model WD0/WD1 programs remain stack-free.

Verified same-card alternating unroll4 P-PAIR completes two fresh processes
per arm, each warmup1/timed2. All 12 full generated sequences per batch
(both arms, both processes, warmup and timed repeats) are bitwise equal.
The new programs pass 512-step C-1/C-2. Threshold is max(1% of control,
maximum within-arm range), applied to the difference of two-process medians.

| B | Fresh control ms | Unroll4 ms | TPOT change | Threshold ms |
|---|---:|---:|---:|---:|
| 1 | 4.524784 | 4.405260 | -2.642% | 0.045248 |
| 16 | 17.510438 | 17.216499 | -1.679% | 0.175104 |

Both improvements are distinguishable. Twelve process/median rows are
appended under explicit `router_unroll4_ppair_old/new` implementation labels;
they do not replace earlier measurements or count as top-four selections.
The original measured catalog is not relabelled as new-source data.
Formal routing and final solver/model acceptance remain open.

Further partial unrolling of the dispatch scans, still computing the same
counts and order, passes another100 legacy-bitwise cases. Three-process
diagnostic P0 router medians are about21.83us(B1) and59.6us(B16), versus
about36us/85.8us for unroll4. These are isolated benchmarks without qualified
cost-table provenance; key ordinals use different random input seeds.
The full-model C-1/C-2 and P-PAIR outcomes are recorded below.

Verified the subsequent unroll16 full-model P-PAIR completes with unchanged
coordinates, prefetch widths and numerical contracts. Both endpoints pass
512-step C-1/C-2, and all12 generated1024-token sequences per batch match
across arms. Twelve additional process/median rows are appended separately.

| B | Fresh unroll4 control ms | Unroll16 ms | TPOT change | Threshold ms |
|---|---:|---:|---:|---:|
| 1 | 4.406173 | 4.116311 | -6.579% | 0.044062 |
| 16 | 17.216037 | 16.588394 | -3.646% | 0.172160 |

Both gains are distinguishable. The backend loop change is committed as
`a5f9e8d33`; this does not promote a final plan or clear formal routing.
Its SGLang/TileMega speedup is about1.025x atB1 and0.889x atB16, below1.1x.
The original B4/B16 predicted-rank0 coordinates still spill with this source
(MT256/frame256B and368B); loop optimization does not solve those original
register-heavy configurations. Thirty further resource rows, including
these failures and routing builds, retain the old resource-CSV prefix.

An isolated unroll64 variant also passes100 legacy-bitwise cases and three
diagnostic benchmark processes. Router medians about20.5us/53.9us are only
about1.3us/5.7us below unroll16. It remains diagnostic: no full-model gain or
default is inferred from that incremental microbenchmark difference.

## Formal routing and stability (verified failures remain open)

The unroll16 diagnostic plans follow their own 576-thread-toD histories
before the per-plan512 branch. Actual512 zero-launch queries are resident1,
local0. Formal64-step eager-HF routing results are:

| B | Positions | Exact sets | Boundary ties | Non-tie failures |
|---|---:|---:|---:|---:|
| 1 | 1536 | 1372 | 128 | 36 |
| 16 | 24576 | 21766 | 2373 | 437 |

Both negative guard exits are checked against complete bound reports. The
diagnostic65-token outputs match all six corresponding normal generations.
No tie threshold, oracle, rounding point or legacy fallback is changed.
Existing matched-input controls in `MX5_moe_native_c550.md` locate one earlier
difference at consumer RMS/BF16 rounding; they do not establish the cause of
every failure in the current36/437 positions.

New B4/B8 coverage passes512-step C-1/C-2 and completes P-PAIR. B2's first
coverage check fails with a shader illegal address before producing C-2
tokens; it is retained and has no timing row. Fresh old/new controls with
per-step synchronization both complete511 decode steps. Two subsequent
fresh exact C-2 checks pass512, one also completing C-1. The initial fault is
not yet localized or cleared; successful repeats are not relabelled as a
fix. A fresh full P-PAIR retry preserves the failed campaign and completes.
In total, three subsequent fresh exact C-2 processes pass512 (two also
complete C-1). Five fresh packing/constructor-only processes pass with zero
native model launches; phase logging may perturb host timing.

The failed shell pipeline exposes a separate reporting bug: without
pipefail, the C-2 output filter can mask its child's exit and start HF with
no token file. `scripts/mx_check_modes.sh` now propagates pipeline failures.
A CPU regression makes the C-2 stub exit23 while its filter succeeds and
asserts that HF is not called; seven targeted CPU tests pass. This improves
error reporting and does not fix or explain the initial illegal address.

## Current five-batch development snapshot (verified)

All five plans compile the same current router-header SHA256
`6ead70d71c6560a9217fc50af486a2fc494f1af4375200b6ddf8e31f8c1e6a8a`.
Each has512-step C-1/C-2 and two fresh uninstrumented timing processes;
all12 full sequences per P-PAIR are bitwise equal across arms. B4 uses the
D1 recovery control in this comparison; its earlier D2 result remains
separate. B2 retains the unresolved earlier illegal-address record.

| B | Current TPOT ms | Paired control ms | TPOT change | Fixed fastest ms | Baseline / TileMega |
|---|---:|---:|---:|---:|---:|
| 1 | 4.116311 | 4.406173 | -6.579% | 4.2208 | 1.0254x |
| 2 | 5.060573 | 5.531358 | -8.511% | 5.4688 | 1.0807x |
| 4 | 7.496973 | 7.953198 | -5.736% | 7.5028 | 1.0008x |
| 8 | 11.380065 | 11.981181 | -5.017% | 10.7495 | 0.9446x |
| 16 | 16.588394 | 17.216037 | -3.646% | 14.7491 | 0.8891x |

B1/B16 paired controls are unroll4; B2/B4/B8 controls precede the router
optimizations. These are distinct comparison cohorts, not one common old
implementation. All five gains exceed their actual max(1%, round-range)
thresholds. Eighteen coverage process/median rows are appended separately;
`MX5_moe_recovery_c550.csv` preserves full precision, TTFT/E2E/throughput,
estimated MBU, identities and open-gate status for the current five plans.

The1.1x target is approached atB2 but is not achieved across the batch grid.
In particular B8/B16 remain slower than fixed SGLang. No final solver choice,
MoE acceptance, default promotion, or DNN phase advance is claimed.

One further CPU-only expert-down diagnostic hoists a repeated direct
epilogue row address at unchanged original B16 geometry. It worsens the
stack frame from368B to384B (MT256); the isolated overlay is not adopted or
launched. Raw: `grouped_direct_row_hoist_cpu/ACTUAL_CPU_COMPILE.json`.

This recovery campaign records81 guarded jobs and3.101389 physical-card
wall hours (per-card interval unions, not hardware busy cycles). Including
the previous closed ledger, the cumulative upper bound is113.102778 GPUh;
the user waived the cumulative cap. All guarded jobs are complete. The
fixed MX-4 serving/LLM-baseline/DNN-baseline CSVs remain byte-identical to
`22a352242`; none was remeasured. Raw snapshot/accounting proof:
`ACTUAL_CURRENT_FIVE_MOE_SNAPSHOT_AND_ACCOUNTING_CPU.json`.

Open boundaries: `python/tilemega/moe/native_routing_check.py:138` rejects
the formal route mismatches; it is a gate location, not proof of one cause.
The B2 failure did not produce a faulting instruction or enough phase data
to distinguish construction from generation at
`python/tilemega/serving/check_modes.py:46`/`:50`.
`tools/commands/mx_moe_v2.cpp:107` still requires an explicit native pin;
the complete final cost catalog and resource-qualified top-four selection
remain unfinished. Dense G-LLM/G-B issues and DNN work are unchanged.
