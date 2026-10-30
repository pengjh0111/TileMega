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
Full-model C-1/C-2 and P-PAIR are running before any adoption decision.
