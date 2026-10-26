# MX-5 LLM C550 results (in progress)

Verified checkpoint: current cc29/A719 source has completed 3/25 dense selections (Llama1B B1 and Qwen0.6B B1/B16); three selections for prior 3ef/Af8 remain historical. MoE remains 0/5; G-LLM and the default switch remain open.

Authority: `/root/Prompt/TileMega_MX5_prompt.md`, SHA256 `75431f98a6286063cc73effe5ece2ed3d462304de4c4192dbd4023efe2bd6f73`. Prior source/producer: `3ef688aa4d0bd7be8135e17e9db3274078dce760` / `e6130715bb4819438927375b216cb4dd5bebae3e669cad25e8303ef01783a267`. Current: `cc29d78304d297346d657c0fbca0a86e76b1f60f` / `24fa50ed913bce7791f575447a97e06c18a9bfce91c337c0bcc9191179c7c3c7`.

All measurements use BF16 greedy, ISL/OSL 64/1024, warmup 1 plus timed 2 per fresh process, two fresh processes per candidate, no trace/diagnostic/step clocks and watchdog 0. TPOT = (E2E − TTFT)/1023. Correctness twins use watchdog 1. External baselines were not remeasured.

| Cell | Source | TPOT ms | TTFT ms | E2E s | token/s | MBU* | vs MX-4 | /vLLM | /SGLang |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B B1 | prior 3ef/Af8 | 1.944021 | 9.376043 | 1.998110 | 512.484 | 0.4175 | -1.993% | 0.6323 | 0.7909 |
| Qwen3-0.6B B16 | prior 3ef/Af8 | 3.567243 | 60.632317 | 3.709922 | 4416.266 | 0.4067 | -21.376% | 0.8873 | 0.8207 |
| Llama-3.1-8B B1 | prior 3ef/Af8 | 10.560823 | 41.125858 | 10.844848 | 94.423 | 0.9215 | -1.444% | 0.9078 | 0.9523 |
| Llama-3.2-1B B1 | cc29/A719 | 2.157002 | 8.957554 | 2.215571 | 462.183 | 0.7449 | -2.445% | 0.7232 | 0.8537 |
| Qwen3-0.6B B1 | cc29/A719 | 1.954394 | 9.284968 | 2.008630 | 509.800 | 0.4153 | -1.470% | 0.6357 | 0.7951 |
| Qwen3-0.6B B16 | cc29/A719 | 3.580819 | 60.759784 | 3.723937 | 4399.644 | 0.4052 | -21.077% | 0.8907 | 0.8239 |

*MBU is inferred from `(streamed linear/head BF16 weights + KV at average L=576)/(1.55 TB/s × TPOT)`. Full embedding-matrix reads, activation traffic, writes and cache effects are excluded. This is a byte lower-bound model, not a measured DRAM counter. Aggregate MBU, throughput and pred/meas are calculated from aggregate times; raw process rows are retained.

Prior-source candidate tables and D7 details: [historical selections](MX5_llm_prior_body_c550.md).
Current-source selections below do not relabel those measurements.

## cc29 five development controls

Verified: each cell passes actual 512-step C-1/C-2, with two fresh WD0 processes
using the standard 64/1024 protocol. These manual same-configuration source
regressions satisfy the five numerical/timing predicates, not final solver
selection or full pipeline conformance. New A719 scoped origins are now complete.

| Cell | Threads | TPOT ms | vs MX-4 |
|---|---:|---:|---:|
| Qwen0.6B B1 | 512 | 1.945727 | -1.907% |
| Qwen0.6B B16 | 512 | 3.573892 | -21.229% |
| Llama1B B1 | 576 | 2.153378 | -2.609% |
| Llama8B B1 | 512 | 10.557096 | -1.479% |
| Llama8B B16 | 512 | 13.218706 | -9.095% |

Raw: `blockinit_remaining3dev_check_time_v1`,
`blockinit_llama1b576_check_time_v1`, `blockinit_B16_manual_timing_v1` under
`runs/mx5/`; combined proof `blockinit_five_development_predicates_v1`.
The prompt4.7.3 crosssegment-overlap claim remains unfulfilled: actual next K/V
requests retire at the first segment-merge CTA sync (AttentionV2.h:318). No
synchronization substitution or default promotion is made.

## Llama1B B1 current-source original top four

Verified: all four independent576 originals pass C-1/C-2 at512 steps and
eight fresh WD0 measurements on card7. Candidate0 is selected by the prescribed
1% rule. Each normal WD0/WD1 is MT164/frame0/API0/resident1; controlled cap
diagnostics from other cells are not used. This was the first current-source selection.

| ID | Predicted ms | Process1 ms | Process2 ms | Median ms | pred/meas | Selected |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 2.180651 | 2.157455 | 2.156549 | 2.157002 | 1.010964 | yes |
| 1 | 2.181271 | 2.157362 | 2.157711 | 2.157537 | 1.011001 | no |
| 2 | 2.181671 | 2.158640 | 2.157389 | 2.158015 | 1.010962 | no |
| 3 | 2.181831 | 2.160851 | 2.161075 | 2.160963 | 1.009657 | no |

Actual TN/WN/D/Pw: qkv32/1/1/1, o32/1/2/2, gate_up64/2/2/4,
down32/1/2/2, head32/8/2/8; allTileRange. Attention b_min1/PK2; threads576;
81 stages/barriers perstep. Allfour C1 maxgap0.09375 and C2 exact.
Evidence: `runs/mx5/blockinit_llama1b576_top4_execution_v2/actual/`; exact pin
`configs/pins/mx5/blockinit_cc29/llama3_2_1b_B1.json`. Its actual selected-stage
ledger below uses matchingPK2; crosssegment/I1 attribution gaps remain open.

Llama1B selected-pin DIAG interventions,us/nominal layer; two fresh processes
per mask, reverse order in round2, no trace. These are nonadditive net deltas.

| Stage | Selected DIAG net | Matching uninstrumented bench |
|---|---:|---:|
| qkv | 9.708 | 12.510 |
| attention | 13.609 | 15.195 |
| o | 6.078 | 10.160 |
| gate_up | 43.138 | 52.660 |
| down | 21.335 | 25.340 |

Head/token-history-confounded net 317.360us versus head bench326.810us;
mask63 floor341.048us/step retains prefetch/barriers/control/timer.
No additive or per-site latency attribution is claimed. The selected attention
bench itself exceeds inferred2–3us; no measured/bench integration penalty>1.2
is observed. Raw:`blockinit_selected_llama1b576_ledger_runtime_v1/`.

## Qwen0.6B B1 current-source original top four

Verified: all4 C-1/C-2 pass512 steps (maxgap0.015625); eight fresh WD0
processes complete. Candidate0 is retained by1% rule; candidate3 has a slightly
smaller median. All actual512 originals/WD1 twins are frame0/resident1.

| ID | Predicted ms | Process1 ms | Process2 ms | Median ms | pred/meas | Selected |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 1.996530 | 1.959958 | 1.948829 | 1.954394 | 1.021560 | yes |
| 1 | 1.996600 | 1.954595 | 1.956875 | 1.955735 | 1.020895 | no |
| 2 | 1.996710 | 1.954150 | 1.963178 | 1.958664 | 1.019424 | no |
| 3 | 1.996820 | 1.953245 | 1.954266 | 1.953756 | 1.022042 | no |

Actual TN/WN/D/Pw: qkv32/2/2/2,o16/1/2/2,gate_up64/1/1/1,
down16/1/2/2,head128/8/1/2;allTileRange. Attention b_min1/PK0,threads512,
141stages/step. Genuine576 reduction counts1/1/0/0 carry into512 with no reset;
cap227 diagnostic frame68/resident0 stays ineligible. Allfour current-source
D7/ResolvedTop4 audits and root independent replay pass their binding scope.
Raw:`blockinit_qwenB1_top4_execution_v1/actual/`; exact pin
`configs/pins/mx5/blockinit_cc29/qwen3_0_6b_B1.json`.

QwenB1 actual selected-pin nonadditive DIAG interventions (us/nominal layer),
two fresh processes per mask, reverse order in round2, no trace:

| Stage | Selected DIAG net | Matching uninstrumented bench |
|---|---:|---:|
| qkv | 6.723 | 10.100 |
| attention | 23.151 | 22.070 |
| o | 4.704 | 8.360 |
| gate_up | 9.823 | 13.770 |
| down | 6.566 | 10.030 |

Head/token-history-confounded net190.932us vsbench195.300us;
mask63 floor331.180us/step retains prefetch/barriers/control/timer.
Attention net/bench=1.049009; its bench itself exceeds inferred2.5–3.5us.
No additive sum or unmeasured per-site split is claimed.

## Qwen0.6B B16 current-source original top four

Verified: all4 C-1/C-2 pass512 steps; eight fresh WD0 processes complete.
Candidate0 is selected by1% rule. Normal512 originals/WD1 twins are allframe0
and API0/resident1. Current-source3/25; default target remains MX4.

| ID | Predicted ms | Process1 ms | Process2 ms | Median ms | pred/meas | Selected |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 3.641533 | 3.582031 | 3.579607 | 3.580819 | 1.016955 | yes |
| 1 | 3.644603 | 3.581527 | 3.580259 | 3.580893 | 1.017792 | no |
| 2 | 3.645143 | 3.582682 | 3.580811 | 3.581747 | 1.017700 | no |
| 3 | 3.645333 | 3.586789 | 3.584752 | 3.585770 | 1.016611 | no |

Actual TN/WN/D/Pw: qkv64/1/1/1,o16/1/1/1,gate_up64/1/1/1,
down16/1/1/1,head128/8/1/2;allTileRange. Attention b_min2/PK0,threads512,
141stages/step. Actualcanonical576 reductions1/0/0/2,used5120; no reset.
Allfour current-source D7/ResolvedTop4 audits and root independent replay
validate their binding scope; cap227 diagnostics stay ineligible.
Raw:`blockinit_qwenB16_top4_execution_v1/actual/`; exact pin
`configs/pins/mx5/blockinit_cc29/qwen3_0_6b_B16.json`.

Unfinished: remaining22 dense cells,MoE5/LLM30 acceptance,MX4-final P-PAIR
(exact archived producer unavailable),current I1/per-body attribution and the
recorded crosssegment overlap claim. Historical timings do not fill these gates.
