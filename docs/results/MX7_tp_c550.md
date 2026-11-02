# MX-7: Qwen3-1.7B TP2 / TP4 on C550

**verified — optimization in progress:** the matrix below is the original vector8 implementation.
The reselected pair-lane TP2 B16 final is3.561300ms; TP4 B16 final is3.071796ms; both selected rebuilds pass C-1 and5/5 fresh C-2 checks.
The ten-scope replacement and new delay/trace gates are running; `MX7_tp_sm90_comparison_c550.md` records current evidence.

Evidence: **verified** = executed here; **stated** = user specification; **inferred** = reasoning.
Raw successes, failures, generated sources and binaries remain under `runs/mx7/`.

## Scope and source identity

- **stated** BF16 greedy, ISL/OSL=64/1024, TP2/TP4 × B=1/2/4/8/16.
- **verified** model directory `/root/models/qwen3_1_7b_rand`.
- **verified** branch `tilemega-metax`; starting HEAD `6cab45b9604e306fef1916863fe16b09d895c32f`.
- **verified** evidence aggregation HEAD `8b32ec8f77907711ee60141943679e79b651cef7`; commits since start: 102.
  The report/publish commit follows this evidence HEAD; final HEAD/push status is in the delivery message.
- **verified** specification SHA256 `5caed7edd4477b8fb63d33bed9254bbc4e33d92041eb5969113fefeb805d7cee`.
- **stated** subsequent all-card authorization and torch.compile continuation:
  SHA256 `f67014253a3cca7234429ff3671e293c0104fb04b68fde1bb9dc52b518c0b284`.
- **verified** TP1 passes the frozen CU-byte/function-instruction-count gate; protected MoE files/final CSV are unchanged; no MoE performance runs.

## Environment, C0 and execution sequence (10-line onboarding summary)

- **verified** MACA3.8.2.6, cucc/mxcc1.0.0(890c70f3aa), driver3.8.1.
- **verified** Torch2.10.0+metax3.8.2.2; Transformers5.12.0; SGLang0.5.14+detached.c432db2fb.maca3.8.2.122; vLLM0.25.0+empty.
- **verified** NCCL/Gloo available; torch.compile TP uses NCCL. TileMega uses direct MACA P2P.
- **verified** read-only `mx-smi topo -m/-p/-d`: MX links, PCIe NODE labels; reachability is separately qualified.
- **verified** LL8/fine-grained allocation/volatile AS1 scalar64 store/`__ldcg`64 load: pair/all-to-all P1024/P262144 each5/5 ×2000epochs ×104CTAs, zero tearing/timeouts.
- **verified** TP2 main{1,3}:2373.504ns single-way; secondary{4,5}:2629.120ns (five processes).
- **verified** At104CTAs on{1,2,3,5}, selected writes:9.385GB/s to one peer,19.088GB/s aggregate to three peers (five processes).
- **verified** TP4 B1 retains{1,3,4,5}; B2/B4/B8/B16 use requalified{1,2,3,5} after GPU4 remained occupied; GPUs0/6/7 unavailable.
- **inferred** Each rank executes qkv→attention→o+AR→gate_up→down+AR per layer, then head LL argmax exchange produces the common token.
- **inferred** Producers write BF16 partials to all peers, then read q in ascending order; epoch=past+1, prompt runs from past0 and suppresses early token writes.

Source hashes, frozen references, exact build/run commands and topology: `runs/mx7/sync/onboarding.md`.

SM90 comparison and ongoing communication diagnostics: `MX7_tp_sm90_comparison_c550.md`; combined measurements: `MX7_tp_comparison_c550.csv`.

## Baselines

**verified** Two fresh processes/cell, each warm1+timed2; TPOT=(E2E_N1024−E2E_N1)/1023.
**stated** Baselines are collected once and reused across physical GPU groups; changing groups does not trigger another measurement.
**verified** The original30 passing cells are frozen in `MX7_tp_baselines_frozen_c550.csv`; original GPU groups remain in their raw records.
TP1 values and baseline scaling efficiencies are retained in the frozen reference. Partial group-change repeats remain archived and are not used.

| TP | B | vLLM ms | SGLang ms | torch.compile ms |
|---:|---:|---:|---:|---:|
| 2 | 1 | 3.912 | 3.084 | 14.034 |
| 2 | 2 | 4.002 | 3.439 | 16.259 |
| 2 | 4 | 4.032 | 3.487 | 16.603 |
| 2 | 8 | 4.302 | 3.777 | 21.627 |
| 2 | 16 | 4.625 | 4.216 | 32.919 |
| 4 | 1 | 3.115 | 2.530 | 16.607 |
| 4 | 2 | 3.852 | 3.042 | 15.453 |
| 4 | 4 | 3.694 | 3.083 | 15.342 |
| 4 | 8 | 3.763 | 3.177 | 16.940 |
| 4 | 16 | 3.938 | 3.366 | 22.851 |

## TileMega final selected plans

**verified** Two additional fresh processes/cell, warm1+timed2; TPOT=(E2E−TTFT)/1023.
TTFT includes prompt mode and63 extra head evaluations. Ratios use the same TP degree.
Efficiency=TP1_frozen_TPOT/(TP×TP_TPOT); throughput counts all B×1024 output tokens.

| TP | B | GPUs | TPOT ms | TTFT ms | E2E ms | tokens/s | fastest/TM | TP1/TM | efficiency |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 1,3 | 2.518 | 156.069 | 2732.281 | 374.778 | 1.224 | 1.298 | 64.9% |
| 2 | 2 | 1,3 | 2.802 | 173.801 | 3039.908 | 673.706 | 1.227 | 1.288 | 64.4% |
| 2 | 4 | 1,3 | 3.655 | 256.689 | 3995.686 | 1025.105 | 0.954 | 1.073 | 53.6% |
| 2 | 8 | 1,3 | 4.806 | 323.228 | 5239.997 | 1563.360 | 0.786 | 0.886 | 44.3% |
| 2 | 16 | 1,3 | 7.246 | 464.117 | 7877.102 | 2079.953 | 0.582 | 0.715 | 35.8% |
| 4 | 1 | 1,3,4,5 | 2.459 | 152.856 | 2668.781 | 383.696 | 1.029 | 1.329 | 33.2% |
| 4 | 2 | 1,2,3,5 | 2.776 | 172.343 | 3012.120 | 679.920 | 1.096 | 1.300 | 32.5% |
| 4 | 4 | 1,2,3,5 | 3.784 | 236.795 | 4108.002 | 997.079 | 0.815 | 1.036 | 25.9% |
| 4 | 8 | 1,2,3,5 | 6.210 | 420.380 | 6773.014 | 1209.508 | 0.512 | 0.686 | 17.1% |
| 4 | 16 | 1,2,3,5 | 12.506 | 885.048 | 13678.825 | 1197.764 | 0.269 | 0.414 | 10.4% |

Each engine ratio, both process values, pin/binary/cost SHA and source identity are in `MX7_tp_final_c550.csv`.
All ten pins come from source-bound k-best/top4 and the1% measured selection rule; no development coordinates are final.
**verified** Per-cell sealed cost tables (SHA256): `44c318739d7c50dd1551a254a5d8af28adee38204a158d3cf12dce8404e14229`, `7ad97df7881ba38ddaa0b720fb1eb6dc287fb60de042d989bee76fea3a3ecaad`.
Competitive noisy keys receive two extra fresh processes; remaining noisy keys use p90.

Selected communication cost terms (diagnostic, µs; final column is an **inferred**28-layer projection):

| TP | B | o TN | down TN | extra o AR | extra down AR | total AR ms/step |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 32 | 32 | 5.87 | 6.64 | 0.350 |
| 2 | 16 | 16 | 32 | 64.86 | 69.82 | 3.771 |
| 4 | 1 | 32 | 32 | 13.35 | 13.08 | 0.740 |
| 4 | 16 | 16 | 16 | 170.00 | 169.30 | 9.500 |

## Correctness and resources

- **verified** shard/guard/ABI/selection/trace CPU tests pass (individual runs retained in correctness CSV).
- **verified** AR and head units: TP2/TP4 × B1/B16, each5/5 fresh processes; exact ordered BF16 reference.
- **verified** injected-delay development litmus: TP2/TP4 × B1/B16, each5/5 fresh1024-token processes;
  sequences exactly match ordinary builds; ordinary development C-2 also passes5/5 in all four cells.
- **verified** all40 qualified candidates pass512-position HF C-1 and two fresh1024-token C-2 processes.
- **verified** all ten selected rebuilds pass512-position C-1 and five fresh1024-token C-2 processes;
  every generated token agrees across ranks, including all timing runs.
- **verified** actual all-rank resident=1; MTreg≤255, stack0, static-smem0, dynamic LDS≤64KiB.
  Rejected candidates and actual resources remain in `MX7_resources_c550.csv`.
- **verified** eight frozen plans pass generated-CU byte equality and per-function instruction-count equality
  after relevant frontend/codegen/backend/runtime changes (`MX7_frozen_gate_c550.csv`).

## TP4 trace: rank0 per-layer medians (µs)

**verified** One diagnostic process/B,1023 decode steps,104CTAs/rank; o/down include AR, head includes exchange.
Each duration uses max CTA release minus the previous stage release. Trace values are not E2E acceptance data.

| B | Layer | qkv | attention | o+AR | gate_up | down+AR |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 7.25 | 25.31 | 18.44 | 13.70 | 22.90 |
| 1 | 1 | 7.51 | 25.16 | 19.70 | 13.61 | 21.86 |
| 1 | 2 | 7.49 | 25.16 | 19.41 | 13.58 | 21.76 |
| 1 | 3 | 7.52 | 25.32 | 19.46 | 13.59 | 23.87 |
| 1 | 4 | 7.50 | 25.13 | 18.45 | 13.60 | 22.47 |
| 1 | 5 | 7.51 | 25.24 | 19.94 | 13.60 | 21.93 |
| 1 | 6 | 7.51 | 25.22 | 20.03 | 13.59 | 21.96 |
| 1 | 7 | 7.48 | 25.32 | 20.24 | 13.64 | 22.08 |
| 1 | 8 | 7.50 | 25.22 | 19.31 | 13.64 | 22.55 |
| 1 | 9 | 7.50 | 25.22 | 19.96 | 13.61 | 21.95 |
| 1 | 10 | 7.48 | 25.12 | 20.09 | 13.57 | 21.98 |
| 1 | 11 | 7.49 | 25.16 | 20.41 | 13.62 | 24.22 |
| 1 | 12 | 7.50 | 25.27 | 19.07 | 13.61 | 21.95 |
| 1 | 13 | 7.49 | 25.11 | 19.66 | 13.62 | 21.71 |
| 1 | 14 | 7.50 | 25.34 | 19.67 | 13.62 | 21.63 |
| 1 | 15 | 7.50 | 25.27 | 19.78 | 13.59 | 21.53 |
| 1 | 16 | 7.46 | 25.20 | 19.65 | 13.68 | 21.62 |
| 1 | 17 | 7.50 | 25.26 | 19.52 | 13.64 | 21.62 |
| 1 | 18 | 7.50 | 25.29 | 19.47 | 13.63 | 21.59 |
| 1 | 19 | 7.50 | 25.16 | 19.60 | 13.65 | 23.31 |
| 1 | 20 | 7.52 | 25.15 | 17.97 | 13.65 | 23.13 |
| 1 | 21 | 7.51 | 25.16 | 20.13 | 13.64 | 22.12 |
| 1 | 22 | 7.51 | 25.10 | 20.01 | 13.65 | 22.08 |
| 1 | 23 | 7.49 | 25.24 | 20.61 | 13.65 | 22.01 |
| 1 | 24 | 7.51 | 25.28 | 19.33 | 13.65 | 22.36 |
| 1 | 25 | 7.52 | 25.27 | 20.07 | 13.63 | 21.91 |
| 1 | 26 | 7.50 | 25.18 | 19.99 | 13.64 | 21.61 |
| 1 | 27 | 7.50 | 25.13 | 20.31 | 13.62 | 29.33 |
| 1 | head | 109.59 | — | — | — | — |
| 16 | 0 | 14.04 | 42.63 | 171.45 | 16.29 | 154.07 |
| 16 | 1 | 14.35 | 42.78 | 169.79 | 16.55 | 155.73 |
| 16 | 2 | 14.46 | 42.73 | 167.44 | 16.31 | 164.96 |
| 16 | 3 | 14.50 | 42.52 | 165.66 | 16.33 | 191.18 |
| 16 | 4 | 14.56 | 42.80 | 165.09 | 16.39 | 201.81 |
| 16 | 5 | 14.36 | 42.55 | 165.26 | 16.31 | 204.36 |
| 16 | 6 | 14.39 | 42.76 | 166.90 | 16.29 | 213.04 |
| 16 | 7 | 14.41 | 42.50 | 166.47 | 16.26 | 207.61 |
| 16 | 8 | 14.47 | 42.66 | 166.61 | 16.27 | 210.90 |
| 16 | 9 | 14.52 | 42.54 | 166.42 | 16.23 | 207.61 |
| 16 | 10 | 14.50 | 42.68 | 165.18 | 16.28 | 204.40 |
| 16 | 11 | 14.44 | 42.60 | 165.61 | 16.29 | 206.45 |
| 16 | 12 | 14.49 | 42.79 | 165.93 | 16.27 | 208.39 |
| 16 | 13 | 14.35 | 42.55 | 165.54 | 16.32 | 207.31 |
| 16 | 14 | 14.44 | 42.70 | 166.96 | 16.29 | 213.77 |
| 16 | 15 | 14.48 | 42.58 | 166.50 | 16.32 | 210.68 |
| 16 | 16 | 14.45 | 42.54 | 166.85 | 16.32 | 209.78 |
| 16 | 17 | 14.47 | 42.57 | 166.22 | 16.28 | 205.57 |
| 16 | 18 | 14.47 | 42.50 | 166.59 | 16.31 | 208.14 |
| 16 | 19 | 14.39 | 42.47 | 166.53 | 16.26 | 209.02 |
| 16 | 20 | 14.46 | 42.57 | 166.65 | 16.31 | 208.51 |
| 16 | 21 | 14.40 | 42.57 | 166.47 | 16.27 | 208.64 |
| 16 | 22 | 14.43 | 42.46 | 165.11 | 16.24 | 204.80 |
| 16 | 23 | 14.38 | 42.54 | 165.55 | 16.31 | 208.21 |
| 16 | 24 | 14.41 | 42.51 | 165.07 | 16.30 | 202.34 |
| 16 | 25 | 14.46 | 42.53 | 165.34 | 16.27 | 207.78 |
| 16 | 26 | 14.40 | 42.42 | 166.63 | 16.31 | 210.28 |
| 16 | 27 | 14.45 | 42.58 | 166.76 | 16.29 | 211.22 |
| 16 | head | 212.99 | — | — | — | — |

Maximum inter-rank stage-duration differences (all layers/steps):
- **verified** B1: o=44.410µs, down=6.600µs.
- **verified** B16: o=83.420µs, down=98.600µs.

Per-rank medians, raw losslessly compressed traces and source hashes: `MX7_tp_trace_c550.csv`.

## Autonomous handling and preserved failures

- **verified, B2/B4** original torch.compile lacked accelerate, then fullgraph rejected HF TP gradient hooks.
  Isolated accelerate1.15.0 plus compiled_autograd=True adapts the compiler; no hooks removed or eager fallback.
- **verified, B1** initial native builds reached256MTreg and336–472 stack bytes. Approved per-item send/receive
  plus constant-field peer-pointer selection removes operand materialization; TP1 frozen output remains identical.
- **verified, B4** old GPU6 queue initialization caused the historical A5 stop before LL launch.
  All-card authorization permits healthy groups; optional system-atomic probes also coincided with GPU0/7 loss.
  Causation is unresolved. Old failures remain; unsafe optional combinations are not retried.
  The optional primitive-combination matrix is incomplete; the chosen production combination is fully qualified.
- **verified, B4** one driver bundled20 fresh TP4 checks under one900s guard: aggregate timeout124.
  Individual runs completed in52–65s. B16 injected-delay checks were isolated and pass5/5; no limit reduction.
- **verified, B4** calibration B8/TN128 non-AR kernel used an undersized fixed LDS workspace.
  Host/device now use the actual GEMM LDS bound; old illegal-access record retained. Repaired scopes pass.
- **verified, B4** cost queue rebalanced five incomplete chunks onto idle cards, based solely on completion;
  completed three-process chunks retained; old partial records preserved and relative paths canonically validated.
- **verified, B4** resource ledger assumption corrected by appending CU-derived LDS values; old rows retained.
  One pending query was prematurely labeled rejected; an appended correction and terminal-cohort checks fix the ledger.
  Trace aggregation now normalizes empty layer keys; one repeated head reference remains, pointing to the same process.
- **verified, B4** low disk space handled by hardlinking byte/permission/mtime-identical sealed inputs (219.9MiB freed);
  logical paths and hashes are unchanged. Rank-by-rank lossless trace compression bounds temporary disk to about1.5GiB.
- **verified, B4** read-only recovery preflight validates four candidate bundles and the selected final artifact before GPU work.
- **verified, B7** cost/constants noise supplements follow the competitive2% rule, otherwise p90.

- **verified, B4** earlier timing pairs shared one guard preflight while retaining two fresh processes, each warm1/timed2;
  per-process PID/command/exit records and guard provenance remain; each timing job retains the900s limit.

- **verified, B5** TP4 B2 candidate3/4 timing cohorts exited guard75 after external GPU3/GPU5 load appeared.
  Invalid cohorts are excluded; each original cohort gets at most one new two-process timing cohort after idle checks.
  Successful candidate builds/checks/timings remain unchanged; failed raw metrics and source-bound retry receipts remain.
- **verified, B4/B5** final TP4 B2 checks later met external GPU3 interference after three C-2 processes.
  The whole interrupted cohort is unqualified; original artifacts/timings remain archived and were not promoted after the group change.
- **verified, B4** resume validation first found a manifest field mismatch and then an archived-root path error.
  Both occurred before a GPU child launch; original logs/receipts are retained separately from the one B5 retry.

- **verified, B4** remaining scopes guard each fresh process separately; passing runs survive later interruptions.
  C-2 still needs5 selected/2 candidate processes; timing still needs2 warm1/timed2 processes. Guard thresholds are unchanged.
  After CUDA-stream initialization stalls, CPU scheduling additionally waits for≤64MiB unattributed memory/card; this narrows launch admission and changes no guard or device code.
- **verified, B4** B16 trace GPU capture and CPU analysis are separate; real B1 reanalysis matches all141 original stage statistics.

- **verified, B4** GPU4 stayed occupied, so only the four unfinished TP4 scopes moved to{1,2,3,5} under all-card authorization.
  P2P and both LL8 all-to-all sizes pass5/5; AR/head units are rechecked and45 TP4 constants are new.
  The partial baseline-repeat campaign was canceled by the user; all comparisons reuse the original frozen30 cells.
  1840 single-rank cost rows and six completed final cells are retained. Remaining scopes are selected again with group-bound costs.
- **verified, B4** one calibration aggregator shadowed the group string with a list before subprocess creation; fixed before GPU launch;
  the original CPU error log remains. No numerical test, sample count or guard threshold was changed.

- **verified, B5** new-group SGLang B4 first initialization exceeded900s; the guard terminated only its own job and retained logs.
  The retry also timed out; both failed repeat attempts remain archived and do not replace the original passing baseline.

- **verified, B4** a one-peer bandwidth launch exposed four devices; the host rejected it before peer kernels. Corrected two-device launches retain that failure.
- **verified, B5** an extra selected-pin/new-group B1 delay check and its one retry met GPU5 interference(guard75) during CUDA-stream creation.
  This additional experiment remains incomplete(1 pass,2 invalid attempts); the four originally required development litmus cohorts remain5/5 each.
  After the startup scheduling fix, a new full selected-pin cohort at B1 and B16 passes5/5 each; the prior read-only stacks/invalid attempts remain.
  No incomplete cohort is reported as a synchronization pass.

## Deviations from inferred expectations

- **verified** TP2 B1: 2.518ms versus stated inferred2.35–2.5ms; no goal-based rework.
- **verified** TP2 B16: 7.246ms versus stated inferred3.3–3.4ms; no goal-based rework.
- **verified** TP4 B1: 2.459ms versus stated inferred1.85–2.05ms; no goal-based rework.
- **verified** TP4 B16: 12.506ms versus stated inferred2.5–2.7ms; no goal-based rework.
- **verified, C** selected pred/meas range0.931–1.000; outliers retained in selection CSV.
- **inferred** P2P packet throughput and measured per-AR overhead limit scaling; trace gives stage locations.
- **stated, C** prompt-mode TTFT includes work absent from the separate TP1 prefill path.
- **verified, C** largest torch.compile fresh-process range=30.2% (TP4 B1); both round values remain in CSV, using the specified two-process median.

## GPU accounting and completion

**inferred** child-job wall time, includes initialization/cleanup, excludes lock waiting/preflight; not hardware busy time.
Per-card hours: 0=0.080, 1=10.725, 2=4.528, 3=10.752, 4=5.458, 5=9.992, 6=2.681, 7=2.685; total=46.902 card-h.
**verified** CP0–CP3 production-path evidence and all required CSVs/pins are present; selected-path tests are complete.
**inferred** further communication designs require a new specification; none added beyond this prompt.

## SM90 comparison and optimization continuation

**verified** SM90 audit: `MX7_tp_sm90_comparison_c550.md`. Shared AR now assigns one contiguous LL8 packet/lane, preserving the wire format, rank-sum order and rounding. Eight frozen builds pass8/8; four AR corner scopes each pass5/5 fresh processes.
**verified** TP4 B16 same-geometry development P-PAIR:12.526515→3.081392ms(4.065×); SGLang/TM1.092×. Full-model C-1 passes; five C-2 starts and the original1024-token sequence match. `MX7_tp_optimization_c550.csv` retains this control separately from the original final table above.
**verified, B4** Initial SDK stream cleanup exited−6 after correct tokens; the failure is retained. Releasing pinned/GPU storage before stream destruction fixes it: pinned-lifetime and fresh full-model injected-delay cohorts each pass5/5.
**verified** No external baseline is rerun. New communication constants and ten-scope top-4 re-selection are in progress; final coordinates continue to come from the solver.
