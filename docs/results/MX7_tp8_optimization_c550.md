# MX-7 TP8 final: Qwen3-1.7B on C550

**verified** BF16 greedy, ISL/OSL64/1024, eight physical GPUs0–7. Only B16 changes; B1/B2/B4/B8 and all external baselines reuse their accepted original measurements.
**stated** the user explicitly requested stopping further validation and publishing the collected performance. No more tests are scheduled.
**verified** performance is complete; the new selected B16 passes HF C-1, five fresh C-2 processes and resource qualification. New delay injection is **partial:2/5**, remaining work cancelled by the user. This is not a claim of complete synchronization acceptance.

## Identity and environment

- **verified** branch `tilemega-metax`; optimization start `2a2b086d7`; measured code HEAD `2efd59ae6`; publication commit follows.
- **verified** MACA3.8.2.6, cucc/mxcc1.0.0(890c70f3aa), driver3.8.1; Torch2.10.0+metax3.8.2.2, Transformers5.12.0 (existing onboarding evidence).
- **verified** MX7 prompt SHA256 `5caed7edd4477b8fb63d33bed9254bbc4e33d92041eb5969113fefeb805d7cee`.
- **verified** original TP8 table SHA256 `c82b865874abcf391313d512b858f08cdb9dd4b32744ceb981abd030c8f44689` remains the historical source.
- **inferred** the user continuation after the concrete TP8-only attentionV3 proposal authorizes this bounded trial; the interpretation is recorded in `runs/mx7/tp8_opt/ATTENTION_V3_CONTINUATION.json`.
- **verified** eight TP1/MoE frozen generated sources and per-function instruction counts pass8/8 (`tp8_attention_v3_optin` in the frozen ledger). TP2/TP4 still require attentionV2; protected MoE code/data remain unchanged.

## Current complete comparison

**verified** TPOT in ms. SGLang is the fastest frozen baseline in all five cells. TileMega uses two fresh processes, warm1/timed2, no trace/DIAG/watchdog/delay; TPOT=(wall E2E−CUDA TTFT)/1023.
External baselines retain the original N1024−N1 TPOT protocol and all30 successful fresh processes. No baseline is remeasured.

| B | vLLM | SGLang | torch.compile | TileMega | fastest/TM | previous TM |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.942771 | 2.397910 | 17.341626 | 1.821471 | 1.316x | 1.821471 |
| 2 | 3.217482 | 2.723171 | 18.446115 | 1.882461 | 1.447x | 1.882461 |
| 4 | 3.558092 | 2.778020 | 19.267516 | 1.983761 | 1.400x | 1.983761 |
| 8 | 3.390532 | 2.824548 | 19.619999 | 2.199825 | 1.284x | 2.199825 |
| 16 | 3.555431 | 3.000452 | 18.769523 | 2.676431 | 1.121x | 2.994476 |

| B | TTFT ms | E2E ms | tokens/s | attention version/split | source |
|---:|---:|---:|---:|---|---|
| 1 | 108.280 | 1971.644 | 519.4 | 2/1 | reused_original |
| 2 | 116.013 | 2041.771 | 1003.1 | 2/1 | reused_original |
| 4 | 121.467 | 2150.855 | 1904.4 | 2/1 | reused_original |
| 8 | 133.725 | 2384.146 | 3436.0 | 2/1 | reused_original |
| 16 | 161.843 | 2899.832 | 5650.0 | 3/4 | paired_candidate |

**verified** TTFT includes prompt-mode head evaluations. TPOT speedup is not an E2E speedup; E2E and throughput are reported above.
Current20-cell table: `MX7_tp8_final_c550.csv`; exact historical table: `MX7_tp8_initial_c550.csv`; native provenance/details: `MX7_tp8_optimization_c550.csv`.

## B16 optimization and selection

**verified** the opt-in admits existing dense attentionV3 for TP8 only. No AR/LL8 peer primitive, numerical contract, barrier, stream adapter, sharding or MoE body changes.
**verified**45 attention coordinates compile/fit,225 length keys are measured with three fresh processes/key. Existing regular/communication costs are reused; one competitive noisy key receives two supplements and remaining noisy keys use p90.
**verified** merged eligible cost SHA256 `708867761fd24e6e1839c2e41b35c5425358d425cc35178874f00f8690493bdd`. Four source-bound k-best candidates pass C-1/C-2 before ordinary timing; candidate4 wins by the1% rule.

| global candidate | predicted ms | measured ms | pred/meas | selected |
|---:|---:|---:|---:|---|
| 1 | 2.666191 | 2.722362 | 0.9794 | False |
| 2 | 2.666471 | 2.705185 | 0.9857 | False |
| 3 | 2.667481 | 2.701925 | 0.9873 | False |
| 4 | 2.667761 | 2.677395 | 0.9964 | True |

**verified** same-group A1/B1/A2/B2: original=2.993403/2.994707ms; optimized=2.676353/2.676510ms.
Original median=2.994055ms, optimized=2.676431ms; TPOT reduction=10.608%. Selection threshold=1.00%.
**verified** the selected rebuild also measures2.674489ms in a separate two-process cohort. The current table uses the completed paired cohort, not the single best observation.

B16 geometry is TN/WN/D/Pw/partition:

| class | geometry |
|---|---|
| qkv | 16/1/1/1/kTileRange |
| o | 32/1/1/1/kTileRange |
| gate_up | 32/1/1/1/kTileRange |
| down | 32/1/2/2/kTileRange |
| head | 16/8/3/16/kTileRange |
| attention | V3, split4, prefetch0 |

## Correctness, resources and stopped work

- **verified** all four B16 candidates:4/4 HF C-1 and8/8 fresh1024-token C-2 processes pass.
- **verified** selected rebuilt B16: HF C-1 passes (max gap0.15625),5/5 fresh1024-token C-2 pass, exact rank tokens in checks and timed requests.
- **verified** selected B16: MTreg225, stack0, static0, LDS32864B, actual resident1 on every rank. Source, compiler, binary, pin and metrics identities remain in the raw receipts.
- **verified** new selected delayed B16:2/5 fresh processes complete with exact ordinary-reference tokens; one in-flight process and remaining tests are cancelled at the user request. No full new delay/synchronization pass is claimed.
- **stated** the user waives further validation for this publication. Numerical criteria are unchanged; incomplete checks are not converted to passes.
- **verified** the original B1/B2/B4/B8 accepted results are retained without any additional GPU validation after the B16-only direction.
- **verified**21 focused CPU tests passed before the user stopped validation; no tests were launched for publication.
- **verified** extra B16 depth candidates are CPU-prepared only. Their GPU screen/qualification, the remaining delay starts and any extra trace are not run.

## Attribution and deviations

**verified** original B16 diagnostic rank0 means of layer medians (us): qkv12.29, attention37.36, o+AR21.05, gate_up15.60, down+AR22.86; head116.33us/step. Trace is diagnostic only.
**inferred** the original attention bottleneck motivates V3/split4. The accepted change also increases down/head depths; there is no post-change stage trace because further work is stopped.
**verified** new B16 pred/meas=0.9968, within0.9–1.1. The accepted five-cell comparison exceeds1.10x in every cell under the standard median protocol.
**inferred** depth-only cached screening previously improved B16 about1.4%, but that screen is timing-ineligible and is not used as final evidence.

Raw successful/failed/interrupted attempts remain in `runs/mx7/tp8_opt/`. Publication receipt: `attention/USER_FINAL_PERFORMANCE.json`; stop record: `attention/USER_STOP_VALIDATION.json`.

## GPU accounting

**inferred** optimization-phase reserved child-job time (first running to first after sample, interrupted jobs through recorded stop/last sample), excluding lock/preflight waiting, including initialization/cooldown/cleanup; not hardware busy time.
Per-card hours: 0=1.626, 1=1.624, 2=1.623, 3=1.622, 4=1.622, 5=1.621, 6=1.621, 7=1.631; total=12.990 card-h. Earlier broad optimization precedes the later B16-only instruction.
