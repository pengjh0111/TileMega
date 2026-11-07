# MX-7: Qwen3-1.7B TP2 / TP4 / TP8 on C550

Evidence: **verified** = executed or source-checked here; **stated** = specification/remote report; **inferred** = reasoning.
BF16 greedy, ISL/OSL=64/1024. Raw successes, failures, generated sources and binaries remain under `runs/mx7/`.

## Scope and identity

- **verified** `tilemega-metax`, MX-7 start `6cab45b9604e306fef1916863fe16b09d895c32f`, SM90 audit start `a7c4a7968a787a5bf5a36fbbfc6532ae1de7a176`.
- **verified** TP2/TP4 aggregation HEAD `a1f86508a8d1d171dac1870972b2b0125f88ebf2`;115 commits since MX-7 start; historical publication follows.
- **verified** prompt SHA256 `5caed7edd4477b8fb63d33bed9254bbc4e33d92041eb5969113fefeb805d7cee`.
- **verified** resume1 SHA256 `f67014253a3cca7234429ff3671e293c0104fb04b68fde1bb9dc52b518c0b284`.
- **stated** later user authorization: all GPUs; SDK-owned ExternalStream adapter; baselines measured once; compare SM90 and optimize TP bodies.
- **verified** MoE protected files/final CSV remain unchanged; eight frozen TP1/MoE plans retain identical generated CU and function instruction counts.

## Environment and execution (10-line onboarding summary)

1. **verified** MACA3.8.2.6, cucc/mxcc1.0.0(890c70f3aa), driver3.8.1.
2. **verified** Torch2.10.0+metax3.8.2.2, Transformers5.12.0; full versions/topology in `runs/mx7/sync/onboarding.md`.
3. **verified** vLLM0.25.0+empty; SGLang0.5.14+detached.c432db2fb.maca3.8.2.122.
4. **verified** NCCL/Gloo available; NCCL used only by torch.compile baseline; TileMega uses MACA P2P.
5. **verified** C0 selected LL8, fine allocation, volatile AS1 scalar64 store and `__ldcg`64 load; no runtime switches.
6. **verified** original pair/all-to-all P1024/P262144: each5/5 fresh processes ×2000epochs ×104CTAs, no tears/timeouts.
7. **verified** final TP2 group{1,3}; final TP4 group{1,2,3,5}; guard admission/locks unchanged.
8. **inferred** each rank: qkv→attention→o+AR→gate_up→down+AR per layer, then head LL argmax exchange.
9. **verified** ascending-rank FP32 sum of BF16 partials, BF16 sum then residual; epoch=past+1; two asynchronous range launches/rank.
10. **verified** prompt mode starts at past0, exchanges every step, suppresses early token writes; one SDK nonblocking stream/rank.

## SM90 comparison and accepted optimization

- **verified** current complete SM90 results/source: `origin/tilemega@213a3884a4e14629ca99f63755d501c65a7d8f53`; no merge or H800 remeasurement.
- **verified** sharding, persistent launch pipeline, AR placement and BF16 arithmetic match. No missing collective or per-token host loop.
- **verified** all10 SM90 pins enable KV-group qkv handoff and disable cross-stage weight streaming; C550 retains V2 grid boundaries. Finer handoff remains a future design, outside this MX-7 protocol.
- **verified** the original MX-7§6.4 mapping gave each lane four LL8 packets at32-byte inter-lane strides. SM90 assigns one adjacent column pair/lane.
- **verified** shared pair-lane mapping preserves wire layout, rank order, rounding, geometry and synchronization; production peer primitives stay unchanged.
- **verified** restoring two-pass sends or replacing only peer primitives did not improve performance; those candidates stay off.
- **verified** same-geometry TP4 B16 A/B/A/B control:12.526515→3.081392ms(4.065×), both arms2 fresh warm1/timed2 processes.
  This control is separate from the solver-selected final matrix below. Audit and microbenchmarks: `MX7_tp_sm90_comparison_c550.md`.
- **verified** all1840 single-rank stage rows are reused;90 communication constants are newly measured,3 processes/key plus competitive-noise supplements.
- **verified** eligible cost SHA256 `2927cf91afbc8589f8706d50699bc837fa7a6898978ed0fefed6fcab2c906f6e`. All10 cells rerun k-best/top4 and the1% E2E selection rule.

## Frozen baselines and final performance

**verified** all30 external cells pass: vLLM/SGLang/torch.compile × TP2/TP4 × B1/2/4/8/16.
**stated** original baseline samples are reused across GPU groups; no external baseline is repeated for this optimization.
**verified** baseline TPOT=(E2E_N1024−E2E_N1)/1023; TileMega TPOT=(wall E2E−CUDA TTFT)/1023.
Both use two fresh processes/cell, warm1+timed2. TTFT includes prompt mode; tokens/s counts B×1024 outputs.

| TP | B | vLLM ms | SGLang ms | torch.compile ms | TileMega ms | fastest/TM | old TM ms | old/new |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 3.912 | 3.084 | 14.034 | 2.253 | 1.369 | 2.518 | 1.118 |
| 2 | 2 | 4.002 | 3.439 | 16.259 | 2.292 | 1.501 | 2.802 | 1.223 |
| 2 | 4 | 4.032 | 3.487 | 16.603 | 2.673 | 1.305 | 3.655 | 1.368 |
| 2 | 8 | 4.302 | 3.777 | 21.627 | 2.998 | 1.260 | 4.806 | 1.603 |
| 2 | 16 | 4.625 | 4.216 | 32.919 | 3.561 | 1.184 | 7.246 | 2.035 |
| 4 | 1 | 3.115 | 2.530 | 16.607 | 1.836 | 1.378 | 2.459 | 1.340 |
| 4 | 2 | 3.852 | 3.042 | 15.453 | 1.880 | 1.618 | 2.776 | 1.477 |
| 4 | 4 | 3.694 | 3.083 | 15.342 | 1.963 | 1.571 | 3.784 | 1.928 |
| 4 | 8 | 3.763 | 3.177 | 16.940 | 2.420 | 1.313 | 6.210 | 2.566 |
| 4 | 16 | 3.938 | 3.366 | 22.851 | 3.072 | 1.096 | 12.506 | 4.071 |

Baseline efficiency uses each engine's frozen TP1 reference: TP1_TPOT/(TP×TP_TPOT).

| TP | B | vLLM efficiency | SGLang efficiency | torch.compile efficiency |
|---:|---:|---:|---:|---:|
| 2 | 1 | 56.4% | 61.6% | 29.4% |
| 2 | 2 | 60.3% | 62.0% | 34.4% |
| 2 | 4 | 60.8% | 62.9% | 47.9% |
| 2 | 8 | 62.4% | 66.4% | 59.4% |
| 2 | 16 | 62.7% | 64.8% | 67.8% |
| 4 | 1 | 35.4% | 37.6% | 12.4% |
| 4 | 2 | 31.3% | 35.1% | 18.1% |
| 4 | 4 | 33.2% | 35.6% | 25.9% |
| 4 | 8 | 35.7% | 39.5% | 37.9% |
| 4 | 16 | 36.8% | 40.6% | 48.8% |

| TP | B | TTFT ms | E2E ms | tokens/s | TP1/TM | scaling efficiency |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 139.703 | 2444.378 | 418.920 | 1.451 | 72.5% |
| 2 | 2 | 141.585 | 2485.993 | 823.816 | 1.575 | 78.7% |
| 2 | 4 | 195.256 | 2929.354 | 1398.261 | 1.467 | 73.3% |
| 2 | 8 | 206.933 | 3273.885 | 2502.226 | 1.420 | 71.0% |
| 2 | 16 | 228.941 | 3872.150 | 4231.241 | 1.455 | 72.8% |
| 4 | 1 | 113.542 | 1991.635 | 514.151 | 1.780 | 44.5% |
| 4 | 2 | 115.299 | 2038.298 | 1004.760 | 1.920 | 48.0% |
| 4 | 4 | 120.017 | 2128.332 | 1924.512 | 1.997 | 49.9% |
| 4 | 8 | 177.585 | 2653.293 | 3087.485 | 1.759 | 44.0% |
| 4 | 16 | 281.397 | 3423.845 | 4785.264 | 1.687 | 42.2% |

Efficiency=TP1_frozen_TPOT/(TP×TP_TPOT). Each engine ratio, process result, binary/pin/cost SHA is retained in `MX7_tp_final_c550.csv`.
Old published pins are preserved under `configs/pins/mx7/history/vector8/`; old CSV rows and original artifacts are retained.

Historical old/new ratios retain their recorded GPU groups; the separate B16 P-PAIR is the same-group causal comparison.

## Selected geometries

Coordinates below are TN/WN/D/Pw/partition; TR=TileRange, SK=StreamK. Attention remains V2.

| TP | B | qkv | o | gate_up | down | head | attn pf/b_min |
|---:|---:|---|---|---|---|---|---|
| 2 | 1 | 32/1/2/2/TR | 32/1/2/2/TR | 64/1/2/2/TR | 32/1/2/2/TR | 32/8/4/8/TR | 0/1 |
| 2 | 2 | 32/1/2/2/TR | 32/1/2/2/TR | 64/1/2/2/TR | 32/1/2/2/TR | 32/8/2/8/TR | 0/1 |
| 2 | 4 | 32/1/2/2/TR | 32/1/2/2/TR | 64/1/2/2/TR | 32/1/2/2/TR | 64/8/2/4/TR | 0/4 |
| 2 | 8 | 32/1/2/2/TR | 32/1/1/1/TR | 64/1/1/1/TR | 32/1/2/2/TR | 64/8/1/4/TR | 0/4 |
| 2 | 16 | 32/1/1/1/TR | 32/1/1/1/TR | 64/1/1/1/TR | 32/1/1/1/TR | 64/8/1/4/TR | 1/4 |
| 4 | 1 | 16/1/4/4/TR | 32/1/2/2/TR | 32/1/2/2/TR | 32/1/2/2/TR | 32/4/2/8/TR | 0/1 |
| 4 | 2 | 16/1/4/4/TR | 32/1/2/2/TR | 32/1/2/2/TR | 32/1/2/2/TR | 32/8/3/8/TR | 0/1 |
| 4 | 4 | 16/1/2/2/TR | 32/1/2/2/TR | 32/1/2/2/TR | 32/1/2/2/TR | 32/4/2/8/TR | 0/1 |
| 4 | 8 | 16/1/2/2/TR | 32/1/2/2/TR | 32/1/2/2/TR | 32/1/1/1/TR | 32/8/3/8/TR | 0/4 |
| 4 | 16 | 16/1/1/1/TR | 32/1/2/2/TR | 32/1/1/1/TR | 32/1/1/1/TR | 32/8/3/8/TR | 1/8 |

## Correctness, synchronization and resources

- **verified** pair-lane AR units: TP2/TP4 × B1/B16, each5/5 fresh processes; nine legal geometries/process against ordered-BF16 reference.
- **verified** original head exchange units pass5/5 for rank-local/cross-rank/all-equal ties; head code unchanged by this optimization.
- **verified** all40 candidates pass512-position HF C-1 and2 fresh1024-token C-2 processes.
- **verified** all10 selected rebuilds pass512-position HF C-1 and5 fresh1024-token C-2 processes; all ranks agree in every run.
- **verified** selected pair-lane injected-delay checks: TP2/TP4 × B1/B16, each5/5 new1024-token processes exactly match ordinary selected outputs.
- **verified** MTreg≤255, stack0, static0, all-rank actual resident1, dynamic LDS≤64KiB. Rejections remain in the resource ledger.
- **verified** eight TP1/MoE frozen generated sources and per-function instruction counts pass8/8 after the shared-body edit.
- **verified**50 targeted CPU tests pass after the SDK lifetime correction; detailed raw evidence remains under `runs/mx7/sm90_comparison/`.

## Final TP4 stage trace (rank0, µs)

**verified** one diagnostic process/B;1023 decode steps ×104CTAs/rank. o/down include AR; head includes exchange.
Trace durations use max CTA release minus previous-stage release; these are diagnostic, not E2E acceptance measurements.

| B | layer | qkv | attention | o+AR | gate_up | down+AR |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 7.39 | 25.37 | 9.04 | 13.94 | 11.30 |
| 1 | 1 | 7.54 | 25.23 | 8.90 | 13.93 | 11.32 |
| 1 | 2 | 7.42 | 25.24 | 8.85 | 13.90 | 11.28 |
| 1 | 3 | 7.40 | 25.40 | 8.94 | 13.92 | 11.35 |
| 1 | 4 | 7.41 | 25.27 | 9.00 | 13.90 | 11.36 |
| 1 | 5 | 7.40 | 25.29 | 8.93 | 13.93 | 11.27 |
| 1 | 6 | 7.41 | 25.26 | 8.91 | 13.94 | 11.33 |
| 1 | 7 | 7.45 | 25.33 | 8.92 | 13.92 | 11.29 |
| 1 | 8 | 7.43 | 25.29 | 8.93 | 13.93 | 11.33 |
| 1 | 9 | 7.42 | 25.22 | 8.90 | 13.95 | 11.34 |
| 1 | 10 | 7.43 | 25.15 | 8.89 | 13.90 | 11.35 |
| 1 | 11 | 7.43 | 25.22 | 8.93 | 13.95 | 11.35 |
| 1 | 12 | 7.40 | 25.25 | 8.88 | 13.93 | 11.40 |
| 1 | 13 | 7.44 | 25.23 | 8.87 | 13.94 | 11.35 |
| 1 | 14 | 7.42 | 25.27 | 8.92 | 13.96 | 11.36 |
| 1 | 15 | 7.42 | 25.28 | 8.84 | 13.91 | 11.30 |
| 1 | 16 | 7.45 | 25.29 | 8.94 | 13.97 | 11.30 |
| 1 | 17 | 7.43 | 25.33 | 8.96 | 13.94 | 11.33 |
| 1 | 18 | 7.41 | 25.32 | 8.85 | 13.93 | 11.34 |
| 1 | 19 | 7.40 | 25.26 | 8.93 | 13.98 | 11.34 |
| 1 | 20 | 7.42 | 25.22 | 9.01 | 13.96 | 11.34 |
| 1 | 21 | 7.42 | 25.28 | 8.89 | 13.98 | 11.28 |
| 1 | 22 | 7.41 | 25.24 | 8.95 | 13.94 | 11.35 |
| 1 | 23 | 7.43 | 25.27 | 8.87 | 13.99 | 11.30 |
| 1 | 24 | 7.41 | 25.31 | 8.88 | 13.93 | 11.29 |
| 1 | 25 | 7.40 | 25.31 | 8.94 | 14.05 | 11.35 |
| 1 | 26 | 7.44 | 25.31 | 8.97 | 14.02 | 11.29 |
| 1 | 27 | 7.42 | 25.24 | 8.91 | 13.95 | 17.18 |
| 1 | head | 108.61 | — | — | — | — |

**verified** B1 maximum inter-rank duration difference: o=7.540µs, down=1.880µs.

| 16 | 0 | 14.08 | 42.54 | 15.21 | 16.76 | 19.37 |
| 16 | 1 | 14.51 | 42.55 | 15.21 | 16.75 | 19.41 |
| 16 | 2 | 14.61 | 42.69 | 15.29 | 16.76 | 19.46 |
| 16 | 3 | 14.77 | 42.53 | 15.36 | 16.82 | 19.49 |
| 16 | 4 | 14.80 | 42.81 | 15.57 | 16.82 | 19.58 |
| 16 | 5 | 14.56 | 42.58 | 15.46 | 16.84 | 19.60 |
| 16 | 6 | 14.60 | 42.69 | 15.62 | 16.80 | 19.50 |
| 16 | 7 | 14.64 | 42.48 | 15.39 | 16.82 | 19.52 |
| 16 | 8 | 14.66 | 42.68 | 15.63 | 16.76 | 19.47 |
| 16 | 9 | 14.80 | 42.57 | 15.40 | 16.82 | 19.52 |
| 16 | 10 | 14.80 | 42.66 | 15.58 | 16.76 | 19.52 |
| 16 | 11 | 14.60 | 42.64 | 15.49 | 16.78 | 19.60 |
| 16 | 12 | 14.63 | 42.79 | 15.67 | 16.80 | 19.50 |
| 16 | 13 | 14.68 | 42.60 | 15.41 | 16.86 | 19.54 |
| 16 | 14 | 14.61 | 42.64 | 15.65 | 16.76 | 19.48 |
| 16 | 15 | 14.82 | 42.68 | 15.36 | 16.82 | 19.50 |
| 16 | 16 | 14.76 | 42.63 | 15.52 | 16.81 | 19.49 |
| 16 | 17 | 14.62 | 42.59 | 15.39 | 16.82 | 19.53 |
| 16 | 18 | 14.63 | 42.59 | 15.45 | 16.76 | 19.49 |
| 16 | 19 | 14.57 | 42.55 | 15.49 | 16.81 | 19.60 |
| 16 | 20 | 14.66 | 42.64 | 15.59 | 16.74 | 19.51 |
| 16 | 21 | 14.59 | 42.60 | 15.50 | 16.80 | 19.56 |
| 16 | 22 | 14.68 | 42.57 | 15.61 | 16.76 | 19.53 |
| 16 | 23 | 14.64 | 42.54 | 15.57 | 16.78 | 19.62 |
| 16 | 24 | 14.67 | 42.64 | 15.69 | 16.79 | 19.55 |
| 16 | 25 | 14.62 | 42.56 | 15.54 | 16.77 | 19.68 |
| 16 | 26 | 14.61 | 42.55 | 15.73 | 16.77 | 19.49 |
| 16 | 27 | 14.59 | 42.61 | 15.53 | 16.82 | 26.65 |
| 16 | head | 137.46 | — | — | — | — |

**verified** B16 maximum inter-rank duration difference: o=9.280µs, down=3.940µs.

**verified** TP2 B16 wall minus CUDA prompt/decode events: 0.741µs/decode step; four timed requests, diagnostic only.
**verified** TP4 B16 wall minus CUDA prompt/decode events: 1.412µs/decode step; four timed requests, diagnostic only.

## Autonomous handling and retained failures

- **verified, B2/B4** torch.compile: isolated accelerate1.15.0 plus compiled_autograd=True fixes HF TP gradient-hook compilation; no eager substitution.
- **verified, B1** original AR builds spilled at256MTreg; approved per-item send/receive and constant peer-pointer selection fit resources.
- **verified, B4** unavailable GPUs and external interference required group changes and bounded fresh-process retries; original failures/guard75 records remain.
- **verified, B4** original calibration B8/TN128 LDS underallocation was repaired; old illegal-access evidence remains; no failed sample is promoted.
- **verified, B4** low disk pressure is managed with immutable-input hardlinks and lossless rank-wise trace compression.
- **verified, B4** initial SDK stream full-model check produced exact tokens but exited−6 during pinned-storage cleanup after stream destruction.
  The failed exit is retained. Owned pageable return storage and GPU/pinned release before stream destruction repair the lifetime order;
  four-card retained-pinned-lifetime and fresh full-model delay cohorts each pass5/5 after the fix.
- **verified, B4** optional relaxed-system primitives pass their LL/AR qualification but do not improve stage time; production primitives stay unchanged.
- **stated** all external baselines are frozen once; interrupted group-change repeats are archived and excluded.

- **verified, B4** one push hit remote `commit_refs` failure; after confirming the remote HEAD, normal retry succeeds without force or GPU changes.

## Deviations and limitations

- **verified, C** TP2 B1: 2.253ms; prompt expectation **inferred** 2.35–2.50ms.
- **verified, C** TP2 B16: 3.561ms; prompt expectation **inferred** 3.3–3.4ms.
- **verified, C** TP4 B1: 1.836ms; prompt expectation **inferred** 1.85–2.05ms.
- **verified, C** TP4 B16: 3.072ms; prompt expectation **inferred** 2.5–2.7ms.
- **verified, C** final pred/meas range0.929–0.998; outliers remain in selection CSV; trace locates device-stage costs.
- **stated, C** prompt-mode TTFT includes63 additional head evaluations compared with the separate TP1 prefill path.
- **verified, C** original torch.compile TP4 B1 round spread30.2%; both frozen rounds remain; no baseline repeat is made.
- **verified** the initial optional C0 primitive matrix is not exhaustive after unavailable-GPU failures (`experiments/mx7/probe_matrix.py:1`); chosen-path LL qualification is complete.
- **inferred** scalar volatile-access waits alone did not explain the slowdown; the controlled lane-mapping result establishes the accepted performance improvement.
- **verified** TP4 B16 attention V2 averages42.614µs/layer in the final trace. **inferred** it is a remaining optimization opportunity; V3 stays outside the MX-7§10.1 domain.
- **stated** SM90 reports TP2/TP4 B16=1.5844/1.5283ms on H800; hardware/platform differences remain and this comparison is not a same-device speedup.

## GPU accounting and completion

**inferred** reserved child-job wall time includes initialization/cleanup, excludes lock/preflight waiting, and is not hardware busy time.
Per-card hours: 0=0.080, 1=16.990, 2=8.110, 3=17.017, 4=5.465, 5=13.581, 6=2.681, 7=2.685; total=66.609 card-h.
**verified** new ten-cell solver selection, final E2E/C-1/C-2, four delay cohorts and two final traces are complete.
**verified** all required CSVs and selected pins are present; external baselines and protected MoE paths are unchanged.
**inferred** no unimplemented correctness gate remains; performance targets are expectations, not acceptance gates.

## TP8 final update (user-stopped validation)

**verified** all15 TP8 external baseline cells are frozen. B1/B2/B4/B8 retain original TileMega data; only B16 changes. BF16 greedy, ISL/OSL64/1024; TPOT in ms.
**verified** TP8 optimization code HEAD `2efd59ae6`; data publication follows. All new GPU work has stopped at the user's instruction.

| B | vLLM | SGLang | torch.compile | TileMega | fastest/TM |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.942771 | 2.397910 | 17.341626 | 1.821471 | 1.316x |
| 2 | 3.217482 | 2.723171 | 18.446115 | 1.882461 | 1.447x |
| 4 | 3.558092 | 2.778020 | 19.267516 | 1.983761 | 1.400x |
| 8 | 3.390532 | 2.824548 | 19.619999 | 2.199825 | 1.284x |
| 16 | 3.555431 | 3.000452 | 18.769523 | 2.676431 | 1.121x |

**verified** new B16 C-1,5/5 C-2 and resources pass; new delay checks complete2/5, remaining work explicitly cancelled by the user. No complete new synchronization gate is claimed.
Detailed performance/provenance and the cancellation: [TP8 report](MX7_tp8_optimization_c550.md); [current20-cell table](MX7_tp8_final_c550.csv).
