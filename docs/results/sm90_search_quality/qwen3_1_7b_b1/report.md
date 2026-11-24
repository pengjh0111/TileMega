# H800 native search selection-quality audit

For Qwen3-1.7B at batch 1 on one H800, the frozen native search evaluated 180 configurations, retained 10 in its pooled shortlist, and passed 4 replay finalists to compilation and capacity verification. Using the same fixed-coordinate implementation and two independent request-wall measurements per valid program, the three sets contained 180, 10, and 4 valid distinct implementations. Their median TPOTs were 1.7808, 1.8067, and 1.8046 ms, respectively. The pooled and replay best-retained gaps were 5.410% and 6.419%, and the pre-measurement predicted selection gap was 6.762% relative to the best measured among evaluated candidates. The equal-budget pooled and replay top-1 rules selected the same executable. Their retained sets were also identical, so this case does not demonstrate an additional retention benefit from replay over equal-budget pooled ranking. Replay uses the implemented fixed-owner skeleton; these results do not isolate a free ownership search. 2 other implementation(s) had two-process ranges overlapping that of the observed best; such close results should be treated as approximately equivalent at this measurement resolution. Independent confirmation measurements are reported separately and do not change the two-process audit ranking.

Frozen HEAD: `7425a5242580af16a7ea43be742e313a6eedb351`. GPU UUID: `GPU-ac78f16f-955d-c08b-e6fe-4e4554c8954d`.
Checkpoint: shared deterministic synthetic seed-zero BF16 weights, CPU-generated once; this is not a pretrained-model accuracy claim.
Protocol: prompt/L1 prefill, L1 persistent decode, ISL64/OSL1024. Each fresh process uses one warmup and three timed requests at N1 and N1024. TPOT is the difference of request-wall medians divided by 1023; candidate score is the median of two process TPOTs. Order is seeded shuffle then reverse.
One 256-position C1 and one same-binary single-step/persistent C2 screening occur outside request timing on the first audit process. All failed/interfered attempts are retained. Namespace-limited GPU ownership is provisional, as recorded by the existing guard.
Deduplication conservatively requires identical generated CUDA, full task/event IR and execution manifest (excluding file-location and source-coordinate identity fields). Original membership is never replaced with calibrated-child membership.
The historical R6 receipt supplied authentic seed/export input identities only; search and all reported candidate latencies use the frozen current implementation. No historical TPOT is mixed into the table. No task-body, ranking, threshold, calibration constant, or core search code was changed.
Calibration provenance: `calibration_body_binding.json` records existing target constants whose recorded header hashes predate some current bodies. Constants were intentionally retained, as requested. Thus this audits the actual existing scoring model, without a new calibration.

| Stage | Raw N | Valid configs / unique | Median ms | Best ms | Best-retained gap |
|---|---:|---:|---:|---:|---:|
| E | 180 | 180 / 180 | 1.780833 | 1.692079 | 0.000% |
| pooled shortlist Q | 10 | 10 / 10 | 1.806748 | 1.783625 | 5.410% |
| replay finalists S | 4 | 4 / 4 | 1.804638 | 1.800697 | 6.419% |

| Selection / control | Original q | TPOT ms | Gap to observed E best |
|---|---|---:|---:|
| predicted_winner | q079 | 1.806499 | 6.762% |
| S_pool_top1 | q079 | 1.806499 | 6.762% |
| E_observed_best | q135 | 1.692079 | 0.000% |
| S_pool_best_retained | q103 | 1.800697 | 6.419% |
| S_replay_best_retained | q103 | 1.800697 | 6.419% |

Pooled and replay top-1 map to the same implementation: True. The counterfactual uses no extra hardware measurements.

| Independent confirmation | TPOT ms | Status |
|---|---:|---|
| q079 | 1.805211 | valid |
| q135 | 1.694660 | valid |
| q000 | 1.690512 | valid |
| q013 | 1.697774 | valid |

Selection and the S_pool/S_replay control: `selection_summary.csv`. Per-process results, including rejected attempts: `timings.csv`. Independent confirmations: `confirmation.csv`.

Combined pooled search plus shortlist replay: 123.899 s. The existing receipt does not separately time these phases; no split is inferred. Finalist compilation/occupancy and extra-E audit costs are retained in `costs.json` and raw command/phase receipts.
Finalist materialization/occupancy phase wall time: 190.6227524280548 s. Sum of finalist compile-command durations: 225.710 s; guarded occupancy duration: 31.991 s. The work sum and parallel phase wall time must not be added together. Compiler setup and one-time CPU checkpoint generation were not independently timed and are outside these phase figures.

Candidate compilation / screening status counts:
{"compiled": 180}
{"valid": 180}

Reproduce from this directory: `python3 freeze.py` after the recorded native search; `python3 audit.py finalists`; `python3 audit.py all`; `python3 measure.py audit`; `python3 measure.py confirm`; run `report.py` with NumPy and matplotlib available. Existing outputs are retained; use a fresh experiment directory for a new audit.

The initial audit stopped during a candidate attempt. The same physical H800 resumed the recorded sequence, retaining accepted processes and retrying the candidate whose guard receipt was unfinished. Attempt timestamps and statuses are in `timings.csv`.
