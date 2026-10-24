# Offline final matrix and frozen baseline ratios

`scripts/summarize_sm90r2_final.py` performs no GPU work. It revalidates each
final pair's artifact contract and re-runs its offline collector against bound
raw measurements, token tables, physical identity and guard receipts. It then
compares all 25 dense cells with the unchanged
`docs/results/R16_final_baselines.csv` (SHA256
`d3bded2e5acbb9302ccba6a42a7274801af1888d0cc3c3661a6aa1d74299f253`).
This tool does not supply missing final artifacts or pass E-3 on their behalf.

The matrix manifest has `schema: 1`, `kind: "sm90r2_final_matrix"`, a
`frozen_baseline` FILE, and exactly 25 entries in `cells`. Each FILE is
`{"path":"...","sha256":"..."}`, absolute or relative to the matrix manifest.
Each cell binds these five FILEs:

- `pair_manifest`: the final pair manifest described in [FINAL_PAIR_MANIFEST.md](FINAL_PAIR_MANIFEST.md).
- `preparation`: its `preparation.json`.
- `queue`: its `queue/queue_final_pair.json`.
- `state`: an immutable snapshot of the completed scheduler state.
- `comparison`: the result previously emitted by the pair collector.

Run from the final artifact source HEAD, with all artifacts and raw receipts
available. Output must be a new directory inside `runs/sm90r2/`:

```text
python scripts/summarize_sm90r2_final.py --manifest MATRIX --out runs/sm90r2/FINAL_SUMMARY
```

The three compact tables are `SM90R2_final.csv`, `SM90R2_ratios.csv`, and
`SM90R2_ratio_summary.csv`. Latency ratios are baseline/TileMega; throughput
ratios are TileMega/baseline. Values above one favor TileMega. Geometric means
and minima use all 25 cells. The fastest-engine comparator chooses minimum
frozen TPOT among vLLM, SGLang and TensorRT-LLM and uses that same engine for all
four metrics; torch.compile is also reported separately.

Cohort metrics retain their medians of fresh-process metrics: TPOT is not
recomputed from the separately aggregated TTFT and E2E medians. A global
reference-performance claim requires all 25 paired decisions to be `faster`;
near-threshold or indistinguishable cells remain unresolved. Ownership remains
provisional when the source guard receipts say so. Final C-1/C-2 remain
`not_performed`, and historical baseline clock, checkpoint and session
differences receive no invented correction. No final matrix has yet been run.
