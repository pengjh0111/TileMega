# A6 unified ranking control

✅ Same historical measurements, configurations, occupancy and measured
FP32-partial combine rates; only `--legacy-task-cost` versus
`--unified-task-cost` differs. The new path reads semantic CG, never a
generated CUDA table. `tools/tilemega-costmodel.cpp:400` is that input switch;
`lib/Solver/CostModel.cpp:603` prepares its shared task-price entry.

| dtype | model | configurations | old / new rho (printed prediction ranks) | old / new top1,3,10 |
|---|---|---:|---:|---|
| BF16 | gqa2 | 770 | .9038734673277106 / same | 0,0,0 / same |
| BF16 | mha4 | 462 | .8910704759067422 / same | 0,0,0 / same |
| FP32 | gqa2 | 1077 | .943179364105432 / same | 1,3,6 / same |
| FP32 | mha4 | 1077 | .9421013798584228 / same | 1,3,6 / same |

`verify_unified_rank.py` independently checks all keys/measurements, computes
tie-aware Spearman, and rejects any FP32 decline (no tolerance).
`verification.json` hashes its exact inputs. The C++ tool also records its
full-precision internal ranking rounded to four decimals in `summary.tsv`;
small differences from ranks of six-digit printed predictions are a TSV
precision effect, not a second measurement set.

⚠️ BF16 uses the historical PASS subset: 770 gqa2 / 462 mha4. The 308 mha4
rows had numerical criterion failures, not runtime failures, and their
timings are absent. Thus these are conditional model comparisons, not a
post-partial-repair full BF16 oracle or evidence of BF16 ranking attainment.
No repair of the labels creates missing timings. FP32 remains the full
1077×2 historical regression anchor.

At the oracle's seq=4/past=3, both dtype comparisons mostly shift every
configuration by the same scalar-stage constant: BF16 about +3.17/+6.34 us,
FP32 about +5.03/+10.07 us. Rankings therefore do not improve. GEMM work and
prices remain under their separate exact stage bit gate; none of these
rounded tables replaces it. This small-sequence result does not extrapolate
to attention-heavy contexts.

The tested costmodel executable SHA256 was
`b67a22eb7b06099ad14d57dd8db970b095bc787dbb41e6671ce5b8e54aee625c`.
Run `--unified-task-cost --full-only` with `--register-dir
docs/experiments/COST_MODEL/raw` for FP32; BF16 additionally uses
`--dtype bf16 --screen-dir docs/experiments/ORACLE/raw_bf16
--register-dir docs/experiments/ORACLE/raw_bf16/cost`.
Output directories are `unified_rank/f32` and `unified_rank/bf16`.
Historical controls are `partial_rate_comparison/{f32,bf16}_measured`.
