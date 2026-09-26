# Serving incremental-preparation equivalence

`run_incremental_serving.py` evaluated the same 20 random six-class geometry
vectors twice per model, once with full Level 1 preparation and once with
incremental reuse. Llama decode used B=1; Qwen3 decode used B=16; both used
past 64:1086 and the R10 inflight target. All 40 paired serving evaluations
completed without error. Their raw Level 1 scores and candidate keys are
identical (`report.json`: maximum relative difference 0).

The command JSON, standard output/error, and raw search TSV for each arm are
retained here. This verifies the incremental part of G-6. The R-2/R-3/R-4
pruning equivalence checks are separate.
