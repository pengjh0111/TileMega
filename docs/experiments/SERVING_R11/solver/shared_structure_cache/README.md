# Shared flow cache across serving structures

The script `../../check_shared_structure_cache.py` compares Level 1 scores
for the same PG-1 configurations with full preparation and incremental
cross-structure cache reuse. All 20 Llama B1 and seven Qwen3 B16 scores are
identical at the precision recorded in `.search.tsv`. The Qwen3 cases cover
lm_head tile-N 32, 64 and 128; the full-control process was deliberately
stopped after the seventh complete result because the first seven took over
six minutes. Its completed `.search.tsv` prefix is preserved here. The
Qwen3 incremental run on those seven cases took 90.79 s. The full-control
wall time for that seven-case prefix is not recorded as a precise comparable
value; `report.json` uses `null` rather than inventing one.

Llama's complete 20-case comparison took 85.18 s without the shared cache
and 51.50 s with it. This tests the exact-score property and a local timing
benefit. It does not establish the ≤600 s per-plan budget, which includes
top-M materialization, compilation and GPU selection.
