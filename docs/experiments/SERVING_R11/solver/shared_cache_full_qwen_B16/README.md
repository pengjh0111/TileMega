# Qwen3 decode B16 full-domain PG-1 CPU search with shared structure cache

The same 355 legal configurations and the same best key/Level 1 score as the
earlier run in `../full_stage2_qwen_B16/` completed in about 494 s rather
than 688 s. The elapsed value uses the search TSV's filesystem birth time
and the completed timing TSV's modification time, rounded to seconds.

`piece_pricing_and_release` fell from 418 s to 266 s. This is a CPU-only
search: top-M materialization, compilation, GPU selection, and full-request
testing are excluded, so it does **not** pass the ≤600 s per-plan gate.
The command, source binary fingerprint, score and files are in `result.json`.
`phases.tar.xz` preserves the large per-candidate phase trace.
