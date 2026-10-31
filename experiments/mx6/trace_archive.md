# Closed trace CSV storage

`scripts/mx_trace_archive.py` archives a closed `mx_stage_trace.csv` using the
raw SHA256 already published in the trace ledger. It verifies a complete gzip
round trip before removing the expanded CSV. Each directory retains an immutable
`.lossless_archive.json` with both hashes and sizes. Engine binaries are separate.

Restore the original path and exact bytes before rerunning an old trace report:

```sh
python scripts/mx_trace_archive.py restore /root/tilemega/runs/mx6/v2/trace_diagnostic_v2/qwen3_30b_a3b_l24/B1/cp1/capture/mx_stage_trace.csv
```

Use `check` instead of `restore` to verify the archive without expanding it.
The tool refuses a changed source, a changed archive, or a different existing
destination. Archive only completed diagnostic files with a published raw hash.

Verified continuation1: 19 historical trace files retain 8,839,676,227 raw bytes
in 1,504,997,104 compressed bytes. All decompressed hashes match, and an actual
original path was restored and checked. Three CPU checks cover multi-block
round trips, wrong source hashes and archive tampering. The index is
`docs/results/MX6_trace_archive_c550.csv`; current trace captures remain expanded.
