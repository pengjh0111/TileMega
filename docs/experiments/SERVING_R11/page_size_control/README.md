# Fixed-geometry 8/16 KiB page control

Llama B1, decode L2, same selected R10 geometry, page transport, prefill, target wait policy and synchronization flags. The generated manifests agree on every field checked by `measure_page_size.py` except the page layout. Both binaries use 101376 B shared memory and 160 threads; 8 KiB has ten pages, 16 KiB has five. The predeclared ownership/idle-power policy is the one in `pg_ablation/measurement_policy.json`. Each arm used one warmup and three timed full 1024-token requests; all timed outputs within each arm matched. Raw logs including per-round guard decisions are archived in `raw.tar.xz` (SHA256 `58a5db07ab307dfaef2120c00b5898b0b7e88ed122f33bcb14b920b3ff596b58`).

8 KiB: 6.329995 s; 16 KiB: 4.751128 s; ratio 0.7506. The 16 KiB result is still 1.421× the separate PG-off fixed-geometry control. This isolates page layout as a large contributor but does not by itself measure the exact cost of a barrier transaction.
