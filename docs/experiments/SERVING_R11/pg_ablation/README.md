# Fixed-geometry PG control, 2026-09-26

The two cells use the same R10-selected GEMM geometry, grid, kappa,
attention coordinates, prefill binary, L2 mode, and calibrated synchronization.
Only the decode transport changes: off, L2 prefetch, or 8 KiB PG-1 pages.
Each arm completed one warmup and three timed 1024-token requests; timed
tokens were identical within each arm. The predeclared guard used idle power
21.81 W plus a 30 W margin and checked GPU ownership before and after runs.
Raw guard decisions, per-run tokens, and step times are in `raw.tar.xz`
(SHA256 `32b64d322f9940af792ddfc0665460189bad123b90b63822425bfd41310af15d`).

| Cell | off E2E | PG-0 E2E | PG-1 E2E |
|---|---:|---:|---:|
| Llama B=1 | 3.3439 s | 3.6652 s | 6.3403 s |
| Qwen3 B=16 | 6.0990 s | 6.4943 s | 10.9151 s |

PG-1 is 1.90× and 1.79× slower than off in these fixed-geometry cells.
This is a performance failure of the current paged transport, not a protocol
failure. The realized-chain attribution in `../page_diagnostics/pg_chain_classes.tsv`
places most Llama B=1 wall time in gate/up, down, and lm_head; it does not yet
separate page-handshake overhead from weight-copy throughput. The queued
8/16 KiB fixed-geometry control tests that distinction.
