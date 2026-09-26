# Fixed-geometry publication-protocol control

The page-ring protocol and model/geometry/placement are identical within each pair. `off` uses the previously generated page binary without `EVENT_SOLO`, `EVENT_RED_PUBLISH`, or `BARRIER_V2`; `combo` enables all three. Both modes passed 50 fresh-process token comparisons per model and endpoint batch before timing. Predeclared guard policy: idle 22.92 W + 30 W; raw per-round decisions are in `raw.tar.xz` (SHA256 `fb03cb5e74815591b51421aebb91cbbf4e2381c0518d809b7e07798c99d62c69`). Each binary had one warmup and three timed full 1024-token requests.

| Cell | Off E2E | Combo E2E | Combo / off |
|---|---:|---:|---:|
| llama B1 | 6.336355 s | 6.316970 s | 0.9969 |
| qwen3 B16 | 10.917311 s | 10.908318 s | 0.9992 |

The measured change is below 0.4% in either cell. This experiment gives no evidence that the publication combination resolves the larger PG-1 regression. It is a single controlled session, not a paired confidence interval.
