# V-E — nvcc compilation baseline

Evidence labels: ✅ measured; ⚠️ extrapolated; ❌ conjecture.

| Phase | Wall time |
|---|---:|
| Full `.cu` to cubin, median of 20 | ✅ 4.657898 s |
| Frontend/template instantiation to PTX | ✅ 4.683487 s |
| Standalone ptxas | ✅ 0.034806 s |
| CUDA device link | ✅ 0.104263 s |

End-to-end and decomposed measurements are separate runs, so their components
are not expected to sum exactly. Template/frontend work dominates this
candidate; ptxas and linking are small.

Four cold sequential compilations took 20.084450 s; four parallel processes
took 5.559738 s, a 3.61× wall-time speedup. No compiler cache was installed, so
cache benefit is ⚠️ unconfirmed; nvcc did not expose a persistent cross-process
template cache in this test.

At the measured median, 17 operators × 10 candidates is ⚠️ 791.843 s (13.20
minutes) serial. Ideal groups of four would be about 3.65 minutes, while the
observed four-job batches imply about 3.94 minutes before scheduling overhead.

## Reproduction

```bash
docs/experiments/V_E/run.sh
cat docs/experiments/V_E/raw/summary.txt
```
