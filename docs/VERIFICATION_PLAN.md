# TileMega pre-construction verification

Evidence labels: ✅ compiled/executed and observed; ⚠️ documented or compiled
but not run; ❌ conjecture.

| Priority | Experiment | State | Evidence |
|---:|---|---|---|
| 1 | V-A cross-block event synchronization | ✅ complete on RTX 4090 | `experiments/V_A/result.md` |
| 2 | V-I four-target cross compilation | pending | — |
| 3 | V-B CUTLASS collective in persistent loop | pending | — |
| 3 | V-D compile-time traits query | pending | — |
| 4 | V-G shared-storage union | pending | — |
| 4 | V-H torch.export coverage | blocked locally: PyTorch absent | — |
| 5 | V-E nvcc compilation baseline | pending | — |
| 5 | V-F symbolic cute dialect behavior | pending | — |
| 6 | V-C cluster DSMEM | pending; SM 9.0 cross-compile only here | — |

Synchronization and race experiments require at least 50 fresh-process runs
per cell, timeout-based hang capture, negative controls, and raw logs. Results
must distinguish observed behavior from architecture documentation and
inference.
