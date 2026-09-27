# OPS-2 one-click smoke: Llama B=1

This is an OPS-2 tool acceptance sample, **not** the solver-selected EV-2
matrix. It used source `96129ccae`, `configs/e2e/llama_b1.json`, BF16 real
weights, 64 prompt tokens, 1,024 generated tokens, `handoff=off`, and no
vLLM arm. The first `python -m tilemega run` produced both plans, performed
three timed full requests and three TTFT runs, checked L1/L2 tokens, and ran
the HF teacher-forced check. The second invocation used the same config and
cache with a new output directory.

| Item | First run | Repeated run |
|---|---:|---:|
| Calibration | only changed `task_bodies` section measured | all sections hit |
| Export | prefill and decode hit | prefill and decode hit |
| Plan | prefill and decode miss | prefill and decode hit |
| Prefill plan solve | 867.372 s; 600 s budget failed | skipped |
| Decode plan solve | 776.872 s; 600 s budget failed | skipped |
| E2E median | 3.8909 s | 3.8896 s |
| `E2E / ΣT_floor` | 1.498 | 1.498 |
| L1/L2 token mismatch | 0 / 1,024 | 0 / 1,024 |
| HF C-1 | 1,024 / 1,024 at gap ≤ 0.5; max 0 | same result |

Both selected `.so` SASS audits report `fp64_total = 0`.

The first prefill solve logged 3,471 Level 1 evaluations; cumulative
`piece_pricing_and_release` was 262.289 s and three megakernel compiles
totaled 266.509 s. The decode solve logged 694 Level 1 evaluations;
`prepare_relations` took 88.969 s, `piece_pricing_and_release` 70.179 s,
`materialize` 56.641 s, and three compiles totaled 240.069 s. Cumulative
phase times are not wall time. The repeated run skipped both solves, but
still performed bench and checks. The plan budget failure is real; it is not
waived by the successful cache hit.

`first/` and `repeat/` retain the reports, command list, cache decisions,
guard decisions, mode/HF checks, and all 1,023 decode step times. `plans/`
retains solver phase totals, top-3 rankings, SASS audit, floor, manifest,
recorded plan keys and output hashes. The binaries and full plans remain in
`/root/.cache/tilemega/plans/` under the recorded keys. The subsequent
`DramFloor` binding cache change is commit `8b4f90fb5`; its timing must be
measured separately and cannot be attributed to this run.
