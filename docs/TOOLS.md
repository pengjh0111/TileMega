# TileMega tools

The default build provides `tilemega`, `tilemega-opt` and `tilemega-unit`.
The Python entry point is `PYTHONPATH=python python -m tilemega`.

| Interface | Purpose |
|---|---|
| `tilemega compile` | Export JSON or CG MLIR to CUDA source or a shared library |
| `tilemega inspect import` | Export JSON to the CG dialect |
| `tilemega-opt` | Parse, verify and transform CG MLIR |
| `tilemega calibrate` | Native hardware calibration |
| `tilemega occupancy` | Compiled resource and occupancy queries |
| `tilemega audit arch` / `sass` | Offline architecture compilation / binary instruction audit |
| `tilemega inspect request-floor` | Request-level DRAM and compute floor |
| `python -m tilemega doctor` | Configuration, checkpoint and toolchain checks |
| `python -m tilemega export` / `build` | Cached export and plan selection |
| `python -m tilemega check` | Model correctness checks |
| `python -m tilemega run` | Configured serving workflow |
| `python -m tilemega dnn` | DNN export, build, forward and correctness commands |
| `python -m tilemega.moe.full` | Full-depth MoE dry construction, memory preflight and checks |

## Compilation and identity

`tilemega compile --options FILE.json` expands an array of argument strings,
preserving order and repeated options. A `.cu` output generates source only;
`.so` also invokes NVCC. `CUDACXX` chooses NVCC. Compiler/source fingerprints,
macros, artifact hashes, implementation and kernel resources are recorded in
`<library>.identity.json`. Analysis rejects incompatible identities; trace
artifacts do not supply end-to-end performance results.

`device.cache_dir` contains calibration targets, exports, plans and artifacts.
Keys include workload, source/calibration fingerprints and compilation options.
The serving execution choice lives in `<library>.serving.json` and `plans.json`;
it does not rewrite the cached manifest. Explicit execution options override
`auto`. Model configurations are under `configs/e2e/`, with pinned model
metadata under `configs/models/`.

## Workloads and measurement

A serving configuration defines `workload.prompt_len`, `max_new_tokens`,
`batch`, and optionally `capacity` or `points: [[ISL, OSL], ...]`. Each sweep
point gets a distinct output directory and workload-sensitive cache keys.
Prompt tables must match the request ISL. Prepare another length with:

```bash
PYTHONPATH=python python -m tilemega.serving.prepare_prompts \
  --model /root/models/llama3_2_1b --family llama3 --prompt-len 128
```

Reusable measurement code is under `python/tilemega/bench/`. A benchmark
configuration uses `schema: "tilemega.bench.v1"`, a `cells` mapping, and a
measurement policy. Each cell specifies `model_path`, `prompt_ids`,
`workload: {"isl": 64, "osl": 1024, "batch": 1}`, and `arms`. A TileMega arm
supplies `label`, `kind: "tm"`, `prefill`, `decode`, `mode`, `decode_loop` and
`prefill_mode`; a vLLM arm supplies `kind: "vllm"` and its Python interpreter.
All paths and binary identities are frozen before enqueueing.

```bash
PYTHONPATH=python python -m tilemega bench --config configs/bench/local.json
PYTHONPATH=python python -m tilemega.bench.status \
  --out runs/bench/scheduler --queue-dir runs/bench/queue
```

Create `local.json` for the built artifacts first; it is not a shipped result.
The scheduler serializes steps, and the guard holds the common
`TILEMEGA_GPU_LOCK`. Occupancy or interference returns 75 and retries; an
interfered round is discarded. Six hours without an idle timing window pauses
timing while non-timing work may continue. Use the same lock for numerical
tests and compiler checks. The monitor emits new failure/completion events;
it does not retry, kill jobs or declare correctness gates passed.

Accepted comparisons rotate arm order and report TPOT, request latency,
output throughput and MBU. Each arm has the same prompt, fixed OSL and EOS
policy. Model correctness is separate from timing; component benchmarks are
diagnostic only. Raw receipts remain under ignored `runs/`.

## Portability

`TILEMEGA_TARGET_ARCH` selects sm_80/89/90/100/120. `check-policy` checks device
architecture dispatch conventions. Offline compilation does not certify
execution on a different GPU. `configs/targets/sm_90_h800.json` is an
uncalibrated H800 resource profile; it must be calibrated on hardware before
performance selection. `doctor --hwcheck --config ...` exercises native
transport paths and numerical smoke checks, not a performance optimum.
