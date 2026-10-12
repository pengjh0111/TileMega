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
| `python -m tilemega.serving.download_models --config-only` | Fetch and verify pinned model configs without downloading weights |
| `python -m tilemega.serving.structure_check` | Check pinned model shapes on meta tensors without reading checkpoint weights |
| `python -m tilemega.dnn.diagnose` | Locate per-image excess error in retained intermediate buffers |

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

For a structural check without allocating model weights:

```bash
PYTHONPATH=python python -m tilemega.serving.structure_check \
  --model-root /root/models --points 64:1024,128:2048 --batches 1,16 \
  --out runs/r15/models/structures.json
```

This executes the exporter's modules on meta tensors; it does not certify CUDA
execution, numerical correctness or performance. The receipt says
`model_gate_passed=false` even when all structural checks pass.

`tilemega inspect parametric REPO ARCHIVE OUT [f32|bf16]` requires an explicit
historical input archive containing `ORACLE`, `E2E_GEN`, `P3_GENERALIZATION`,
`COST_MODEL` and `BF16`. It writes only to `OUT`. The compiler's fixed cost-model
regression inputs live in `test/fixtures/cost_model/`.

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

For NAFNet diagnostics, run `tilemega.dnn.check_generated --diagnostics FILE`
under the numerical-test guard on a separate `--reuse none` diagnostic build.
Then run `python -m tilemega.dnn.diagnose --intermediates FILE --bridge BRIDGE
--library SO --out REPORT`. It verifies the artifact and bridge binding before
finding the first observed excess error for each image. Fused or unexposed
nodes cannot be localized by this report. The dataset gate remains separate.

MoE checks accept `--hf-max-tokens 256` to bound the HF and routing reference
prefix while `--steps 1024` retains the full same-binary L1/L2 comparison.
The correctness receipt records these lengths separately. Native bindings and
cached CUDA allocations are released before starting the HF subprocess.

## Portability

`TILEMEGA_TARGET_ARCH` selects sm_80/89/90/100/120. `check-policy` checks device
architecture dispatch conventions. Offline compilation does not certify
execution on a different GPU. `configs/targets/sm_90_h800.json` is an
uncalibrated H800 resource profile; it must be calibrated on hardware before
performance selection. `doctor --hwcheck --config ...` exercises native
transport paths and numerical smoke checks, not a performance optimum.
