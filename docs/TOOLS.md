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

Before adding a recovery queue, validate its dependencies against existing
definitions and receipts:

```bash
PYTHONPATH=python python -m tilemega.bench.queue_check \
  --queue-dir runs/bench/queue --state runs/bench/scheduler/state.json \
  --proposed runs/bench/proposed.json --out runs/bench/dependency_check.json
```

Recovery steps need new names and must depend on the repaired numerical run.
The check rejects failed required ancestors, unknown dependencies and cycles;
`after_any` still permits diagnostics after failure. It never edits receipts.

Accepted comparisons rotate arm order and report TPOT, request latency,
output throughput and MBU. Each arm has the same prompt, fixed OSL and EOS
policy. Model correctness is separate from timing; component benchmarks are
diagnostic only. Raw receipts remain under ignored `runs/`.

Summarize only the scheduler's accepted attempts after all comparison rounds
finish. `--prefix` selects the comparison step names, excluding build/smoke
steps. Missing rounds, changed execution identities, trace builds, rejected
guard attempts and changed prompts are errors:

```bash
PYTHONPATH=python python -m tilemega.bench.audit \
  --state runs/bench/scheduler/state.json --prefix comparison_ \
  --compare reference candidate --out runs/bench/accepted.json
```

The report pairs each TM/vLLM throughput ratio within its round. Configure
`step_bytes` and calibrated `dram_gbps` per cell to obtain MBU; absent inputs
produce an unavailable MBU, not a fabricated bandwidth estimate. An accepted
timing report does not replace model correctness checks or select defaults.

`scripts/check_serving_release.py` freezes a locally built candidate and an
optional reference for release validation. Its recipe names the selected
`plans.json`, model, prompts, workload, calibrated target directory, Python
executables and policy. `freeze` rejects incomplete local safetensors shards
and records binary, manifest, identity and execution-sidecar hashes.
`correctness` runs under the scheduler guard, generates a vLLM reference,
and checks full-request L1/L2/loop equality and teacher-forced HF thresholds
for each TM arm. It skips only the optional HF free-greedy diagnostic.
Reference-generation timing is excluded from performance reports; schedule
rotated `bench.anchor` rounds only after this correctness step succeeds.
No checkpoint is downloaded and no default is automatically promoted.

Local cleanup is prepared separately from acceptance and never runs during
project jobs. Both commands below are read-only:

```bash
python scripts/cleanup_inventory.py --out runs/r15/cleanup_inventory
python scripts/cleanup_plan.py --inventory runs/r15/cleanup_inventory/inventory.json \
  --scheduler-state runs/r15/rf1/scheduler/state.json --out runs/r15/cleanup_dry_run
```

The dry-run preserves checkpoints and datasets, and blocks different owners,
symlinks, unpublished commits (including detached worktrees), stashes, edits
and local sources. Nested candidates count only once. It performs no deletion;
repeat all checks after the final push before reviewing deletion candidates.

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

`TILEMEGA_TARGET_ARCH` selects sm_80/89/90/100/120. Stated: on H800, configure
with `-DTILEMEGA_TARGET_ARCH=sm_90`; `auto` uses nvcc's native base target and
does not enable Hopper architecture-accelerated instructions. Explicit sm_90
and sm_100 use compute_90a/sm_90a and compute_100a/sm_100a gencode respectively,
without requiring CMake support for an `a` architecture suffix. Target JSON
accepts `nvcc_arch`; omitted values map sm_90 to sm_90a and sm_100 to sm_100a,
with other tags unchanged. Verified: CUDA 12.8 `-arch=sm_90a` also emits a
compute_90 pass for linked outputs; TileMega uses explicit gencode for accelerated
targets so Hopper-only instructions are checked exclusively in compute_90a.
Stated: the additional capability bits follow the
[CUDA 12.8 PTX ISA](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html);
WGMMA is Hopper-only, while setmaxnreg requires an accelerated target (including
sm_120a when used on Sm120). `check-policy` checks device
architecture dispatch conventions. Offline compilation does not certify
execution on a different GPU. `configs/targets/sm_90_h800.json` is an
uncalibrated H800 resource profile; it must be calibrated on hardware before
performance selection. `doctor --hwcheck --config ...` exercises native
transport paths and numerical smoke checks, not a performance optimum.


Hopper decode GEMM can be explicitly selected with
`--gemm-impl-{qkv,o,gate-up,down,lm-head} wgmma`. Verified: the implementation
uses WGMMA SS on transposed operands and the existing FP32-tile epilogue.
Only TileM=16, TileN=64/128/256, TileK=64/128, stages=2 are accepted; paged
execution additionally requires each B stage to fit one page. Its activation
workspace uses 1024-byte alignment. Nonpaged transfer currently uses cp.async;
paged transfer retains the bulk/TMA loader. Im2col and scaled activation paths
reject this selection. WGMMA stays outside automatic candidate enumeration
until calibrated costs and model-level end-to-end A/B justify enabling it.


The experimental `--dm-wgmma-inline 1` coordinate specializes MoE access modes
and inlines the complete narrow WGMMA dispatch. Omission is zero. Its legal
domain requires native nonpaged decode, `--moe-activation-bulk 1`,
`--attention-noinline 1`, and every GEMM at M16N64K64 with WGMMA. The packed
weight layout must request TN64 before import; changing generated geometry
is unsupported. Verified: the full 48-layer private CPU prototype had no
serialization warnings or local-memory instructions in audited WGMMA loops;
cold-path compiler spills remain. Official artifact and model-level checks
must pass before this candidate supplies performance results. Defaults and
SM80 execution remain unchanged.


Narrow native decode GEMV accepts `--gemv-load ldg|nc|bulk`; omission keeps
`ldg`. Verified: `bulk` retains the SIMT arithmetic and uses two 32 KiB weight
slots, mbarrier completion and vector shared loads. It requires bulk-copy
capability, explicit nonpaged `--pg off|l2`, an active GEMV implementation,
TN8/16/32 and complete K tiles. Prefill, DM operands, `--moe-gemv` and forced
SM80 transport are rejected. The exact shared-memory allocation participates
in legality, occupancy and the cache key. Isolated bulk tests passed432 cases
in each of five fresh processes; this does not establish full-model speedup.
Inferred: the candidate initially inherits the SIMT cost fit; a separately
measured bulk fit and end-to-end acceptance remain outstanding. Defaults stay
unchanged.


Hopper prefill/forward dense GEMM is an explicit candidate through
`--forward-gemm-impl wgmma` with the skeleton solver. It selects M64/128,
N64/128/256, K64, stages=2 (M*N<=16384) for dense classes with at least 64
active rows, preserving existing bodies for convolution, gathers and smaller
classes. Verified: offline cubins contain HGMMA and UTMALDG with no spills or
WGMMA serialization warnings. A/B tensor maps use per-invocation boxes, with
two shared slots and mbarriers; packed B uses bulk. Each slot is overwritten
only after its WGMMA group completes. End-to-end acceptance and calibrated
costs remain required before automatic defaults change.

Forward DNN export seeds can use `--seed-arch native --forward-seed-tile
M64N128K64S2` without a skeleton solve. The explicit tuple is recorded in the
plan identity and applies to unfused GEMMs; DW/PW bodies retain their existing
specialized shapes. Omission preserves M128N64K16S3. WGMMA and im2col requests
use the solver's existing access/layout eligibility checks, retaining MMA for
unsupported classes. The seed coordinate rejects solved CGs, variants, paged
execution and shared-layout overrides. Stated: this declares a candidate,
not a solver optimum; guarded resource, numerical and end-to-end gates remain
required before changing defaults.

Resource probes reserve 8192 MiB of actual free disk by default. A bounded
experiment may explicitly set `--resource-probe-min-free-mib 2048` (minimum
1024); smaller reservations serialize compiler probes. Each probe records
its selected reservation and observed free space in `disk_budget.json` and
still runs the real CUDA occupancy query under the GPU guard. This changes
workspace admission only; numerical and performance criteria stay fixed.
