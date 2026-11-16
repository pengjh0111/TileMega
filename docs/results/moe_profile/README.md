# Qwen3-30B-A3B execution profiling

This branch is an independent measurement study, based on `tilemega`
`ebe2fb0ed9d56b9b1e1abdd57c8fa8fd6a835f4c`. It does not enable a TileMega
optimization. Raw logs, traces, outputs and checkpoint bytes stay in the
ignored `runs/moe_profile/` or the task-owned checkpoint directory.

The completed study is in [report.md](report.md): 36 accepted baseline cells,
18 native timeline windows, 11 source-attribution cases and nine route cases.
See [coverage.json](coverage.json) for executed scope and
[cleanup.json](cleanup.json) for the subsequent payload deletion receipt.

## Scope and evidence

- **stated / design:** primary experiment A is a closed finite cohort: submit
  B requests once and insert no new requests. This isolates execution from an
  external arrival process. It is sufficient for the main Decode study;
  online continuous-batch and TP2 supplements were not executed. No arrival,
  queue, expert-parallel or multi-GPU communication conclusion is made.
- **stated / design:** use real pretrained weights and frozen natural technical
  prose/source-code tokens. Random weights cannot establish the pretrained
  model's expert distribution. Random inputs/routes remain microbenchmark
  controls only. The input corpus is technical, not a general-user sample.
- **verified:** the current device map has **three**, not four, H800 GPUs.
  Physical UUID/PCI identity and owner/memory checks precede each GPU job.
  Each timing job takes the shared NUMA0 lock and records clocks, power,
  temperature and other visible owners. Host PID visibility is limited by
  the container; no claim of host-wide ownership verification is made.
- **verified:** tiny CUDA canaries pass in all three existing environments.
  Nsight Systems GPU capture works. Nsight Compute returns
  `ERR_NVGPUCTRPERM`; hardware counters are unavailable and remain blank.
  No driver, clock, power or profiling permission setting was changed.

The complete protocol is [workload.json](../../../configs/moe_profile/workload.json).
Versions and source fingerprints are in [environment.json](environment.json).
The real model is pinned to
[`ad44e777`](https://huggingface.co/Qwen/Qwen3-30B-A3B/tree/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39).
An official Qwen ModelScope mirror is permissible only when each actual
shard SHA256 equals this pinned Hugging Face revision. A tmpfs checkpoint
cache reduces disk use; inference still places weights on GPU with no
runtime offload.

## Reproduce

Use the existing independent Python environments recorded in the protocol.
All Python command entries live under `/root/python`; the generic
`run_repo_script.py --entry ABSOLUTE_REPOSITORY_SCRIPT` executes repo tools.
The model, downloaded tokenizer copies, replay tensor payloads and complete
input-ID file were deleted after collection at the user's request. The final
branch HEAD contains identities/manifests and a deterministic input recipe,
not the full input-ID payload. Reproduction downloads/regenerates them.
On a new machine, create the generic launcher once:

```bash
mkdir -p /root/python
cat > /root/python/run_repo_script.py <<'PY'
from pathlib import Path
import argparse, runpy, sys
p = argparse.ArgumentParser()
p.add_argument('--entry', type=Path, required=True)
a, rest = p.parse_known_args()
entry = a.entry.resolve()
sys.path.insert(0, str(entry.parent))
sys.argv = [str(entry), *rest]
runpy.run_path(str(entry), run_name='__main__')
PY
```

For example, with `moe_repo=/root/TileMega-moe-profile`:

```bash
/workspace/llm-stack/envs/vllm/bin/python /root/python/run_repo_script.py \
  --entry "$moe_repo/scripts/moe_profile_fetch.py" \
  --transport modelscope --direct \
  --target /dev/shm/tilemega-moe-profile-qwen3-ad44e777 \
  --out "$moe_repo/runs/moe_profile/model_metadata"

/workspace/llm-stack/envs/vllm/bin/python /root/python/run_repo_script.py \
  --entry "$moe_repo/scripts/moe_profile_inputs.py" \
  --model /dev/shm/tilemega-moe-profile-qwen3-ad44e777 \
  --out "$moe_repo/configs/moe_profile"

/root/TileMega/runs/r16/env/tensorrt-dnn/bin/python /root/python/run_repo_script.py \
  --entry "$moe_repo/scripts/moe_profile_matrix.py" \
  --model /dev/shm/tilemega-moe-profile-qwen3-ad44e777 \
  --inputs "$moe_repo/configs/moe_profile/inputs.json.gz" --gpus 0,1
```

Verify the regenerated input SHA256 against `data_identity.json` and every
shard against the pinned download receipt. Only after ordinary measurements:

```bash
/root/TileMega/runs/r16/env/tensorrt-dnn/bin/python /root/python/run_repo_script.py \
  --entry "$moe_repo/scripts/moe_profile_diagnostics.py" \
  --kind timeline --framework all --gpus 1

/root/TileMega/runs/r16/env/tensorrt-dnn/bin/python /root/python/run_repo_script.py \
  --entry "$moe_repo/scripts/moe_profile_diagnostics.py" \
  --kind attribution --framework all --gpus 0 --untimed

/root/TileMega/runs/r16/env/tensorrt-dnn/bin/python /root/python/run_repo_script.py \
  --entry "$moe_repo/scripts/moe_profile_diagnostics.py" \
  --kind attribution --framework all --batch 4 --phase prefill --gpus 0 --untimed

/root/TileMega/runs/r16/env/tensorrt-dnn/bin/python /root/python/run_repo_script.py \
  --entry "$moe_repo/scripts/moe_profile_diagnostics.py" \
  --kind routing --framework all --gpus 0 --untimed

/root/TileMega/runs/r16/env/tensorrt-dnn/bin/python /root/python/run_repo_script.py \
  --entry "$moe_repo/scripts/moe_profile_diagnostics.py" \
  --kind timeline --framework vllm --batch 1 --phase decode --gpus 1 --refresh

/root/TileMega/runs/r16/env/tensorrt-dnn/bin/python /root/python/run_repo_script.py \
  --entry "$moe_repo/scripts/moe_profile_diagnostics.py" \
  --kind timeline --framework vllm --batch 1 --phase decode --gpus 1 --graph-trace graph
```

For the replay controls, set `route_json` to the **accepted** corresponding
vLLM `routing/.../attemptNN/worker/routing_stats/routing.json`, and `replay_b`
to 1 or 4. Run each once, preserving its output directory:

```bash
CUDA_VISIBLE_DEVICES=1 TILEMEGA_DEVICE_INDEX=1 \
TILEMEGA_GPU_LOCK=/tmp/tilemega_gpu1.lock \
TILEMEGA_GPU_PID_MODE=namespace_limited TILEMEGA_NUMA_NODE=0 \
TILEMEGA_TIMING=1 OMP_NUM_THREADS=8 \
taskset -c 16-31 /root/TileMega/runs/r16/env/tensorrt-dnn/bin/python \
  /root/python/run_repo_script.py --entry "$moe_repo/scripts/moe_profile_guard.py" \
  --policy "$moe_repo/configs/moe_profile/guard_policy.json" \
  --out "$moe_repo/runs/moe_profile/replay_graph/b$replay_b/guard" \
  --needs-free-mib 4096 --timeout-s 300 -- \
  /workspace/llm-stack/envs/vllm/bin/python /root/python/run_repo_script.py \
  --entry "$moe_repo/scripts/moe_profile_replay.py" \
  --input "/dev/shm/tilemega-moe-profile-qwen3-ad44e777/diagnostic_replay/layer0_b$replay_b.pt" \
  --routing "$route_json" \
  --out "$moe_repo/runs/moe_profile/replay_graph/b$replay_b/worker"
```

After both controls, publish the compact tables:

```bash
/workspace/llm-stack/envs/vllm/bin/python /root/python/run_repo_script.py \
  --entry "$moe_repo/scripts/moe_profile_collect.py"

/workspace/llm-stack/envs/vllm/bin/python /root/python/run_repo_script.py \
  --entry "$moe_repo/scripts/moe_profile_summarize.py"
```

The final consolidator requires the two control experiments recorded in
`controls.json`; exact original commands are also retained under the linked
raw evidence directories. For timeline/route-only consolidation, use
`--part timelines` and `--part routing`. Routing helper patches and the
SGLang opt-in import hook live in the scripts directory; installed packages
were not edited. Replays require a new routing run to regenerate layer-zero
tensor payloads before running `moe_profile_replay.py` under the same guard.
All source/route runs disable graphs/compile for attribution only; their
times are not native baseline times.

`moe_profile_matrix.py` launches separate guarded cells and preserves every
failed attempt. It resumes only a successful result with an accepted guard,
leaves a gap between jobs and never signals another user's process. GPU
ordinals are checked against the current physical manifest; refresh the
manifest deliberately on a different host.

## Measurement definitions

All main cells use real BF16 weights/activations/KV, no quantization,
offload, speculation or prefix reuse. CUDA Graphs/fusion stay enabled.
Greedy no-thinking generation ignores EOS and forces the declared output
length, keeping cohort members alive. This is not a quality benchmark.

Warmup is three complete rounds after initialization/capture/autotuning;
measurement is another three complete rounds. TTFT is cohort-start to each
request's first host-observed token. TPOT is `(last-first)/(O-1)` per request.
Throughput is `B*O / cohort_wall_seconds`. Report the median and process-round
range. Store actual input/output counts, token hashes and stream chunk sizes.
Coalesced SDK events cannot establish individual GPU iteration durations.
Raw engine timing fields retain their engine-specific definitions.

Nominal cohort B does not guarantee every kernel has M=B. Chunked prefill,
preemption and graph padding must be diagnosed and labelled. In particular,
TP1 B64 I8192/O256 KV plus weights exceed an H800's memory lower bound.
Those three executed cells are closed-cohort capacity-limited scheduling
results, not resident fixed-B64 Decode. They completed without silently
changing actual lengths. No TP2 result replaces a TP1 cell.

Only after ordinary timings: capture short Prefill and stable Decode ROIs,
then target measured hotspots with available profiler capabilities. The
Nsight exporter keeps kernel sums, per-device busy unions and first-to-last
GPU spans separate. Unknown stage mappings stay `unresolved`; source/shape
or NVTX confirmation is mandatory. Fused regions remain fused. Hardware
counter columns are not populated from occupancy heuristics or sampled
`nvidia-smi` utilization.

Routing instrumentation belongs to independent diagnostic runs. Save
device-resident samples then copy after the ROI, preserving the actual
model routing. A low Tensor Core percentage, small batch or high bandwidth
alone cannot establish an implementation defect. Rank measured time
contributions before selecting single-factor causal checks.
