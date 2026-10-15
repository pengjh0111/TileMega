#!/usr/bin/env python3
"""Generate reviewable R16 baseline queues; this script never enqueues GPU work."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MODELS = ("llama3_2_1b", "qwen3_0_6b", "qwen3_1_7b", "llama3_2_3b", "llama3_1_8b", "qwen3_30b_a3b")
SYSTEMS = ("vllm", "sglang", "trtllm", "torchcompile")
ENVIRONMENTS = {"vllm": "vllm", "sglang": "sglang", "trtllm": "trtllm", "torchcompile": "vllm"}


def step(root, env_root, model, system, gpu, node, round_index, *, phase="m1", batch="all",
         max_tokens=1024, repeats=3, pid_mode="strict", cpu_bind_only=False, timing=True):
    config = root/"runs/r16/models"/model/"config.json"
    if not config.is_file():
        raise FileNotFoundError(config)
    model_config = json.loads(config.read_text())
    prompts = root/"configs/prompts"/("qwen_isl64.json" if model.startswith("qwen") else "llama3_isl64.json")
    rows = json.loads(prompts.read_text())
    if len(rows) < 16 or any(len(row) != 64 for row in rows):
        raise ValueError("M1 requires at least sixteen frozen 64-token prompts")
    if max(max(row) for row in rows) >= model_config["vocab_size"]:
        raise ValueError("frozen prompt token exceeds model vocabulary")
    harness = root/f"python/tilemega/serving/{system}_baseline.py"
    python = env_root/ENVIRONMENTS[system]/"bin/python"
    if not harness.is_file() or not python.is_file():
        raise FileNotFoundError(f"missing harness/interpreter: {harness} {python}")
    env = {"TILEMEGA_DEVICE_INDEX": str(gpu), "CUDA_VISIBLE_DEVICES": str(gpu),
           "TILEMEGA_NUMA_NODE": str(node), "TILEMEGA_GPU_PID_MODE": pid_mode,
           "CUDA_HOME": "/usr/local/cuda-12.8", "TORCH_CUDA_ARCH_LIST": "9.0",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "8",
           "VLLM_USE_V1": "1"}
    command = ["numactl", f"--cpunodebind={node}"]
    if not cpu_bind_only:
        command.append(f"--membind={node}")
    command += [str(python), "-u", str(harness), "--model", str(config.parent),
                "--prompt-ids", str(prompts), "--batch", str(batch), "--max-tokens", str(max_tokens),
                "--warmup", "1", "--repeats", str(repeats), "--round-index", str(round_index),
                "--load-format", "dummy", "--policy", str(root/"configs/bench/measurement_policy.json"),
                "--out", "{attempt_dir}/measurements"]
    return {"name": f"{phase}_{model}_{system}_B{batch}_r{round_index}_gpu{gpu}",
            "gpu": True, "timing": timing, "timeout_s": 14400 if timing else 1800,
            "needs_free_mib": 74000, "cwd": str(root), "env": env, "command": command,
            "priority": 100, "harness_sha256": hashlib.sha256(harness.read_bytes()).hexdigest(),
            "protocol": {"isl": 64, "osl": max_tokens, "warmup": 1, "repeats": repeats,
                "fresh_process_round": round_index, "precision": "BF16 weights/activations/KV",
                "random_weights": True, "shared_checkpoint_identity": False,
                "ownership": "provisional_container_pid_namespace" if pid_mode == "namespace_limited" else "strict",
                "numa_memory_binding": "unavailable_explicit_cpu_only_deviation" if cpu_bind_only else "required"}}


def build_plan(root, env_root, gpus=(7, 6), numa=None, pid_mode="strict", cpu_bind_only=False):
    root, env_root = Path(root).resolve(), Path(env_root).resolve()
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("GPU preference must contain distinct device indices")
    numa = {6: 1, 7: 1} if numa is None else numa
    if any(gpu not in numa for gpu in gpus):
        raise ValueError("supply the actual NUMA node for every GPU")
    if pid_mode not in ("strict", "namespace_limited"):
        raise ValueError("unknown ownership mode")
    phases = {}
    def add(phase, job):
        phases.setdefault(phase, {}).setdefault(int(job["env"]["TILEMEGA_DEVICE_INDEX"]), []).append(job)
    def make(model, system, gpu, rnd, **options):
        return step(root, env_root, model, system, gpu, numa[gpu], rnd,
                    pid_mode=pid_mode, cpu_bind_only=cpu_bind_only, **options)
    for phase in ("anchors_start", "anchors_end"):
        for rnd in range(3):
            for batch in (1, 16):
                job = make("llama3_2_1b", "vllm", gpus[0], rnd, phase=phase, batch=str(batch))
                job["priority"] = rnd*2+(batch == 16)
                add(phase, job)
    for rnd in range(3):
        rotated = SYSTEMS[rnd % len(SYSTEMS):] + SYSTEMS[:rnd % len(SYSTEMS)]
        for index, model in enumerate(MODELS):
            gpu = gpus[index % len(gpus)]
            for order, system in enumerate(rotated):
                job = make(model, system, gpu, rnd)
                job["priority"] = rnd*100+index*10+order
                add("m1", job)
    if len(gpus) >= 2:
        for phase in ("m0_solo", "m0_concurrent"):
            for rnd in range(3):
                for gpu in gpus[:2]:
                    job = make("llama3_2_1b", "vllm", gpu, rnd, phase=phase, batch="16")
                    job["priority"] = rnd
                    if phase == "m0_concurrent":
                        job["env"]["TILEMEGA_NUMA_CONCURRENCY_TRIAL"] = "1"
                        job["protocol"]["requires_guard_concurrency_trial_support"] = True
                    add(phase, job)
    recovery_gpu = gpus[1] if len(gpus) >= 2 else gpus[0]
    for system in ("vllm", "trtllm"):
        add("recovery_preview", make("llama3_2_1b", system, recovery_gpu, 0,
            phase="recovery_preview_v2", batch="1", max_tokens=16, repeats=1, timing=False))
    add("recovery_torchcompile", make("llama3_2_1b", "torchcompile", gpus[0], 0,
        phase="recovery_staticcache_v2", batch="1", max_tokens=16, repeats=1, timing=False))
    return phases


def build_recovery_plan(root, env_root, gpus=(7, 6), numa=None, pid_mode="strict", cpu_bind_only=False):
    plan = build_plan(root, env_root, gpus, numa, pid_mode, cpu_bind_only)
    phases = {"recovery_mpi_and_prewarm": {}, "m1_recovered": {}}
    for gpu, jobs in plan["m1"].items():
        recovered = []
        for job in jobs:
            if "trtllm" not in job["name"] and "torchcompile" not in job["name"]:
                continue
            job["name"] = job["name"].replace("m1_", "m1_recovered_v3_", 1)
            if "trtllm" in job["name"]:
                job["env"]["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
                job["protocol"]["executor_deviation"] = "official TP1 single-process mode avoids MPI spawn"
            recovered.append(job)
        if recovered:
            phases["m1_recovered"][gpu] = recovered
    nodes = {6: 1, 7: 1} if numa is None else numa
    trt_gpu = gpus[1] if len(gpus) > 1 else gpus[0]
    trt = step(Path(root).resolve(), Path(env_root).resolve(), "llama3_2_1b", "trtllm", trt_gpu,
        nodes[trt_gpu], 0, phase="recovery_tp1_singleprocess_v3", batch="1", max_tokens=16,
        repeats=1, timing=False, pid_mode=pid_mode, cpu_bind_only=cpu_bind_only)
    trt["env"]["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    trt.update(priority=0, timeout_s=300)
    phases["recovery_mpi_and_prewarm"].setdefault(trt_gpu, []).append(trt)
    for count in (64, 1024):
        gpu = gpus[0]
        job = step(Path(root).resolve(), Path(env_root).resolve(), "llama3_2_1b", "torchcompile", gpu,
            nodes[gpu], 0, phase=f"recovery_prewarm_v3_N{count}", batch="1", max_tokens=count,
            repeats=3, timing=count == 1024, pid_mode=pid_mode, cpu_bind_only=cpu_bind_only)
        job.update(priority=0 if count == 64 else 1, timeout_s=900)
        phases["recovery_mpi_and_prewarm"].setdefault(gpu, []).append(job)
    return phases


def write_plan(directory, phases, root, pid_mode, cpu_bind_only):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    files = []
    for phase, cards in phases.items():
        for gpu, jobs in cards.items():
            destination = directory/phase/f"gpu{gpu}"
            destination.mkdir(parents=True)
            filename = destination/f"queue_{phase}.json"
            filename.write_text(json.dumps(jobs, indent=2) + "\n")
            files.append({"phase": phase, "gpu": gpu, "path": str(filename.resolve()), "jobs": len(jobs)})
    manifest = {"evidence": "verified_queue_generation_only", "enqueued": False,
        "g_m0": "not_run", "g_base": "not_run", "pid_mode": pid_mode,
        "ownership_verified": None, "cpu_bind_only": cpu_bind_only,
        "m1_cells": sum(len(jobs) for phase, cards in phases.items() if phase.startswith("m1") for jobs in cards.values())*5//3,
        "fresh_processes": sum(len(jobs) for phase, cards in phases.items() if phase.startswith("m1") for jobs in cards.values()),
        "m1_measurements_per_cell": 3*3,
        "files": files,
        "activation_order": ["recovery smokes after confirming no matching system is running",
            "anchors_start", "m0_solo", "m0_concurrent (explicit trial support required)",
            "m1 only after smoke and M0 review", "anchors_end after M1 completion"],
        "m0_recipe": {"concurrency_threshold": 0.01, "anchor_drift_threshold": 0.02,
            "cross_card_threshold": 0.02, "verified": False,
            "instructions": [
                "Run m0_solo card queues separately, preserving all three fresh-process measurements.",
                "For m0_concurrent, use an explicit guard trial override, never set NUMA_CONCURRENCY_VERIFIED prematurely.",
                "Launch both card queues together; inspect per-request guard timestamps and require measured N1024 intervals to overlap.",
                "Compare per-card median TPOT and E2E against solo; every difference must be <=1%.",
                "Only after that check succeeds may production timing jobs set TILEMEGA_NUMA_CONCURRENCY_VERIFIED=1.",
                "Compare start/end anchors at B1/B16; drift >2% requires affected batches to be repeated.",
                "Random-weight MoE traffic is synthetic. Record expert counts or leave routing-derived MBU unavailable."]},
        "scheduler_recipe": [
            f"cd {Path(root).resolve()}",
            "Copy only the approved phase's queue file into a dedicated active queue directory.",
            "For each card set TILEMEGA_DEVICE_INDEX=i CUDA_VISIBLE_DEVICES=i TILEMEGA_NUMA_NODE=n TILEMEGA_GPU_LOCK=/tmp/tilemega_gpu${i}.lock PYTHONPATH=python.",
            "Run python3 -m tilemega.bench.scheduler --queue-dir ACTIVE_CARD_QUEUE --out UNIQUE_CARD_STATE --policy configs/bench/guard_policy.json --deadline-hours 96.",
            "Guard obtains the per-card flock; timing additionally obtains the NUMA lock until M0 passes.",
            "Move new recovery jobs only after their framework is no longer running. Never overwrite state or failed receipts."]}
    (directory/"plan.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--env-root", type=Path, default=Path("/workspace/llm-stack/envs"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[7, 6])
    parser.add_argument("--numa-map", type=Path, help="JSON mapping physical GPU index to NUMA node")
    parser.add_argument("--pid-mode", choices=("strict", "namespace_limited"), default="strict")
    parser.add_argument("--recovery-only", action="store_true", help="preview official TP1 and compiled-cache recovery plus remaining matrix")
    parser.add_argument("--cpu-bind-only", action="store_true", help="record unavailable membind as an explicit container deviation")
    args = parser.parse_args()
    numa = {int(key): int(value) for key, value in json.loads(args.numa_map.read_text()).items()} if args.numa_map else None
    builder = build_recovery_plan if args.recovery_only else build_plan
    phases = builder(args.root, args.env_root, tuple(args.gpus), numa, args.pid_mode, args.cpu_bind_only)
    manifest = write_plan(args.out, phases, args.root, args.pid_mode, args.cpu_bind_only)
    print(json.dumps({key: manifest[key] for key in ("enqueued", "m1_cells", "fresh_processes", "g_m0", "g_base")}))


if __name__ == "__main__":
    main()
