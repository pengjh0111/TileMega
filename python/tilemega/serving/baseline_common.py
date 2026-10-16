"""Standalone benchmark protocol; deliberately imports no TileMega modules."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import uuid


class InvalidMeasurement(RuntimeError):
    retryable = True


class InvalidTiming(InvalidMeasurement):
    retryable = False

    def __init__(self, message, **diagnostics):
        self.diagnostics = diagnostics
        super().__init__(f"{message}: {json.dumps(diagnostics, sort_keys=True)}")


def arguments(system):
    parser = argparse.ArgumentParser(description=f"Independent {system} baseline")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-ids", type=Path, required=True)
    parser.add_argument("--batch", choices=("1", "2", "4", "8", "16", "all"), default="all")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--load-format", choices=("auto", "dummy"), default="auto")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--round-index", type=int, default=0)
    args = parser.parse_args()
    if args.max_tokens < 1 or args.warmup < 1 or args.repeats < 1:
        parser.error("max-tokens, warmup and repeats must be positive")
    ids = json.loads(args.prompt_ids.read_text())
    limit = 16 if args.batch == "all" else int(args.batch)
    validate_prompts(ids, limit)
    if not (args.model / "config.json").is_file():
        parser.error("model must be a local directory containing config.json")
    shared_checkpoint_identity(args.model, args.load_format)
    args.out.mkdir(parents=True, exist_ok=True)
    return args, ids


def validate_prompts(ids, batch):
    if not isinstance(ids, list) or len(ids) < batch or not ids or not isinstance(ids[0], list) or not ids[0]:
        raise ValueError("frozen prompts do not cover the requested batch")
    length = len(ids[0])
    if any(not isinstance(row, list) or len(row) != length or
           any(type(token) is not int or token < 0 for token in row) for row in ids):
        raise ValueError("frozen prompts must be rectangular nonnegative integer token IDs")
    return length


def require_guard():
    index = os.environ.get("TILEMEGA_DEVICE_INDEX")
    if index is None or not index.isdigit() or os.environ.get("CUDA_VISIBLE_DEVICES") != index:
        raise ValueError("set TILEMEGA_DEVICE_INDEX=i and CUDA_VISIBLE_DEVICES=i explicitly")
    if os.environ.get("TILEMEGA_GPU_LOCK_HELD") != "1":
        raise ValueError("launch through the guarded scheduler holding /tmp/tilemega_gpu<i>.lock")
    return index


def versions():
    result = {}
    for package in ("torch", "vllm", "sglang", "tensorrt-llm", "transformers", "triton",
                    "flash-attn", "flashinfer-python", "nvidia-cuda-runtime-cu12",
                    "nvidia-cudnn-cu12", "tensorrt", "numpy"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def shared_checkpoint_identity(model, load_format):
    path = model / "random_checkpoint.json"
    if not path.is_file():
        return {}
    manifest = json.loads(path.read_text())
    if load_format != "auto" or manifest.get("state") != "live":
        raise ValueError("shared checkpoints require a live supervisor and the official auto loader")
    if hashlib.sha256((model / "config.json").read_bytes()).hexdigest() != manifest['config_sha256']:
        raise ValueError("shared checkpoint configuration differs from its manifest")
    identity = {key: manifest[key] for key in
                ("algorithm", "config_sha256", "seed", "aliases", "tensors")}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True,
                         separators=(",", ":")).encode()).hexdigest()
    if digest != manifest['checkpoint_sha256']:
        raise ValueError("shared checkpoint identity is inconsistent")
    if (manifest['tensor_count'] != len(manifest['tensors']) or
            manifest['total_bytes'] != sum(tensor['bytes'] for tensor in manifest['tensors']) or
            manifest['total_bytes'] != sum(shard['data_bytes'] for shard in manifest['shards'])):
        raise ValueError("shared checkpoint tensor and shard counts are inconsistent")
    if not manifest['shards'] or any(
            not (model / shard['file']).is_file() or
            (model / shard['file']).stat().st_size != shard['bytes']
            for shard in manifest['shards']):
        raise ValueError("shared checkpoint shard is unavailable or has a different size")
    return dict(checkpoint={key: manifest[key] for key in
                ("algorithm", "seed", "config_sha256", "checkpoint_sha256", "total_bytes", "tensor_count")},
                shared_checkpoint_identity=True)


def _own_pids():
    parents = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            parents[int(path.parent.name)] = int(path.read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
    own = {os.getpid()}
    while True:
        added = {pid for pid, parent in parents.items() if parent in own} - own
        if not added:
            return own
        own.update(added)


def observe(path, label):
    index = require_guard()
    fields = ("power.draw", "clocks.sm", "clocks.mem", "temperature.gpu", "memory.used",
              "clocks_throttle_reasons.hw_slowdown", "clocks_throttle_reasons.hw_thermal_slowdown",
              "clocks_throttle_reasons.hw_power_brake_slowdown",
              "clocks_throttle_reasons.sw_thermal_slowdown", "clocks_throttle_reasons.active")
    raw = subprocess.check_output(["nvidia-smi", "-i", index, "--query-gpu=" + ",".join(fields),
                                   "--format=csv,noheader,nounits"], text=True)
    values = [value.strip() for value in raw.splitlines()[0].split(",")]
    if len(values) != len(fields):
        raise InvalidMeasurement("incomplete GPU telemetry")
    telemetry = dict(zip(fields, values))
    apps = subprocess.check_output(["nvidia-smi", "-i", index,
        "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True)
    pids, visible_mib = [], 0
    for line in apps.splitlines():
        if not line.strip():
            continue
        pid, memory = (x.strip() for x in line.split(","))
        pids.append(int(pid))
        visible_mib += int(memory)
    hidden_mib = max(0, int(float(telemetry["memory.used"])) - visible_mib)
    not_own = set(pids) - _own_pids()
    unresolved = sorted(pid for pid in not_own if not Path(f"/proc/{pid}").exists())
    limited = os.environ.get("TILEMEGA_GPU_PID_MODE") == "namespace_limited"
    foreign = sorted(not_own - set(unresolved)) if limited else sorted(not_own)
    # Initialization already consumes GPU power. Only the outer guard compares
    # idle power; applying its idle threshold here would reject our own engine.
    throttle_values = values[5:9]
    try:
        active_mask = int(values[9], 0)
    except ValueError as error:
        raise InvalidMeasurement("GPU throttle mask unavailable") from error
    valid = not foreign and hidden_mib <= 1024 and not (active_mask & 0xE8) and all(
        value.lower() in ("not active", "inactive", "0", "no") for value in throttle_values)
    sample = {"time": time.time(), "label": label, "gpu_index": int(index),
              "telemetry": telemetry, "pids": pids, "foreign_pids": foreign,
              "hidden_mib": hidden_mib, "accepted": valid, "unresolved_host_pids": unresolved,
              "ownership_verified": not limited,
              "acceptance": "provisional_container_pid_namespace" if limited else "strict"}
    with Path(path).open("a") as stream:
        stream.write(json.dumps(sample) + "\n")
    if not valid:
        raise InvalidMeasurement(f"GPU sample rejected: {sample}")
    return sample


def _process_start(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] in ("Z", "X") else fields[19]
    except (OSError, IndexError):
        return None


def m0_pair_barrier(directory, gpu, participants, round_index, run, timeout_s=120, poll_s=.01):
    participants = tuple(sorted(str(value) for value in participants))
    gpu = str(gpu)
    if len(participants) != 2 or len(set(participants)) != 2 or gpu not in participants:
        raise InvalidTiming("M0 requires two distinct physical GPU participants")
    if not math.isfinite(timeout_s) or timeout_s <= 0 or poll_s <= 0:
        raise InvalidTiming("invalid M0 barrier timeout")
    directory = Path(directory)/f"round{round_index}"/f"N1024-run{run}"
    directory.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    own = dict(gpu=gpu, pid=os.getpid(), process_start=_process_start(os.getpid()), nonce=nonce,
               round=round_index, run=run, participants=list(participants), ready_against={})
    own_path = directory/f"gpu{gpu}.json"
    def write():
        temporary = directory/f"gpu{gpu}.{nonce}.tmp"
        temporary.write_text(json.dumps(own) + "\n")
        temporary.replace(own_path)
    write()
    peer_gpu = next(value for value in participants if value != gpu)
    peer_path = directory/f"gpu{peer_gpu}.json"
    deadline = time.monotonic()+timeout_s
    peer = None
    while True:
        try:
            peer = json.loads(peer_path.read_text())
            valid = (peer.get('gpu') == peer_gpu and peer.get('round') == round_index and peer.get('run') == run
                     and peer.get('participants') == list(participants) and peer.get('nonce')
                     and peer.get('process_start') is not None
                     and _process_start(peer.get('pid')) == peer.get('process_start'))
        except (OSError, ValueError, TypeError):
            valid = False
        if valid:
            if own['ready_against'].get(peer_gpu) != peer['nonce']:
                own['ready_against'][peer_gpu] = peer['nonce']
                write()
            # Both fresh nonces must be acknowledged. A live old receipt alone
            # cannot release a retry, even when the OS later reuses its PID.
            if peer.get('ready_against', {}).get(gpu) == nonce:
                return dict(directory=str(directory.resolve()), participants=list(participants),
                            round=round_index, run=run, nonce=nonce, peer_nonce=peer['nonce'],
                            peer_pid=peer['pid'], peer_process_start=peer['process_start'])
        if time.monotonic() >= deadline:
            failure = dict(reason='M0 peer readiness timeout', own=own, peer=peer, timeout_s=timeout_s)
            (directory/f"gpu{gpu}.{nonce}.failed.json").write_text(json.dumps(failure, indent=2) + "\n")
            raise InvalidTiming("M0 peer readiness timeout", directory=str(directory), gpu=gpu,
                                round=round_index, run=run, peer=peer)
        time.sleep(poll_s)


def optional_m0_barrier(args, count, run):
    directory = os.environ.get('TILEMEGA_M0_BARRIER_DIR')
    if not directory or count != 1024 or run < args.warmup:
        return None
    participants = os.environ.get('TILEMEGA_M0_PARTICIPANTS', '').split(',')
    return m0_pair_barrier(directory, require_guard(), participants, args.round_index, run,
                          timeout_s=float(os.environ.get('TILEMEGA_M0_BARRIER_TIMEOUT_S', '120')))


def check_tokens(tokens, batch, count):
    if (not isinstance(tokens, list) or len(tokens) != batch or
            any(not isinstance(row, list) or len(row) != count or
                any(type(token) is not int or token < 0 for token in row) for row in tokens)):
        raise AssertionError("backend did not return the exact requested token table")


def summarize(rows, batch, count):
    def median(n):
        values = [row["wall_seconds"] for row in rows if row["N"] == n and not row["warmup"]]
        if not values or any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("invalid or empty measurements")
        return statistics.median(values)
    e2e, ttft = median(count), median(1)
    if count > 1 and e2e <= ttft:
        raise InvalidTiming("E2E must exceed TTFT for a positive TPOT",
                            e2e_seconds=e2e, ttft_seconds=ttft, max_tokens=count)
    return {"e2e_seconds": e2e, "ttft_seconds": ttft,
            "tpot_seconds": (e2e-ttft)/(count-1) if count > 1 else None,
            "output_tokens_per_second": batch*count/e2e}


def measure(args, ids, system, generate, metadata, synchronize=lambda: None):
    checkpoint = shared_checkpoint_identity(args.model, args.load_format)
    batches = (1, 2, 4, 8, 16) if args.batch == "all" else (int(args.batch),)
    for batch in batches:
        out = args.out / f"B{batch}"
        out.mkdir(parents=True, exist_ok=True)
        guard_path, rows = out / "guard.jsonl", []
        unresolved = set()
        # N=1 first exposes initialization/compile failures before a long run.
        for count in dict.fromkeys((1, args.max_tokens)):
            for run in range(args.warmup + args.repeats):
                try:
                    barrier = optional_m0_barrier(args, count, run)
                except InvalidTiming as error:
                    (out/"invalid_measurements.json").write_text(json.dumps({
                        "valid": False, "reason": str(error), "retryable": False,
                        "diagnostics": error.diagnostics, "generation_runs": rows}, indent=2) + "\n")
                    raise
                sample = observe(guard_path, f"N{count}-run{run}-before")
                unresolved.update(sample.get("unresolved_host_pids", []))
                synchronize()
                start = time.perf_counter()
                tokens = generate(ids[:batch], count)
                synchronize()
                elapsed = time.perf_counter() - start
                with (out/"raw_timings.jsonl").open("a") as stream:
                    stream.write(json.dumps({"N": count, "run": run, "warmup": run < args.warmup,
                        "wall_seconds": elapsed, "validation": "pending_post_guard_and_tokens"}) + "\n")
                sample = observe(guard_path, f"N{count}-run{run}-after")
                unresolved.update(sample.get("unresolved_host_pids", []))
                check_tokens(tokens, batch, count)
                token_file = f"tokens_N{count}_run{run}.json"
                (out/token_file).write_text(json.dumps(tokens, separators=(",", ":")) + "\n")
                record = {"N": count, "run": run, "warmup": run < args.warmup,
                          "wall_seconds": elapsed, "tokens_file": token_file}
                if barrier is not None:
                    record["m0_barrier"] = barrier
                rows.append(record)
        try:
            metrics = summarize(rows, batch, args.max_tokens)
        except (InvalidMeasurement, ValueError) as error:
            (out/"invalid_measurements.json").write_text(json.dumps({
                "valid": False, "reason": str(error), "retryable": getattr(error, "retryable", False),
                "diagnostics": getattr(error, "diagnostics", {}), "generation_runs": rows,
                "configuration": metadata}, indent=2) + "\n")
            raise
        if shared_checkpoint_identity(args.model, args.load_format) != checkpoint:
            raise InvalidTiming("shared checkpoint changed during measurement")
        summary = {**checkpoint, "schema_version": 1, "evidence": "verified", "system": system,
                   "model": str(args.model.resolve()), "model_config_sha256": hashlib.sha256(
                       (args.model/"config.json").read_bytes()).hexdigest(),
                   "prompt_ids_sha256": hashlib.sha256(args.prompt_ids.read_bytes()).hexdigest(),
                   "batch": batch, "prompt_length": len(ids[0]), "max_tokens": args.max_tokens,
                   "round_index": args.round_index, "pid": os.getpid(),
                   "gpu_index": int(require_guard()), "versions": versions(),
                   "load_format": args.load_format,
                   "weight_semantics": "backend-native random initialization; cross-system checkpoint identity not established"
                       if args.load_format == "dummy" else "local checkpoint",
                   "guard_log": str(guard_path.resolve()), "generation_runs": rows,
                   "ownership_verified": os.environ.get("TILEMEGA_GPU_PID_MODE") != "namespace_limited",
                   "unresolved_host_pids": sorted(unresolved),
                   "acceptance": "provisional_container_pid_namespace"
                       if os.environ.get("TILEMEGA_GPU_PID_MODE") == "namespace_limited" else "strict",
                   "measurement_policy": json.loads(args.policy.read_text()) if args.policy else {},
                   "configuration": metadata, **metrics}
        if system == "vllm":
            summary["vllm_version"] = summary["versions"]["vllm"]
        (out/"measurements.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({key: value for key, value in summary.items() if key != "generation_runs"}), flush=True)


def guarded_main(main):
    # A native crash must release the guard, rather than wait indefinitely on
    # a host crash collector that cannot inspect this container's namespace.
    import resource
    _, maximum = resource.getrlimit(resource.RLIMIT_CORE)
    resource.setrlimit(resource.RLIMIT_CORE, (0, maximum))
    # Linux ignores RLIMIT_CORE for a piped core_pattern. Disable dumping on
    # this worker as well, so a native destructor crash cannot retain its GPU.
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(4, 0, 0, 0, 0) != 0:  # PR_SET_DUMPABLE
        raise OSError(ctypes.get_errno(), 'cannot disable worker core dumps')
    try:
        main()
    except InvalidMeasurement as error:
        print(str(error), flush=True)
        raise SystemExit(75 if error.retryable else 3) from error
