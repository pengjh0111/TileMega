"""Measure one complete static-batch request with an exclusive-GPU guard."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch

from .engine import ServingEngine
from .measurement_policy import TimingPolicy


def _gpu_owners() -> tuple[set[int], int, int]:
    output = subprocess.check_output(
        ["nvidia-smi", "-i", os.environ.get("TILEMEGA_DEVICE_INDEX", "0"), "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader,nounits"], text=True)
    rows = [line.split(",") for line in output.splitlines() if line.strip()]
    pids = {int(row[0].strip()) for row in rows}
    visible_mib = sum(int(row[-1].strip()) for row in rows)
    used_mib = int(subprocess.check_output(
        ["nvidia-smi", "-i", os.environ.get("TILEMEGA_DEVICE_INDEX", "0"), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True).splitlines()[0].strip())
    return pids, visible_mib, used_mib


def _preflight_external_memory(out: Path) -> None:
    """Reject an invisible GPU allocation before loading model weights.

    The normal exclusivity check runs after ServingEngine construction; an
    external allocation can cause weight packing to OOM before that check.
    """
    _, visible_mib, used_mib = _gpu_owners()
    hidden_mib = max(0, used_mib - visible_mib)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "guard.jsonl").open("a") as stream:
        stream.write(json.dumps({"label": "before-weight-load",
                                 "visible_mib": visible_mib,
                                 "used_mib": used_mib,
                                 "hidden_mib": hidden_mib,
                                 "accepted": hidden_mib <= 1024}) + "\n")
    if hidden_mib > 1024:
        raise SystemExit(75)


def _exclusive(path: Path, label: str, wait: bool, wait_s: int = 1800) -> bool:
    deadline = time.monotonic() + (wait_s if wait else 0)
    while True:
        observed, visible_mib, used_mib = _gpu_owners()
        # Keep the device-wide accounting difference for diagnosis. NVML can
        # attribute the current process's CUDA allocations to memory.used
        # before they appear in the per-process table, so it is not an
        # exclusivity criterion. The required guard is the process list.
        hidden_mib = max(0, used_mib - visible_mib)
        good = observed == {os.getpid()}
        with path.open("a") as output:
            output.write(json.dumps({"label": label, "pids": sorted(observed),
                                     "visible_mib": visible_mib,
                                     "used_mib": used_mib,
                                     "hidden_mib": hidden_mib,
                                     "exclusive": good, "time": time.time()}) + "\n")
        if good or time.monotonic() >= deadline:
            return good
        time.sleep(10)


def _clocks() -> str:
    return subprocess.check_output(
        ["nvidia-smi", "-i", os.environ.get("TILEMEGA_DEVICE_INDEX", "0"), "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw",
         "--format=csv,noheader,nounits"], text=True).strip()


def measure(engine: ServingEngine, prompts: torch.Tensor,
            out: Path, *, warmup: int = 1, repeats: int = 3, policy_path=None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    guard = out / "guard.jsonl"
    rows: list[dict] = []
    policy = TimingPolicy(policy_path)
    for count in (engine.max_new_tokens, 1):
        for run in range(warmup + repeats):
            for attempt in range(3):
                if policy.options.get("guard", True) and not _exclusive(guard, f"N{count}-run{run}-before", True):
                    raise SystemExit(75)
                if not policy.observe(guard, f"N{count}-run{run}-attempt{attempt}-before"):
                    continue
                clocks_before = _clocks()
                try:
                    result = engine.generate(prompts, count)
                except BaseException:
                    (out / "watchdog.json").write_text(
                        json.dumps(engine.decode.watchdog(), indent=2) + "\n")
                    raise
                clocks_after = _clocks()
                power_ok = policy.observe(guard, f"N{count}-run{run}-attempt{attempt}-after")
                if (not policy.options.get("guard", True) or _exclusive(guard, f"N{count}-run{run}-after", False)) and power_ok:
                    break
            else:
                raise SystemExit(75)
            tokens = result.tokens.tolist()
            row = {"N": count, "run": run, "warmup": run < warmup,
                   "e2e_seconds": result.e2e_ms / 1e3,
                   "gpu_step_ms": result.step_ms,
                   "decode_loop_used": result.decode_loop_used,
                   "step_ns_read": result.step_ns_read,
                   "clocks_before": clocks_before, "clocks_after": clocks_after,
                   "tokens": tokens}
            rows.append(row)
            (out / f"tokens_N{count}_run{run}.json").write_text(
                json.dumps(tokens, separators=(",", ":")) + "\n")
    timed = lambda n: [r["e2e_seconds"] for r in rows
                       if r["N"] == n and not r["warmup"]]
    full = timed(engine.max_new_tokens)
    first = timed(1)
    e2e = statistics.median(full)
    ttft = statistics.median(first)
    timed_rows = [r for r in rows if r['N'] == engine.max_new_tokens and not r['warmup']]
    same_tokens = all(r['tokens'] == timed_rows[0]['tokens'] for r in timed_rows)
    if not same_tokens:
        raise AssertionError('C-2: timed generations produced different tokens')
    decode = sorted(ms / 1000 for r in timed_rows for ms in r['gpu_step_ms'][1:])
    def quantile(q):
        if not decode: return 0.0
        at = q * (len(decode) - 1); low = int(at); high = min(low + 1, len(decode) - 1)
        return decode[low] + (decode[high] - decode[low]) * (at - low)
    summary = {"timed_tokens_identical": same_tokens,
               "decode_loop_used": all(r["decode_loop_used"] for r in timed_rows),
               "step_ns_read": all(r["step_ns_read"] for r in timed_rows),
               "tpot_mean_seconds": statistics.mean(decode) if decode else 0.0,
               "tpot_p50_seconds": quantile(.5), "tpot_p90_seconds": quantile(.9),
               "measurement_policy": policy.options,
               "batch": engine.batch, "max_tokens": engine.max_new_tokens,
               "e2e_seconds": e2e, "ttft_seconds": ttft,
               "tpot_seconds": (e2e - ttft) / (engine.max_new_tokens - 1),
               "output_tokens_per_second": engine.batch * engine.max_new_tokens / e2e,
               "runs": rows}
    if not decode:
        # No per-step CUDA events (and no device timestamps) means these
        # quantiles were not observed. Zero would look like a timing result.
        for key in ('tpot_mean_seconds','tpot_p50_seconds','tpot_p90_seconds'):
            summary.pop(key)
    (out / "measurements.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out / "step_times.tsv").open("w") as stream:
        stream.write("run\tstep\tgpu_ms\n")
        for row in rows:
            if row["N"] == engine.max_new_tokens and not row["warmup"]:
                for step, elapsed in enumerate(row["gpu_step_ms"]):
                    stream.write(f"{row['run']}\t{step}\t{elapsed:.9g}\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prefill-so", type=Path, required=True)
    parser.add_argument("--decode-so", type=Path, required=True)
    parser.add_argument("--prompt-ids", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("auto", "L1", "L2"), default="L2")
    parser.add_argument("--decode-loop", choices=("0", "1", "auto"), default="1")
    parser.add_argument("--step-events", type=int, choices=(0, 1), default=1)
    parser.add_argument("--prefill-mode", choices=("L1", "L2", "auto"))
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--policy", type=Path)
    args = parser.parse_args()
    ids = json.loads(args.prompt_ids.read_text())
    if len(ids) != 16 or any(len(row) != 64 for row in ids):
        raise ValueError("the frozen prompt table must be 16 by 64")
    prompts = torch.tensor(ids[:args.batch], dtype=torch.int32)
    _preflight_external_memory(args.out)
    with ServingEngine(args.model, args.prefill_so, args.decode_so,
                       args.batch, max_new_tokens=args.max_new_tokens,
                       mode=args.mode, decode_loop=("auto" if args.decode_loop == "auto" else bool(int(args.decode_loop))),
                       step_events=bool(args.step_events),
                       prefill_mode=args.prefill_mode) as engine:
        result = measure(engine, prompts, args.out, warmup=args.warmup, repeats=args.repeats, policy_path=args.policy)
    if args.decode_loop == "1" and args.max_new_tokens > 1 and not result["decode_loop_used"]:
        raise RuntimeError("requested decode loop was not used")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "runs"}))


if __name__ == "__main__":
    main()
