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


def _gpu_owners() -> tuple[set[int], int, int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader,nounits"], text=True)
    rows = [line.split(",") for line in output.splitlines() if line.strip()]
    pids = {int(row[0].strip()) for row in rows}
    visible_mib = sum(int(row[-1].strip()) for row in rows)
    used_mib = int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True).splitlines()[0].strip())
    return pids, visible_mib, used_mib


def _exclusive(path: Path, label: str, wait: bool) -> bool:
    deadline = time.monotonic() + (30 * 60 if wait else 0)
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
        ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw",
         "--format=csv,noheader,nounits"], text=True).strip()


def measure(engine: ServingEngine, prompts: torch.Tensor,
            out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    guard = out / "guard.jsonl"
    rows: list[dict] = []
    for count in (engine.max_new_tokens, 1):
        for run in range(4):
            for attempt in range(3):
                if not _exclusive(guard, f"N{count}-run{run}-before", True):
                    raise RuntimeError("GPU remained occupied for 30 minutes")
                clocks_before = _clocks()
                result = engine.generate(prompts, count)
                clocks_after = _clocks()
                if _exclusive(guard, f"N{count}-run{run}-after", False):
                    break
            else:
                raise RuntimeError("GPU was contaminated on all three attempts")
            tokens = result.tokens.tolist()
            row = {"N": count, "run": run, "warmup": run == 0,
                   "e2e_seconds": result.e2e_ms / 1e3,
                   "gpu_step_ms": result.step_ms,
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
    summary = {"batch": engine.batch, "max_tokens": engine.max_new_tokens,
               "e2e_seconds": e2e, "ttft_seconds": ttft,
               "tpot_seconds": (e2e - ttft) / (engine.max_new_tokens - 1),
               "output_tokens_per_second": engine.batch * engine.max_new_tokens / e2e,
               "runs": rows}
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
    parser.add_argument("--mode", choices=("auto", "L1", "L2"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    ids = json.loads(args.prompt_ids.read_text())
    if len(ids) != 16 or any(len(row) != 64 for row in ids):
        raise ValueError("the frozen prompt table must be 16 by 64")
    prompts = torch.tensor(ids[:args.batch], dtype=torch.int32)
    with ServingEngine(args.model, args.prefill_so, args.decode_so,
                       args.batch, max_new_tokens=args.max_new_tokens,
                       mode=args.mode) as engine:
        result = measure(engine, prompts, args.out)
    print(json.dumps({key: value for key, value in result.items()
                      if key != "runs"}))


if __name__ == "__main__":
    main()
