"""Time top-three serving plans with finite synthetic device-resident data."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import time

import torch

from .measure import _clocks, _exclusive, _gpu_owners
from .plan import PlanLibrary
from .buffers import _external_buffers
from .smoke import run as smoke_run




def measure_one(plan: PlanLibrary, batch: int, vocab: int,
                past_mid: int, out: Path, reverse_modes: bool = False,
                warmup_override: int | None = None,
                timed_override: int | None = None,
                mode_only: str = "L2", guard_wait_s: int = 1800,
                smoke_steps: int = 0, loop: int = 0, loop_steps: int | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    if not _exclusive(out / "guard.jsonl", "candidate-before", True, guard_wait_s):
        raise SystemExit(75)
    # This check runs before allocating the synthetic weights, when our own
    # context is tiny. It catches GPU users omitted from the compute-app PID
    # table, which otherwise turn the candidate measurement into CUDA OOM.
    deadline = time.monotonic() + guard_wait_s
    while True:
        _, visible_mib, used_mib = _gpu_owners()
        hidden_mib = max(0, used_mib - visible_mib)
        with (out / "guard.jsonl").open("a") as guard:
            guard.write(json.dumps({"label": "candidate-hidden-before",
                                    "hidden_mib": hidden_mib,
                                    "accepted": hidden_mib <= 1024}) + "\n")
        if hidden_mib <= 1024:
            break
        if time.monotonic() >= deadline:
            raise SystemExit(75)
        time.sleep(10)
    buffers = _external_buffers(plan, batch, vocab)
    warmup, timed = (8, 32) if plan.info.phase == 1 else (2, 5)
    if warmup_override is not None:
        warmup = warmup_override
    if timed_override is not None:
        timed = timed_override
    if loop_steps is not None:
        timed = loop_steps
    past = past_mid if plan.info.phase == 1 else 0
    stream = torch.cuda.current_stream()
    clocks_before = _clocks()
    if smoke_steps:
        smoke = smoke_run(plan, batch, smoke_steps, vocab, out / "smoke")
        if not smoke["pass"]:
            (out / "smoke.json").write_text(json.dumps(smoke, indent=2) + "\n")
            raise SystemExit(3)
    modes = [{"L1": 1, "L2": 2}[mode_only]]
    if reverse_modes:
        modes.reverse()
    manifest_path = Path(str(plan.path) + ".plan.json")
    paged = manifest_path.exists() and json.loads(manifest_path.read_text()).get("pg") == "pages"
    measured = {}
    # Each candidate is timed on the L2 event executor. L1 is an explicit ablation.
    instance = plan.create(batch, {name: tensor.data_ptr()
                                   for name, tensor in buffers.items()},
                           torch.cuda.current_device())
    try:
        instance.set_steps([past] * (warmup + timed))
        for mode in modes:
            print(f"candidate mode {mode}: begin", flush=True)
            starts = [torch.cuda.Event(enable_timing=True)
                      for _ in range(timed)]
            ends = [torch.cuda.Event(enable_timing=True)
                    for _ in range(timed)]
            try:
                if loop and plan.info.phase == 1:
                    if not instance.loop_modes() & mode:
                        raise RuntimeError("requested candidate loop is unavailable")
                    if warmup:
                        instance.launch_steps(0, warmup, mode, stream.cuda_stream)
                    starts[0].record(stream)
                    instance.launch_steps(warmup, timed, mode, stream.cuda_stream)
                    ends[0].record(stream)
                    stream.synchronize()
                    values = [starts[0].elapsed_time(ends[0]) / timed]
                else:
                    for step in range(warmup):
                        instance.launch(step, mode, stream.cuda_stream)
                    for step in range(timed):
                        starts[step].record(stream)
                        instance.launch(warmup + step, mode, stream.cuda_stream)
                        ends[step].record(stream)
                    stream.synchronize()
                    values = [a.elapsed_time(b) for a, b in zip(starts, ends)]
            except BaseException:
                (out / "watchdog.json").write_text(json.dumps(instance.watchdog(), indent=2) + "\n")
                raise
            print(f"candidate mode {mode}: synchronized", flush=True)
            measured["L1" if mode == 1 else "L2"] = {
                "mean_ms": statistics.mean(values),
                "median_ms": statistics.median(values),
                "samples_ms": values,
                "decode_loop_used": bool(loop and plan.info.phase == 1),
            }
    finally:
        instance.close()
    if not _exclusive(out / "guard.jsonl", "candidate-after", False):
        raise SystemExit(75)
    report = {"plan": str(plan.path), "batch": batch, "past": past,
              "warmup": warmup, "timed": timed,
              "clocks_before": clocks_before, "clocks_after": _clocks(),
              "modes": measured}
    (out / "measurements.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--so", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--past-mid", type=int, default=575)
    parser.add_argument("--past-list", help="comma-separated past values; one new instance per value")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reverse-modes", action="store_true")
    parser.add_argument("--warmup", type=int, choices=range(0, 33))
    parser.add_argument("--timed", type=int, choices=range(1, 33))
    parser.add_argument("--mode", choices=("L1", "L2"))
    parser.add_argument("--loop", type=int, choices=(0,1), default=0)
    parser.add_argument("--loop-steps", type=int)
    parser.add_argument("--guard-wait-s", type=int, default=1800)
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    lock_path = Path(os.environ.get("TILEMEGA_GPU_LOCK", "/root/r10_work/serving_gpu.lock"))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        if os.environ.get("TILEMEGA_GPU_LOCK_HELD") != "1":
            fcntl.flock(lock, fcntl.LOCK_EX)
        # cuda.init alone does not necessarily register this PID with NVML.
        # Keep a tiny allocation alive so the exclusivity guard can require
        # our own process to appear in the visible compute-app table.
        guard_allocation = torch.empty(1, device="cuda")
        config = json.loads((args.model / "config.json").read_text())
        if args.past_list:
            pasts = [int(value) for value in args.past_list.split(",")]
            if not pasts or any(past < 0 for past in pasts):
                raise ValueError("invalid --past-list")
            by_mode = {}
            for past in pasts:
                one = measure_one(PlanLibrary(args.so), args.batch, config["vocab_size"],
                                  past, args.out / f"past{past}", args.reverse_modes,
                                  args.warmup, args.timed, args.mode or "L2",
                                  args.guard_wait_s, args.smoke_steps, args.loop, args.loop_steps)
                for mode, values in one["modes"].items():
                    by_mode.setdefault(mode, {"by_past": {}})["by_past"][str(past)] = values
            report = dict(plan=str(args.so), batch=args.batch, modes=by_mode)
            (args.out / "measurements.json").write_text(json.dumps(report, indent=2)+"\n")
        else:
            report = measure_one(PlanLibrary(args.so), args.batch,
                                 config["vocab_size"], args.past_mid, args.out,
                                 args.reverse_modes, args.warmup, args.timed,
                                 args.mode or "L2", args.guard_wait_s, args.smoke_steps, args.loop, args.loop_steps)
        del guard_allocation
    print(json.dumps({mode: data.get("mean_ms", data.get("by_past"))
                      for mode, data in report["modes"].items()}))


if __name__ == "__main__":
    main()
