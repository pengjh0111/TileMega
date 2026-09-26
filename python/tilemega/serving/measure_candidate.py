"""Time top-three serving plans with finite synthetic device-resident data."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import statistics

import torch

from .measure import _clocks, _exclusive
from .plan import PlanLibrary


def _external_buffers(plan: PlanLibrary, batch: int,
                      vocab: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(20260925)
    result = {}
    dtypes = {0: torch.bfloat16, 1: torch.float32, 2: torch.int32}
    for buffer in plan.buffers:
        if buffer.role != 1:
            continue
        elements = buffer.elements_constant + batch * buffer.elements_per_batch
        if buffer.dtype == 2:
            tensor = torch.randint(0, vocab, (elements,), dtype=torch.int32,
                                   device="cuda")
        else:
            tensor = torch.randn(elements, dtype=torch.float32,
                                 device="cuda").mul_(0.02).to(dtypes[buffer.dtype])
        result[buffer.name] = tensor
    return result


def measure_one(plan: PlanLibrary, batch: int, vocab: int,
                past_mid: int, out: Path, reverse_modes: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    if not _exclusive(out / "guard.jsonl", "candidate-before", True):
        raise RuntimeError("GPU remained occupied for 30 minutes")
    buffers = _external_buffers(plan, batch, vocab)
    warmup, timed = (8, 32) if plan.info.phase == 1 else (2, 5)
    past = past_mid if plan.info.phase == 1 else 0
    stream = torch.cuda.current_stream()
    clocks_before = _clocks()
    modes = [mode for mode in (1, 2) if plan.info.modes & mode]
    if reverse_modes:
        modes.reverse()
    measured = {}
    # The two modes occupy disjoint event rows. Exercise both on one instance
    # so the candidate timing also checks their independent ticket sequences.
    instance = plan.create(batch, {name: tensor.data_ptr()
                                   for name, tensor in buffers.items()},
                           torch.cuda.current_device())
    try:
        instance.set_steps([past] * (warmup + timed))
        for mode in modes:
            starts = [torch.cuda.Event(enable_timing=True)
                      for _ in range(timed)]
            ends = [torch.cuda.Event(enable_timing=True)
                    for _ in range(timed)]
            for step in range(warmup):
                instance.launch(step, mode, stream.cuda_stream)
            for step in range(timed):
                starts[step].record(stream)
                instance.launch(warmup + step, mode, stream.cuda_stream)
                ends[step].record(stream)
            stream.synchronize()
            values = [a.elapsed_time(b) for a, b in zip(starts, ends)]
            measured["L1" if mode == 1 else "L2"] = {
                "mean_ms": statistics.mean(values),
                "median_ms": statistics.median(values),
                "samples_ms": values,
            }
    finally:
        instance.close()
    if not _exclusive(out / "guard.jsonl", "candidate-after", False):
        raise RuntimeError("candidate GPU timing was contaminated")
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
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reverse-modes", action="store_true")
    args = parser.parse_args()
    lock_path = Path("/root/r10_work/serving_gpu.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        torch.cuda.init()
        config = json.loads((args.model / "config.json").read_text())
        report = measure_one(PlanLibrary(args.so), args.batch,
                             config["vocab_size"], args.past_mid, args.out,
                             args.reverse_modes)
    print(json.dumps({mode: data["mean_ms"]
                      for mode, data in report["modes"].items()}))


if __name__ == "__main__":
    main()
