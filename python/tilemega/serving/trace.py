"""Diagnostic trace of one solved decode plan at its middle past value."""
from __future__ import annotations

import argparse
import ctypes as C
import json
import os
from pathlib import Path

import torch

from .engine import ServingEngine
from .measure import _exclusive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prefill-so", type=Path, required=True)
    parser.add_argument("--decode-so", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--past", type=int, default=575)
    parser.add_argument("--launches", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["TILEMEGA_TRACE_V2"] = "1"
    os.environ["TILEMEGA_TRACE_V2_OUT"] = str(args.out.resolve())
    with ServingEngine(args.model, args.prefill_so, args.decode_so,
                       args.batch) as engine:
        if not hasattr(engine.decode_lib.lib, "tm_plan_dump_trace_v2"):
            raise RuntimeError("decode plan is not a trace-enabled build")
        if not _exclusive(args.out / "guard.jsonl", "before", True):
            raise RuntimeError("GPU remained occupied before serving trace")
        engine.state.tokens.zero_()
        engine.state.kv_storage.zero_()
        stream = torch.cuda.current_stream()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        step = args.past - engine.prompt_len
        if not 0 <= step < 1023:
            raise ValueError("past is outside the decode step ring")
        start.record(stream)
        for _ in range(args.launches):
            engine.decode.launch(step, 2, stream.cuda_stream)
        end.record(stream)
        end.synchronize()
        step_ms = start.elapsed_time(end) / args.launches
        dump = engine.decode_lib.lib.tm_plan_dump_trace_v2
        dump.argtypes = [C.c_void_p, C.c_float]
        dump.restype = C.c_int
        status = dump(engine.decode.handle, step_ms)
        if status:
            raise RuntimeError(f"tm_plan_dump_trace_v2 returned {status}")
        if not _exclusive(args.out / "guard.jsonl", "after", False):
            raise RuntimeError("GPU trace was contaminated")
        report = {"model": str(args.model), "batch": args.batch,
                  "past": args.past, "mode": "L2 diagnostic",
                  "launches": args.launches, "mean_step_ms": step_ms,
                  "trace_compiled": True,
                  "trace_files": ["meta.tsv", "slots.tsv", "waits.tsv",
                                  "events.tsv", "runtime_dependencies.cuh"]}
        (args.out / "trace.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))


if __name__ == "__main__":
    main()
