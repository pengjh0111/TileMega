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


def serving_trace(args):
    """Launch-local stamps; every past uses a fresh instance and fresh counters."""
    reports = []
    os.environ["TILEMEGA_TRACE_LAUNCHES"] = str(max(64, args.launches * args.steps))
    for past in ([int(v) for v in args.past_list.split(",")] if args.past_list else [args.past]):
        out = args.out / f"past{past}"
        out.mkdir(parents=True, exist_ok=True)
        with ServingEngine(args.model, args.prefill_so, args.decode_so, args.batch,
                           mode=args.mode, decode_loop=bool(args.decode_loop),
                           prefill_mode="L1") as engine:
            dump = getattr(engine.decode_lib.lib, "tm_plan_dump_serving_trace", None)
            if dump is None:
                raise RuntimeError("decode plan lacks stage/step trace instrumentation")
            if not _exclusive(out / "guard.jsonl", "before", True):
                raise SystemExit(75)
            engine.state.tokens.zero_()
            engine.state.kv_storage.zero_()
            stream = torch.cuda.current_stream()
            first = past - engine.prompt_len
            if first < 0 or first + args.steps > 1023:
                raise ValueError("trace past interval is outside the step ring")
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            mode = 1 if args.mode == "L1" else 2
            if args.decode_loop and not engine.loop_modes & mode:
                raise RuntimeError("requested trace loop is unavailable")
            begin.record(stream)
            for _ in range(args.launches):
                if args.decode_loop:
                    engine.decode.launch_steps(first, args.steps, mode, stream.cuda_stream)
                else:
                    for step in range(first, first + args.steps):
                        engine.decode.launch(step, mode, stream.cuda_stream)
            end.record(stream)
            end.synchronize()
            dump.argtypes = [C.c_void_p, C.c_char_p]
            dump.restype = C.c_int
            status = dump(engine.decode.handle, os.fsencode(out.resolve()))
            if status:
                raise RuntimeError(f"serving trace dump failed: {status}")
            if not _exclusive(out / "guard.jsonl", "after", False):
                raise SystemExit(75)
            report = dict(past=past, mode=args.mode, loop=bool(args.decode_loop),
                          launches=args.launches, steps=args.steps,
                          mean_step_ms=begin.elapsed_time(end)/(args.launches*args.steps))
            (out / "trace.json").write_text(json.dumps(report, indent=2)+"\n")
            reports.append(report)
    print(json.dumps(reports))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prefill-so", type=Path, required=True)
    parser.add_argument("--decode-so", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--past", type=int, default=575)
    parser.add_argument("--launches", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--step", action="store_true")
    parser.add_argument("--past-list")
    parser.add_argument("--mode", choices=("L1", "L2"), default="L2")
    parser.add_argument("--decode-loop", type=int, choices=(0, 1), default=0)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage or args.step:
        serving_trace(args)
        return
    os.environ["TILEMEGA_TRACE_V2"] = "1"
    os.environ["TILEMEGA_TRACE_V2_OUT"] = str(args.out.resolve())
    with ServingEngine(args.model, args.prefill_so, args.decode_so,
                       args.batch) as engine:
        if not hasattr(engine.decode_lib.lib, "tm_plan_dump_trace_v2"):
            raise RuntimeError("decode plan is not a trace-enabled build")
        if not _exclusive(args.out / "guard.jsonl", "before", True):
            raise SystemExit(75)
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
            raise SystemExit(75)
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
