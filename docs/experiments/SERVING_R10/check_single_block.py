#!/usr/bin/env python3
"""Fresh-process regression for single-block attention dependency closure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


EXPECTED = [13, 578, 31657, 315, 264, 2363, 11, 439]


def child(args: argparse.Namespace) -> int:
    import torch
    from tilemega.serving.engine import ServingEngine

    prompts = torch.tensor(json.loads(args.prompts.read_text())[:1])
    with ServingEngine(args.model, args.prefill, args.decode, 1) as engine:
        engine.prefill_mode = engine.decode_mode = 1
        l1 = engine.generate(prompts, len(EXPECTED)).tokens[0].tolist()
        engine.prefill_mode = engine.decode_mode = 2
        l2 = engine.generate(prompts, len(EXPECTED)).tokens[0].tolist()
    print(json.dumps({"pid": os.getpid(), "L1": l1, "L2": l2,
                      "pass": l1 == l2 == EXPECTED}))
    return int(l1 != l2 or l1 != EXPECTED)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prefill", type=Path, required=True)
    parser.add_argument("--decode", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=50)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        return child(args)
    if args.processes < 50:
        raise ValueError("the repository race-evidence gate needs 50 processes")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, __file__, "--child", "--model", str(args.model),
               "--prefill", str(args.prefill), "--decode", str(args.decode),
               "--prompts", str(args.prompts), "--out", str(args.out)]
    results = []
    with args.out.open("w") as log:
        for _ in range(args.processes):
            run = subprocess.run(command, text=True, capture_output=True,
                                 timeout=120, env={**os.environ,
                                                   "PYTHONPATH": str(Path(__file__).resolve().parents[3] / "python")})
            lines = [line for line in run.stdout.splitlines()
                     if line.startswith('{"pid"')]
            result = json.loads(lines[-1]) if lines else {
                "pid": None, "pass": False, "returncode": run.returncode,
                "stderr": run.stderr[-2000:]}
            result["returncode"] = run.returncode
            results.append(result)
            log.write(json.dumps(result) + "\n")
            log.flush()
    passed = sum(bool(item["pass"]) and item["returncode"] == 0
                 for item in results)
    print(f"single-block fresh processes: {passed}/{len(results)}")
    return int(passed != len(results))


if __name__ == "__main__":
    raise SystemExit(main())
