"""Check complete L1/L2 token equality on one plan instance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .engine import ServingEngine
from .measure import _exclusive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prefill-so", type=Path, required=True)
    parser.add_argument("--decode-so", type=Path, required=True)
    parser.add_argument("--prompt-ids", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.prompt_ids.read_text())
    prompts = torch.tensor(rows[:args.batch], dtype=torch.int32)
    with ServingEngine(args.model, args.prefill_so, args.decode_so,
                       args.batch, mode="L1") as engine:
        results = {}
        for mode in ("L1", "L2"):
            if not _exclusive(args.out / "guard.jsonl", mode + "-before", True):
                raise RuntimeError("GPU was occupied before the mode check")
            engine.prefill_mode = engine.decode_mode = {"L1": 1, "L2": 2}[mode]
            generation = engine.generate(prompts, 1024)
            if not _exclusive(args.out / "guard.jsonl", mode + "-after", False):
                raise RuntimeError("GPU was occupied during the mode check")
            results[mode] = generation.tokens
            (args.out / f"tokens_{mode}.json").write_text(
                json.dumps(generation.tokens.tolist(), separators=(",", ":")) + "\n")
        mismatch = int((results["L1"] != results["L2"]).sum().item())
    report = {"batch": args.batch, "tokens": args.batch * 1024,
              "mismatches": mismatch, "pass": mismatch == 0,
              "same_plan_instances": True}
    (args.out / "mode_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if mismatch:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
