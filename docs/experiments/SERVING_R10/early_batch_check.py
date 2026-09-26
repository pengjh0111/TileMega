#!/usr/bin/env python3
"""Short exploratory L1/L2 batch smoke; not the EV-1 C-2 gate."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

import torch

from tilemega.serving.engine import ServingEngine


HERE = Path(__file__).resolve().parent
WORK = Path("/root/r10_work/plans")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("llama", "qwen3"))
    parser.add_argument("--batch", required=True, type=int)
    parser.add_argument("--tokens", type=int, default=64)
    args = parser.parse_args()
    checkpoint = Path("/root/models") / (
        "llama3_2_1b" if args.model == "llama" else "qwen3_1_7b")
    ids = json.loads((HERE / "prompts" / f"{args.model}_ids.json").read_text())
    prompts = torch.tensor(ids[:args.batch], dtype=torch.int32)
    prefill = WORK / f"{args.model}_prefill_B{args.batch}" / "plan.so"
    decode = WORK / f"{args.model}_decode_B{args.batch}" / "plan.so"
    out = HERE / "early_batch_check"
    out.mkdir(parents=True, exist_ok=True)
    with Path("/root/r10_work/serving_gpu.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with ServingEngine(checkpoint, prefill, decode, args.batch,
                           max_new_tokens=args.tokens, mode="L1") as engine:
            first = engine.generate(prompts, args.tokens).tokens
            engine.prefill_mode = engine.decode_mode = 2
            second = engine.generate(prompts, args.tokens).tokens
            result = {"model": args.model, "batch": args.batch,
                      "tokens_per_request": args.tokens,
                      "compared_tokens": int(first.numel()),
                      "mismatches": int((first != second).sum().item()),
                      "first_row_prefix": first[0, :8].tolist(),
                      "prefill": str(prefill), "decode": str(decode)}
    (out / f"{args.model}_B{args.batch}_modes{args.tokens}.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if result["mismatches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
