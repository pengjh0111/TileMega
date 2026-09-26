#!/usr/bin/env python3
"""Compare all serving prefill attention tilings with a PyTorch BF16 oracle."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, default=ROOT / "docs/experiments/SERVING_R10/task_body_tests/prefill_attention")
parser.add_argument("--binary", type=Path, default=ROOT / "build-portable/serving_prefill_attention_test")
args = parser.parse_args()
OUT = args.out
OUT.mkdir(parents=True, exist_ok=True)
binary = args.binary
run = subprocess.run([str(binary), "--dump-dir", str(OUT)], capture_output=True,
                     text=True)
(OUT / "cuda_stdout.txt").write_text(run.stdout)
(OUT / "cuda_stderr.txt").write_text(run.stderr)
if run.returncode:
    raise SystemExit(f"CUDA prefill probe failed: {run.returncode}")

results = []
for d, q in ((64, 4), (128, 2)):
    width = (q + 2) * d
    t = torch.arange(64).view(64, 1)
    i = torch.arange(width).view(1, width)
    qkv = (((t * 19 + i * 7) % 43 - 21).float() / 256).to(torch.bfloat16)
    qkv = qkv.reshape(64, q + 2, d)
    qweight = (0.75 + (torch.arange(d) % 7).float() / 32).to(torch.bfloat16)
    kweight = (0.875 + (torch.arange(d) % 5).float() / 32).to(torch.bfloat16)

    def normalized(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        inverse = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return ((x.float() * inverse).to(torch.bfloat16).float()
                * weight.float()).to(torch.bfloat16)

    for norm in (False, True):
        queries = qkv[:, :q, :]
        keys = qkv[:, q, :]
        if norm:
            queries = normalized(queries, qweight)
            keys = normalized(keys, kweight)
        values = qkv[:, q + 1, :]
        expected = torch.empty((64, q, d), dtype=torch.float32)
        for token in range(64):
            score = (queries[token].float() @ keys[:token + 1].float().T) / math.sqrt(d)
            raw = torch.exp(score - score.max(-1, keepdim=True).values)
            numerator = raw.to(torch.bfloat16).float() @ values[:token + 1].float()
            expected[token] = numerator / raw.sum(-1, keepdim=True)
        for rows in (16, 64):
            path = OUT / f"D{d}_Q{q}_R{rows}_N{int(norm)}.bin"
            actual = torch.frombuffer(bytearray(path.read_bytes()),
                                      dtype=torch.bfloat16).reshape(64, q, d).float()
            delta = (actual - expected).abs()
            limit = expected.abs() / 128 + 2 ** -8
            violations = int((delta > limit).sum())
            item = {"head_dim": d, "qperkv": q, "query_rows": rows,
                    "qk_norm": norm, "elements": actual.numel(),
                    "violations": violations, "max_absolute_error": delta.max().item(),
                    "max_allowed_error": limit.max().item(), "dump": str(path)}
            results.append(item)
            print(json.dumps(item))
(OUT / "torch_comparison.json").write_text(json.dumps(results, indent=2) + "\n")
if any(item["violations"] for item in results):
    raise SystemExit("PyTorch prefill attention comparison failed")
