#!/usr/bin/env python3
"""Independent plan and solver-control coverage checks."""
import csv
import hashlib
import json
from pathlib import Path

base = Path(__file__).resolve().parent / "unified_solver"
result = {"plan_checks": [], "repeat_price_fields": 0}
stable = ("model", "mode", "configs", "ctas_per_sm", "measured_rank", "band", "measured_ms", "best_ms", "transitions")
for dtype in ("bf16", "f32"):
    old_dir, new_dir = base / (dtype + "_legacy"), base / dtype
    before, after = [list(csv.DictReader((directory / "summary.tsv").open(), delimiter="\t")) for directory in (old_dir, new_dir)]
    if len(before) != 8 or len(after) != 8:
        raise RuntimeError("incomplete two-model, four-mode solver evidence")
    for a, b in zip(before, after):
        if any(a[field] != b[field] for field in stable):
            raise RuntimeError("unified solver changed the historical selection/control")
    for model in ("gqa2", "mha4"):
        old, new = [directory / f"plan_{model}.tsv" for directory in (old_dir, new_dir)]
        if old.read_bytes() != new.read_bytes():
            raise RuntimeError("per-GEMM configurations changed")
        result["plan_checks"].append({"dtype": dtype, "model": model, "equal": True,
                                      "sha256": hashlib.sha256(new.read_bytes()).hexdigest()})
    outcome = "PASS" if dtype == "f32" else "FAIL"
    for directory in (old_dir, new_dir):
        if f"models: {outcome}" not in (directory / "run.txt").read_text():
            raise RuntimeError("rank acceptance status missing or relabeled")

original, repeat = [list(csv.DictReader((base / directory / "summary.tsv").open(), delimiter="\t"))
                    for directory in ("f32", "f32_audited")]
if len(original) != len(repeat):
    raise RuntimeError("incomplete audited repeat")
for a, b in zip(original, repeat):
    if any(a[field] != b[field] for field in (*stable, "predicted_ms")):
        raise RuntimeError("audited repeat changed a non-timing field")
    result["repeat_price_fields"] += 1
if "ISL_CONTEXT remaining=0" not in (base / "f32_audited/run.txt").read_text():
    raise RuntimeError("audited repeat lacks zero-reference evidence")
result["scope"] = "CPU plans and printed price fields; BF16 old/new ranking acceptance remains FAIL"
with (base / "verification.json").open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(json.dumps(result, indent=2))
