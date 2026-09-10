#!/usr/bin/env python3
"""Recompute ranks from recorded predictions, without SciPy or tolerances."""
import csv
import hashlib
import json
import math
from pathlib import Path


def ranks(values):
    indices = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    begin = 0
    while begin < len(indices):
        end = begin + 1
        while end < len(indices) and values[indices[end]] == values[indices[begin]]:
            end += 1
        for i in indices[begin:end]:
            result[i] = (begin + end - 1) / 2
        begin = end
    return result


def score(rows):
    measured = [float(row["measured_ms"]) for row in rows]
    predicted = [float(row["model_ms"]) for row in rows]
    a, b = ranks(measured), ranks(predicted)
    mean = (len(rows) - 1) / 2
    rho = sum((x - mean) * (y - mean) for x, y in zip(a, b)) / math.sqrt(
        sum((x - mean) ** 2 for x in a) * sum((x - mean) ** 2 for x in b))
    oracle = set(sorted(range(len(rows)), key=measured.__getitem__)[:math.ceil(len(rows) * .03)])
    order = sorted(range(len(rows)), key=predicted.__getitem__)
    return {"rho_from_printed_predictions": rho, "top1_top3_top10": [
        sum(i in oracle for i in order[:k]) for k in (1, 3, 10)]}


def main():
    base = Path(__file__).resolve().parent
    result = {"note": "BF16 comparison is conditional on historical PASS subset; not a new GPU oracle.", "cells": []}
    for dtype in ("bf16", "f32"):
        for model in ("gqa2", "mha4"):
            paths = [base / "partial_rate_comparison" / (dtype + "_measured") / f"predictions_{model}.tsv",
                     base / "unified_rank" / dtype / f"predictions_{model}.tsv"]
            old, new = [list(csv.DictReader(path.open(), delimiter="\t")) for path in paths]
            keys = ("tile_m", "tile_n", "tile_k", "stages", "split_k", "ctas_per_sm", "measured_ms")
            if len(old) != len(new) or any(any(a[key] != b[key] for key in keys) for a, b in zip(old, new)):
                raise RuntimeError("configuration or measured-data mismatch")
            before, after = score(old), score(new)
            if dtype == "f32" and (after["rho_from_printed_predictions"] < before["rho_from_printed_predictions"]
                                    or any(a < b for a, b in zip(after["top1_top3_top10"], before["top1_top3_top10"]))):
                raise RuntimeError("FP32 ranking regression")
            changes = [float(b["model_ms"]) - float(a["model_ms"]) for a, b in zip(old, new)]
            result["cells"].append({"dtype": dtype, "model": model, "configs": len(old), "old": before, "unified": after,
                                    "delta_ms_min_max": [min(changes), max(changes)],
                                    "inputs": [{"path": str(p.relative_to(base)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]})
    output = base / "unified_rank" / "verification.json"
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
