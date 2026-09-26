#!/usr/bin/env python3
"""Recompute serving TaskBody fit errors from raw cycle measurements."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, median


HERE = Path(__file__).resolve().parent
FIT = json.loads((HERE / "target_serving.json").read_text())[
    "calibration_by_dtype"]["bf16"]["task_body"]["serving"]
by_kind: dict[str, list[dict[str, float]]] = {}
with (HERE / "serving_task_bodies.tsv").open() as stream:
    for row in csv.DictReader(stream, delimiter="\t"):
        kind = row["kind"]
        sample = FIT[kind]
        actual = float(row["body_ns"])
        predicted = (sample["fixed_ns"] + sample["byte_ns"] * float(row["bytes"])
                     + sample["flop_ns"] * float(row["flops"]))
        by_kind.setdefault(kind, []).append({"actual_ns": actual,
            "predicted_ns": predicted,
            "relative_error": abs(predicted - actual) / actual})

report = {}
for kind, samples in sorted(by_kind.items()):
    errors = [item["relative_error"] for item in samples]
    report[kind] = {"samples": len(samples), "median_relative_error": median(errors),
                    "mean_relative_error": mean(errors), "max_relative_error": max(errors),
                    "fixed_ns": FIT[kind]["fixed_ns"],
                    "byte_ns": FIT[kind]["byte_ns"],
                    "flop_ns": FIT[kind]["flop_ns"]}
    if len(samples) != FIT[kind]["samples"] or len(samples) > 12:
        raise SystemExit(f"{kind}: sample count mismatch or exceeds 12")

(HERE / "serving_task_bodies_errors.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n")
for kind, item in report.items():
    print(f"{kind}\t{item['samples']}\t{item['median_relative_error']:.6f}"
          f"\t{item['mean_relative_error']:.6f}\t{item['max_relative_error']:.6f}")
