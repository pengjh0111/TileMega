#!/usr/bin/env python3
"""Paired CPU ranking audit; archived measurements are never regenerated."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    base = Path(__file__).resolve().parent
    module = importlib.util.spec_from_file_location(
        "rank_audit", base.parent / "COST_MODEL/verify_unified_rank.py")
    rank = importlib.util.module_from_spec(module)
    module.loader.exec_module(rank)
    root = base / "cache_curve"
    result = {"scope": "CPU prediction comparison on archived measured subsets; no GPU ranking claim",
              "cells": []}
    deltas = []
    keys = ("tile_m", "tile_n", "tile_k", "stages", "split_k", "ctas_per_sm", "measured_ms")
    for dtype in ("bf16", "f32"):
        for model in ("gqa2", "mha4"):
            paths = [root / (dtype + "_" + arm) / ("predictions_" + model + ".tsv")
                     for arm in ("cdf", "curve")]
            rows = [list(csv.DictReader(path.open(), delimiter="\t")) for path in paths]
            if len(rows[0]) != len(rows[1]) or not rows[0]:
                raise RuntimeError("incomplete paired configurations")
            for old, new in zip(*rows):
                if any(old[key] != new[key] for key in keys):
                    raise RuntimeError("changed configuration or measured observation")
                deltas.append(dict(dtype=dtype, model=model, **{k: old[k] for k in keys},
                                   cdf_ms=old["model_ms"], curve_ms=new["model_ms"],
                                   delta_ms=float(new["model_ms"])-float(old["model_ms"])))
            baseline = base.parent / "COST_MODEL/unified_rank" / dtype / paths[0].name
            if baseline.read_bytes() != paths[0].read_bytes():
                raise RuntimeError("CDF control differs from the recorded A6 baseline")
            scores = [rank.score(r) for r in rows]
            result["cells"].append(dict(dtype=dtype, model=model, configs=len(rows[0]),
                cdf=scores[0], curve=scores[1], cdf_baseline_byte_equal=True,
                rho_regression=scores[1]["rho_from_printed_predictions"] < scores[0]["rho_from_printed_predictions"],
                inputs=[dict(path=str(p.relative_to(base)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]))
        for arm in ("cdf", "curve"):
            log = (root / (dtype + "_" + arm) / "run.txt").read_text()
            if "ISL_CONTEXT remaining=0" not in log:
                raise RuntimeError("missing explicit zero-reference evidence")
    with (root / "pointwise_deltas.tsv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=deltas[0], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(deltas)
    with (root / "verification.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
