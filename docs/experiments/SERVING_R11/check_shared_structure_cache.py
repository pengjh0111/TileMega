#!/usr/bin/env python3
"""Compare exact PG-1 flow scores with and without cross-structure caches."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
R10 = ROOT / "docs/experiments/SERVING_R10/incremental_equivalence"
TARGET = ROOT / "docs/experiments/SERVING_R11/r10_control/calibration/target_serving.json"


def scores(path: Path) -> list[tuple[str, float, str]]:
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split("\t")
        if fields[0] == "EVALUATE":
            rows.append((fields[2], float(fields[3]), fields[6]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", type=Path,
                        default=Path("/root/r11_work/build/tools/tilemega"))
    parser.add_argument("--work", type=Path,
                        default=Path("/root/r11_work/shared_structure_cache"))
    parser.add_argument("--out", type=Path,
                        default=ROOT / "docs/experiments/SERVING_R11/solver/shared_structure_cache")
    parser.add_argument("--models", default="llama,qwen3")
    parser.add_argument("--qwen-cases", type=int, default=7)
    parser.add_argument("--partial-full-search", type=Path,
                        help="Completed prefix of an interrupted full-control search")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "report.json"
    result = json.loads(report_path.read_text()) if report_path.exists() else {}
    for model, batch in (("llama", 1), ("qwen3", 16)):
        if model not in args.models.split(","):
            continue
        filename = ("qwen" if model == "qwen3" else model) + "_cases.json"
        original = json.loads((R10 / filename).read_text())
        cases = original["cases"]
        if model == "qwen3":
            cases = cases[:args.qwen_cases]
            # R10's Qwen sample has a constant lm_head tile-N. Switch that
            # coordinate so the test really traverses the structural cache.
            for index, case in enumerate(cases):
                case["geometries"][-1]["tile_n"] = (32, 64, 128)[index % 3]
        case_file = args.work / f"{model}_cases.json"
        case_file.write_text(json.dumps({"cases": cases}, indent=2) + "\n")
        (args.out / f"{model}_cases.json").write_bytes(case_file.read_bytes())
        pair = {}
        for reuse, label in ((0, "full"), (1, "reuse")):
            prefix = args.work / f"{model}_{label}.cu"
            cmd = [str(args.compiler), "compile",
                   f"/root/r10_work/export/{model}_decode/bridge.json", str(prefix),
                   "--serving", "decode", "--batch", str(batch),
                   "--past-range", "64:1086", "--capacity", "1088",
                   "--solver", "skeleton", "--solve", str(TARGET),
                   "--hop-curve", str(ROOT / "docs/experiments/SIMULATOR/hop_ns.tsv"),
                   "--variant-cache", "/root/r9_work/variant_resources",
                   "--emit", "serving", "--sync", "legacy", "--pg", "pages",
                   "--page-bytes", "16384", "--flow-search-only", "1",
                   "--evaluate-configs", str(case_file),
                   "--incremental-prepare", str(reuse)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
            if model == "qwen3" and label == "full" and args.partial_full_search:
                search = args.partial_full_search
                elapsed = None
            else:
                start = time.monotonic()
                run = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
                elapsed = time.monotonic() - start
                (args.out / f"{model}_{label}.stdout").write_text(run.stdout)
                (args.out / f"{model}_{label}.stderr").write_text(run.stderr)
                if run.returncode:
                    raise RuntimeError(f"{model} {label} failed: {run.stderr[-1000:]}")
                search = Path(str(prefix) + ".search.tsv")
            (args.out / f"{model}_{label}.search.tsv").write_bytes(search.read_bytes())
            pair[label] = dict(rows=scores(search), seconds=elapsed, command=cmd)
        full, reuse = pair["full"]["rows"], pair["reuse"]["rows"]
        if len(full) != len(cases) or len(reuse) != len(cases):
            raise AssertionError(f"{model}: incomplete score comparison")
        maximum = 0.0
        for index, (left, right) in enumerate(zip(full, reuse)):
            if left[0] != right[0] or left[2] or right[2] or not all(
                    math.isfinite(row[1]) for row in (left, right)):
                raise AssertionError(f"{model}: invalid case {index}: {left} {right}")
            maximum = max(maximum, abs(left[1] - right[1]) / max(abs(left[1]), 1))
        result[model] = dict(cases=len(cases), maximum_relative_error=maximum,
                             full_seconds=pair["full"]["seconds"],
                             reuse_seconds=pair["reuse"]["seconds"],
                             full_search_is_completed_prefix=bool(model == "qwen3" and
                                 args.partial_full_search),
                             structure_tile_ns=sorted({case["geometries"][-1]["tile_n"]
                                                        for case in cases}),
                             commands={label: pair[label]["command"] for label in pair})
        report_path.write_text(json.dumps(result, indent=2) + "\n")
        print(model, json.dumps(result[model]), flush=True)
        if maximum > 1e-12:
            raise AssertionError(f"{model}: cache changed a Level 1 score")


if __name__ == "__main__":
    main()
