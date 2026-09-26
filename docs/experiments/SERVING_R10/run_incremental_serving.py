#!/usr/bin/env python3
"""Recheck full and incremental Level 1 preparation on serving exports."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = HERE / "incremental_serving"
WORK = Path("/root/r10_work/incremental_serving")
TARGET = HERE / "calibration/target_serving.json"


def evaluations(path: Path) -> list[tuple[str, float, str]]:
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split("\t")
        if fields[0] == "EVALUATE":
            rows.append((fields[2], float(fields[3]),
                         fields[6] if len(fields) > 6 else ""))
    return rows


def main() -> int:
    global OUT, WORK
    parser=argparse.ArgumentParser()
    parser.add_argument("--compiler",type=Path,default=ROOT / "build-portable/tools/tilemega-compile")
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--work",type=Path,default=WORK)
    args=parser.parse_args()
    OUT,WORK=args.out,args.work
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    report = {}
    failed = False
    for model, batch in (("llama", 1), ("qwen3", 16)):
        source_cases = HERE / "incremental_equivalence" / (
            ("qwen" if model == "qwen3" else model) + "_cases.json")
        cases = json.loads(source_cases.read_text())["cases"]
        all_rows = {}
        for enabled, arm in ((0, "full"), (1, "reuse")):
            prefix = WORK / f"{model}_{arm}.cu"
            command = [
                str(args.compiler),
                f"/root/r10_work/export/{model}_decode/bridge.json",
                str(prefix), "--serving", "decode", "--batch", str(batch),
                "--past-range", "64:1086", "--capacity", "1088",
                "--solver", "skeleton", "--solve", str(TARGET),
                "--hop-curve", str(ROOT / "docs/experiments/SIMULATOR/hop_ns.tsv"),
                "--variant-cache", "/root/r9_work/variant_resources",
                "--emit", "serving", "--flow-search-only", "1",
                "--evaluate-configs", str(source_cases),
                "--incremental-prepare", str(enabled),
            ]
            start = time.perf_counter()
            process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            duration = time.perf_counter() - start
            (OUT / f"{model}_{arm}.stdout").write_text(process.stdout)
            (OUT / f"{model}_{arm}.stderr").write_text(process.stderr)
            (OUT / f"{model}_{arm}.command.json").write_text(
                json.dumps({"command": command, "seconds": duration,
                            "returncode": process.returncode}, indent=2) + "\n")
            if process.returncode:
                print(f"{model} {arm} FAIL: compiler returned {process.returncode}")
                failed = True
                continue
            raw = Path(str(prefix) + ".search.tsv")
            copied = OUT / f"{model}_{arm}.search.tsv"
            shutil.copyfile(raw, copied)
            all_rows[arm] = evaluations(copied)
        full = all_rows.get("full", [])
        reuse = all_rows.get("reuse", [])
        valid = 0
        maximum = 0.0
        for left, right in zip(full, reuse):
            if left[0] != right[0] or left[2] or right[2]:
                continue
            if not math.isfinite(left[1]) or not math.isfinite(right[1]):
                continue
            valid += 1
            maximum = max(maximum, abs(left[1] - right[1]) /
                          max(abs(left[1]), 1.0))
        passed = len(cases) >= 20 and len(full) == len(cases) and len(reuse) == len(cases)
        passed &= valid >= 20 and maximum <= 1e-12
        report[model] = {"cases": len(cases), "full_rows": len(full),
                         "reuse_rows": len(reuse), "valid": valid,
                         "max_relative_error": maximum, "pass": passed}
        print(f"{model} {'PASS' if passed else 'FAIL'}: {valid}/{len(cases)}, "
              f"maximum relative difference {maximum:.3g}")
        failed |= not passed
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
