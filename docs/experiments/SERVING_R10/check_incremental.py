#!/usr/bin/env python3
"""Compare raw full and incremental Level 1 scores for random class vectors."""

from __future__ import annotations

import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent / "incremental_equivalence"


def evaluations(path: Path) -> list[tuple[str, float, str]]:
    result = []
    for line in path.read_text().splitlines():
        fields = line.split("\t")
        if fields[0] == "EVALUATE":
            result.append((fields[2], float(fields[3]), fields[6] if len(fields) > 6 else ""))
    return result


def main() -> int:
    report = {}
    failed = False
    for model in ("llama", "qwen"):
        cases = json.loads((HERE / f"{model}_cases.json").read_text())["cases"]
        full = evaluations(HERE / f"{model}_full.search.tsv")
        reuse = evaluations(HERE / f"{model}_reuse.search.tsv")
        valid = 0
        maximum = 0.0
        for index, (left, right) in enumerate(zip(full, reuse)):
            if left[0] != right[0] or left[2] != right[2]:
                failed = True
                print(f"{model} case={index} FAIL key/error mismatch")
                continue
            if left[2] or not math.isfinite(left[1]) or not math.isfinite(right[1]):
                print(f"{model} case={index} FAIL evaluation error: {left[2]}")
                failed = True
                continue
            relative = abs(left[1] - right[1]) / max(abs(left[1]), 1.0)
            maximum = max(maximum, relative)
            valid += 1
        passed = (len(cases) >= 20 and len(full) == len(cases) and
                  len(reuse) == len(cases) and valid >= 20 and maximum <= 1e-12)
        failed |= not passed
        report[model] = {"cases": len(cases), "full_rows": len(full),
                         "incremental_rows": len(reuse), "valid": valid,
                         "max_relative_error": maximum, "pass": passed}
        print(f"{model} {'PASS' if passed else 'FAIL'}: {valid}/{len(cases)} "
              f"valid, max_relative_error={maximum:.3g}")
    (HERE / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
