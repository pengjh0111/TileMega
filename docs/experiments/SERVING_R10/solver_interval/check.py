#!/usr/bin/env python3
"""Recompute the restricted decode interval check from raw search logs."""

import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def scores(name):
    result = {}
    for line in (ROOT / name).read_text().splitlines():
        fields = line.split("\t")
        if fields[0] == "EVALUATE" and math.isfinite(float(fields[3])):
            result[fields[2]] = float(fields[3])
    return result


points = [scores(f"flow_b16_price_p{past}.cu.search.tsv")
          for past in (64, 575, 1086)]
interval = scores("flow_b16_interval.cu.search.tsv")
common = set(interval).intersection(*(set(point) for point in points))
differences = [abs(interval[key] -
                   (points[0][key] + 4 * points[1][key] + points[2][key]) / 6)
               for key in common]
changed = sum(points[0][key] != points[2][key] for key in common)
qwen = scores("flow_qwen_b16_p64.cu.search.tsv")
maximum = max(differences, default=float("inf"))
print(f"Llama B16 common={len(common)} changed_with_past={changed} "
      f"Simpson_max_abs_ns={maximum:.12g}; Qwen B16 past64_finite={len(qwen)}")
assert common and changed and maximum <= 1e-6 and qwen
