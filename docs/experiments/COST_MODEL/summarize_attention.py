#!/usr/bin/env python3
"""Paired statistics for the frozen, state-rotated attention experiment."""
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


def interval(values):
    rng = random.Random(20260910)
    draws = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(20000))
    return [draws[500], draws[19499]]


def signed_rank(values):
    ordered = sorted((abs(v), v > 0) for v in values if v)
    n = len(ordered)
    if not n:
        return 1.0
    positive = ties = 0
    i = 0
    while i < n:
        j = i + 1
        while j < n and ordered[j][0] == ordered[i][0]:
            j += 1
        positive += (i + 1 + j) / 2 * sum(sign for _, sign in ordered[i:j])
        ties += (j - i) ** 3 - (j - i)
        i = j
    variance = n * (n + 1) * (2 * n + 1) / 24 - ties / 48
    z = max(0, abs(positive - n * (n + 1) / 4) - .5) / math.sqrt(variance)
    return math.erfc(z / math.sqrt(2))


root = Path(__file__).resolve().parent / "attention_models"
rows = list(csv.DictReader((root / "correctness.tsv").open(), delimiter="\t"))
lookup = {}
for row in rows:
    model, seq, chunk, round_ = row["model"], int(row["seq"]), int(row["chunk"]), int(row["round"])
    key = model, seq, chunk, round_
    if key in lookup or row["status"] != "PASS":
        raise ValueError("duplicate or failed attention receipt")
    log = root / "logs" / f"{model}_c{chunk}_s{seq}_r{round_}.txt"
    if hashlib.sha256(log.read_bytes()).hexdigest() != row["log_sha256"]:
        raise ValueError("changed attention GPU log")
    lookup[key] = row
expected = {(m, s, c, r) for m in ("gqa2", "mha4") for s in (4, 128)
            for c in (1, 2, 4, 8) for r in range(50)}
if lookup.keys() != expected:
    raise ValueError("incomplete attention matrix")
results = []
for model in ("gqa2", "mha4"):
    for seq in (4, 128):
        for chunk in (2, 4, 8):
            for rounds in (25, 50):
                for metric in ("l05_ms", "l1_ms", "l2_ms"):
                    base = [float(lookup[model, seq, 1, r][metric]) for r in range(rounds)]
                    new = [float(lookup[model, seq, chunk, r][metric]) for r in range(rounds)]
                    differences = [b - a for a, b in zip(base, new)]
                    ratios = [b / a for a, b in zip(base, new)]
                    results.append(dict(model=model, seq=seq, chunk=chunk, rounds=rounds, metric=metric,
                        baseline_median_ms=statistics.median(base), chunk_median_ms=statistics.median(new),
                        paired_delta_ms=statistics.median(differences), delta_ci95=interval(differences),
                        paired_ratio=statistics.median(ratios), ratio_ci95=interval(ratios),
                        wilcoxon_normal_tie_corrected_p=signed_rank(differences)))
output = root / "paired_stats.json"
output.write_text(json.dumps(dict(correctness="800/800", primary_rounds=25,
    sensitivity_rounds=50, bootstrap_draws=20000, gpu_run=False, results=results), indent=2) + "\n")
print(f"PASS {len(results)} paired statistics; verified 800 frozen GPU logs")
