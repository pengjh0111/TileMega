#!/usr/bin/env python3
"""Archive logical fusion candidate legality without claiming GPU lowering."""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--build", default="build-portable")
args = parser.parse_args()
repo = Path(__file__).resolve().parents[3]
output = Path(__file__).resolve().parent / "logical_candidates"
output.mkdir(exist_ok=True)
summary = {}
for model in ("gqa2", "mha4"):
    command = [str(repo / args.build / "tools/tilemega-fusion-probe"),
               str(repo / f"docs/experiments/SEQSCAN/raw/export/{model}.json")]
    result = subprocess.run(command, text=True, capture_output=True)
    (output / f"{model}.tsv").write_text(result.stdout)
    (output / f"{model}.txt").write_text(result.stderr)
    result.check_returncode()
    rows = list(csv.DictReader(io.StringIO(result.stdout), delimiter="\t"))
    legal = [row for row in rows if row["status"] == "LEGAL_L_TASK"]
    existing = [row for row in rows if row["status"] == "EXISTING_RUNTIME_FUSION"]
    assert legal, "production logical graph has no verified new candidate"
    assert all(row["conservation"] == "1" for row in legal + existing)
    assert "remaining=0" in result.stderr
    summary[model] = dict(new_logical_candidates=len(legal), existing_runtime_fusions=len(existing),
                          rejected=sum(row["status"] == "REJECT" for row in rows),
                          gpu_lowering_verified=False)
command = [str(repo / args.build / "tools/tilemega-scalar-error-probe"), str(repo)]
result = subprocess.run(command, text=True, capture_output=True)
(output / "errors.txt").write_text(result.stdout + result.stderr)
result.check_returncode()
assert "SCALAR_ERRORS branches=23 reference_delta=0" in result.stdout
(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, sort_keys=True))
