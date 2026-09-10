#!/usr/bin/env python3
"""Keep mixed-task CPU tests distinct from unimplemented fusion GPU arms."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[3]
output = args.out.resolve()
output.mkdir(exist_ok=False)
archive = repo / "docs/experiments/COST_MODEL/attention_models/correctness.tsv"
rows = list(csv.DictReader(archive.open(), delimiter="\t"))
tool = repo / "build-portable/tools/tilemega-fusion-cost"
(output / "manifest.json").write_text(json.dumps(dict(
    source=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
    binary_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),
    registers_source_sha256=hashlib.sha256(archive.read_bytes()).hexdigest()), indent=2)+"\n")
(output / "source.diff").write_bytes(subprocess.check_output(["git", "diff"], cwd=repo))
status = output / "status.txt"
status.write_text("RUNNING\n")
try:
    for model in ("gqa2", "mha4"):
        registers = {row["reg"] for row in rows if row["model"] == model and row["chunk"] == "1"}
        assert len(registers) == 1
        run = subprocess.run([str(tool), str(repo), model, registers.pop()], text=True,
                             capture_output=True, timeout=600)
        (output / (model+".tsv")).write_text(run.stdout)
        (output / (model+".txt")).write_text(run.stderr)
        run.check_returncode()
        assert "remaining=0" in run.stderr and "FUSION_FANOUT" in run.stderr
        print(model, "mixed phase prices and replication passed", flush=True)
    with (output / "gemm_prices.tsv").open("w") as prices, (output / "gemm_gate.txt").open("w") as diagnostics:
        gate = subprocess.run([str(tool.with_name("tilemega-task-cost-gate")), str(repo)], text=True,
                              stdout=prices, stderr=diagnostics, timeout=3600)
    gate.check_returncode()
    assert len(list(csv.DictReader((output / "gemm_prices.tsv").open(), delimiter="\t"))) == 4308
    gate_errors = (output / "gemm_gate.txt").read_text()
    assert gate_errors.count("status=PASS stage_entry_bits_equal=1") == 4
    assert "ISL_CONTEXT remaining=0" in gate_errors
    status.write_text("PASS production logical mixed prices; counterfactual fanout; 4308 GEMM bit groups; no GPU run\n")
except BaseException as error:
    status.write_text(f"STOP {error}\n")
    raise
