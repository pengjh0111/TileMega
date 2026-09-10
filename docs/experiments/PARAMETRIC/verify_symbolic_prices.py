#!/usr/bin/env python3
"""Archive exact task prices and the real CG-interface degree stop separately."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[3]
out = args.out.resolve()
out.mkdir(exist_ok=False)
tool = repo / "build-portable/tools/tilemega-symbolic-task-price"
receipts = repo / "docs/experiments/COST_MODEL/attention_models/correctness.tsv"
resources = list(csv.DictReader(receipts.open(), delimiter="\t"))
(out / "manifest.json").write_text(json.dumps(dict(
    source=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
    binary_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),
    resources_sha256=hashlib.sha256(receipts.read_bytes()).hexdigest()), indent=2)+"\n")
(out / "source.diff").write_bytes(subprocess.check_output(["git", "diff"], cwd=repo))
status = out / "status.txt"
status.write_text("RUNNING\n")
summary = []
try:
    for model in ("gqa2", "mha4"):
        for mode, extent in (("collective", 512), ("scalar", 16), ("dp", 16)):
            command = [str(tool), str(repo), model, mode, str(extent)]
            if mode == "dp":
                registers = {r["reg"] for r in resources if r["model"] == model and r["chunk"] == "1"}
                assert len(registers) == 1
                command.append(registers.pop())
            run = subprocess.run(command, text=True, capture_output=True, timeout=600,
                                 env=dict(os.environ, TILEMEGA_ISL_AUDIT="1"))
            prefix = model+"_"+mode
            (out / (prefix+".tsv")).write_text(run.stdout)
            (out / (prefix+".txt")).write_text(run.stderr)
            assert "ISL_CONTEXT remaining=0" in run.stderr
            if mode == "dp":
                assert run.returncode == 2 and "degree exceeds two" in run.stderr
                assert "CG interface" in run.stderr and "^3" in run.stderr
            else:
                run.check_returncode()
                assert "SYMBOLIC_"+mode.upper() in run.stderr
            summary.append(dict(model=model, mode=mode, seq_end=extent, exit=run.returncode,
                                result="degree_stop" if mode == "dp" else "PASS"))
            print(prefix, summary[-1]["result"], flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    status.write_text("PASS task-price controls; STOP B3.2 cubic CG-interface price; not a completed symbolic DP\n")
except BaseException as error:
    status.write_text(f"STOP runner: {error}\n")
    raise
