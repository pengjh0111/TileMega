#!/usr/bin/env python3
"""CPU fusion interval/projection evidence, not fused GPU acceptance."""
import argparse
import csv
import hashlib
import io
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
status = out / "status.txt"
status.write_text("RUNNING\n")
env = dict(os.environ, TILEMEGA_ISL_AUDIT="1")
records = []
try:
    for model in ("gqa2", "mha4"):
        for seq in (4, 128):
            for mode in ("events", "dp"):
                tool = repo / "build-portable/tools" / ("tilemega-fusion-"+mode)
                source = repo / "docs/experiments/SEQSCAN/raw/export" / (model+".json")
                target = repo / "configs/targets/sm_89.json"
                command = [str(tool), str(source), str(target)]
                files = [tool, source, target]
                if mode == "dp":
                    log = repo / "docs/experiments/OCCUPANCY/raw/ptxas" / (model+"_occ1_full.build")
                    command.append(str(log))
                    files.append(log)
                command += [str(seq), "3"]
                prefix = f"{model}_{seq}_{mode}"
                (out / (prefix+"_inputs.json")).write_text(json.dumps({
                    "command": command,
                    "sha256": {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                }, indent=2)+"\n")
                run = subprocess.run(command, cwd=repo, env=env, text=True, capture_output=True, timeout=3600)
                (out / (prefix+".tsv")).write_text(run.stdout)
                (out / (prefix+".txt")).write_text(run.stderr)
                run.check_returncode()
                assert "ISL_CONTEXT remaining=0" in run.stderr
                rows = list(csv.DictReader(io.StringIO(run.stdout), delimiter="\t"))
                assert rows
                if mode == "events":
                    assert all(float(row["delta_ns"]) < 0 for row in rows)
                else:
                    assert "baseline_bits_equal=1" in run.stderr
                    assert any(int(row["fusion_count"]) == 0 for row in rows)
                    assert any(int(row["fusion_count"]) > 0 for row in rows)
                records.append(dict(model=model, seq=seq, mode=mode, rows=len(rows)))
                print(prefix, len(rows), "PASS", flush=True)
    unit = subprocess.run([str(repo / "build-portable/fused_runtime_projection_test")],
                          env=env, text=True, capture_output=True, timeout=600)
    (out / "projection_unit.txt").write_text(unit.stdout+unit.stderr)
    unit.check_returncode()
    assert "checks=24 errors=16 remaining=0" in unit.stdout
    (out / "summary.json").write_text(json.dumps(records, indent=2)+"\n")
    status.write_text("PASS CPU fixed-implementation fusion intervals and event projection; GPU not verified\n")
except BaseException as error:
    status.write_text(f"STOP {error}\n")
    raise
