#!/usr/bin/env python3
"""Standalone selected L-task pass checks; no fused GPU claim."""
import argparse
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
tools = repo / "build-portable/tools"
env = dict(os.environ, TILEMEGA_ISL_AUDIT="1")
status = out / "status.txt"
status.write_text("RUNNING\n")
summary = []
try:
    for model in ("gqa2", "mha4"):
        source = repo / "docs/experiments/SEQSCAN/raw/export" / (model+".json")
        imported = subprocess.run([str(tools / "tilemega-import"), str(source)],
                                  capture_output=True, text=True, env=env, timeout=600)
        (out / (model+"_import.txt")).write_text(imported.stderr)
        imported.check_returncode()
        (out / (model+".mlir")).write_text(imported.stdout)
        for name, producer, consumer, success in (
            ("rope_kv", "l0.s05.rope", "l0.s06.append", True),
            ("nonadjacent", "l0.s00.norm", "l0.s06.append", False),
        ):
            command = [str(tools / "tilemega-opt"), str(out / (model+".mlir")),
                       f"--tilemega-fuse-task-pair=producer={producer} consumer={consumer}"]
            run = subprocess.run(command, text=True, capture_output=True, env=env, timeout=600)
            prefix = model+"_"+name
            (out / (prefix+".mlir")).write_text(run.stdout)
            (out / (prefix+".txt")).write_text(run.stderr)
            if success:
                run.check_returncode()
                assert "tilemega.fused_task_space" in run.stdout
                assert "tilemega.fusion_pending_lowering" in run.stdout
                verify = subprocess.run([str(tools / "tilemega-opt"), str(out / (prefix+".mlir"))],
                                        text=True, capture_output=True, env=env, timeout=600)
                (out / (prefix+"_verify.txt")).write_text(verify.stderr)
                verify.check_returncode()
                assert "ISL_CONTEXT remaining=0" in verify.stderr
            else:
                assert run.returncode != 0 and "adjacent logical" in run.stderr
            assert "ISL_CONTEXT remaining=0" in run.stderr
            summary.append(dict(model=model, case=name, exit=run.returncode))
            print(prefix, "verified" if success else "rejected", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    unit = subprocess.run([str(repo / "build-portable/fusion_rewrite_test")], text=True,
                          capture_output=True, env=env, timeout=600)
    (out / "transaction_and_incidence.txt").write_text(unit.stdout+unit.stderr)
    unit.check_returncode()
    assert "edge_identities=1890 errors=6 remaining=0" in unit.stdout
    status.write_text("PASS standalone L-task writeback and reverify; GPU lowering not implemented\n")
except BaseException as error:
    status.write_text(f"STOP {error}\n")
    raise
