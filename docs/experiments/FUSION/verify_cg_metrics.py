#!/usr/bin/env python3
"""Retain positive and negative CG metric-verifier receipts with ISL audit."""
import os
from pathlib import Path
import subprocess

repo = Path(__file__).resolve().parents[3]
output = Path(__file__).resolve().parent / "cg_metric_guard"
output.mkdir(exist_ok=False)
for name, expected in (("valid", 0), ("fanout_mismatch", 1), ("wait_mismatch", 1)):
    run = subprocess.run([str(repo / "build-portable/tools/tilemega-opt"),
        str(repo / f"test/Dialect/CouplingGraph/{name}.mlir")], capture_output=True, text=True,
        env=dict(os.environ, TILEMEGA_ISL_AUDIT="1"))
    (output / (name+".txt")).write_text(run.stdout+run.stderr)
    assert run.returncode == expected and "ISL_CONTEXT remaining=0" in run.stderr
run = subprocess.run([str(repo / "build-portable/tools/tilemega-scalar-error-probe"), str(repo)],
                     capture_output=True, text=True)
(output / "scalar_errors.txt").write_text(run.stdout+run.stderr)
run.check_returncode()
assert "SCALAR_ERRORS branches=27 reference_delta=0" in run.stdout
(output / "status.txt").write_text("PASS 3 verifier paths and 27 scalar/fusion error paths; zero ISL references\n")
