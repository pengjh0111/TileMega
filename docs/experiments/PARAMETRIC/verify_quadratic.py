#!/usr/bin/env python3
"""Record exact algebra tests separately from the unfinished model DP gate."""
import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--build", default="build-portable")
args = parser.parse_args()
repo = Path(__file__).resolve().parents[3]
output = Path(__file__).resolve().parent / "quadratic"
output.mkdir(exist_ok=True)
for name in ("lane_intersections_test", "cache_service_curve_test"):
    result = subprocess.run([str(repo / args.build / name)], text=True, capture_output=True)
    (output / (name + ".txt")).write_text(result.stdout + result.stderr)
    result.check_returncode()
    assert "reference_delta=0" in result.stdout
print("exact algebra tests passed; full symbolic model DP not asserted")
