#!/usr/bin/env python3
"""Check phased attention work and preserve the pre-change scalar prices."""
import argparse
import json
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--build", default="build-portable")
args = parser.parse_args()
repo = Path(__file__).resolve().parents[3]
output = Path(__file__).resolve().parent / "attention_work"
output.mkdir(exist_ok=True)
tool = repo / args.build / "tools"
result = subprocess.run([str(tool / "tilemega-attention-work"), str(repo)], text=True, capture_output=True)
(output / "phases.tsv").write_text(result.stdout)
(output / "phases.txt").write_text(result.stderr)
result.check_returncode()
assert "per_task_checks=2720 errors=14 reference_delta=0" in result.stderr
previous = Path(__file__).resolve().parent / "scalar_work/verified"
checked = 0
rejected = 0
for source in sorted(previous.glob("*.tsv")):
    match = re.fullmatch(r"(bf16|f32)_(gqa2|mha4)_s(\d+)_p(\d+)\.tsv", source.name)
    if not match:
        continue
    dtype, model, seq, past = match.groups()
    result = subprocess.run([str(tool / "tilemega-scalar-cost-probe"), str(repo), model, dtype, seq, past],
                            capture_output=True)
    (output / source.name).write_bytes(result.stdout)
    (output / (source.stem + ".txt")).write_bytes(result.stderr)
    old_error = source.with_suffix(".txt").read_bytes()
    if b"runtime dimension outside exported parameter domain" in old_error:
        if result.returncode != 2 or result.stderr != old_error or result.stdout != source.read_bytes():
            raise RuntimeError(f"historical domain rejection changed: {source.name}")
        rejected += 1
        print(f"retained domain rejection: {source.name}", flush=True)
        continue
    result.check_returncode()
    if result.stdout != source.read_bytes():
        raise RuntimeError(f"scalar resource extraction changed frozen price bytes: {source.name}")
    checked += 1
    print(f"scalar control {checked}: {source.name}", flush=True)
assert checked > 0
(output / "summary.json").write_text(json.dumps(dict(phase_task_checks=2720,
    rejection_branches=14, scalar_tables_byte_identical=checked,
    retained_domain_rejections=rejected, gpu_run=False), indent=2) + "\n")
