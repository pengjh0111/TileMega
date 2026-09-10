#!/usr/bin/env python3
"""Rebuild unfused production controls after shared-body/runtime changes."""
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
(out / ".gitignore").write_text("bin/\n")
(out / "bin").mkdir()
status = out / "status.txt"
status.write_text("RUNNING\n")
if any(key.startswith("TILEMEGA_") for key in os.environ):
    raise RuntimeError("remove inherited TILEMEGA overrides")
try:
    manifest = []
    for model in ("gqa2", "mha4"):
        source = repo / "docs/experiments/PLACE/balanced_runtime/src" / (model + "_p0.cu")
        binary = out / "bin" / model
        command = ["/usr/local/cuda/bin/nvcc", "-std=c++17", "-O2", "-arch=sm_89", "-lineinfo",
                   "-Xptxas=-v,--warn-on-spills", "--expt-relaxed-constexpr",
                   "-DTILEMEGA_FUSION_RUNTIME=0", "-DTILEMEGA_PLACEMENT=0",
                   *["-I"+str(repo/p) for p in ("include", "third_party/cutlass/include",
                       "third_party/cutlass/tools/util/include", "third_party/cutlass/test")],
                   str(source), str(repo/"build-portable/libtilemega.a"),
                   "-L/usr/local/cuda/lib64", "-lcudart", "-o", str(binary)]
        built = subprocess.run(command, text=True, capture_output=True, timeout=600)
        (out / (model+"_ptxas.txt")).write_text(built.stdout+built.stderr)
        built.check_returncode()
        manifest.append(dict(model=model, command=command,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    cases = [(model, seq) for model in ("gqa2", "mha4") for seq in (4, 128)]
    for round_id in range(50):
        shift = round_id % len(cases)
        for model, seq in cases[shift:]+cases[:shift]:
            fixture = repo / f"docs/experiments/SEQSCAN/raw/fixture/{model}_s{seq}_p3"
            run = subprocess.run([str(out/"bin"/model), str(fixture)], text=True, capture_output=True,
                timeout=180, env=dict(os.environ, TILEMEGA_WARMUP="5", TILEMEGA_REPEAT="11"))
            text = run.stdout+run.stderr
            (out / f"{model}_s{seq}_r{round_id:03d}.txt").write_text(text)
            run.check_returncode()
            if "RESULT status=PASS" not in text:
                raise RuntimeError("production unfused regression did not pass")
        if (round_id+1) % 10 == 0:
            print(f"{(round_id+1)*4}/200 fresh production processes PASS", flush=True)
    status.write_text("PASS 200/200 unfused BF16 production processes; not fused lowering acceptance\n")
except BaseException as error:
    status.write_text(f"STOP {error}\n")
    raise
