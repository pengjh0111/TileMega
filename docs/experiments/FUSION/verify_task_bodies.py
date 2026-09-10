#!/usr/bin/env python3
"""Fresh-process shared GEMM body tests; not full-model fusion acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--processes", type=int, default=50)
args = parser.parse_args()
if args.processes < 50:
    parser.error("synchronization evidence requires at least 50 fresh processes")
repo = Path(__file__).resolve().parents[3]
out = args.out.resolve()
out.mkdir(exist_ok=False)
(out / ".gitignore").write_text("task_body_bf16\ntask_body_fp32\n")
status = out / "status.txt"
status.write_text("RUNNING\n")
try:
    source = repo / "test/unit/fused_gemm_task_body_test.cu"
    for dtype in ("bf16", "fp32"):
        binary = out / ("task_body_" + dtype)
        command = ["/usr/local/cuda/bin/nvcc", "-std=c++17", "-O3", "-arch=sm_89",
                   "-DTILEMEGA_MODEL_BF16=" + str(int(dtype == "bf16")),
                   "-DTILEMEGA_GEMM_TILE_M=32", "-DTILEMEGA_GEMM_TILE_N=128",
                   "-DTILEMEGA_GEMM_TILE_K=16", "-DTILEMEGA_GEMM_STAGES=3",
                   "-Iinclude", "-Ithird_party/cutlass/include",
                   "-Ithird_party/cutlass/tools/util/include", "--expt-relaxed-constexpr",
                   "--ptxas-options=-v", str(source), "-o", str(binary)]
        (out / (dtype + "_command.json")).write_text(json.dumps(command) + "\n")
        build = subprocess.run(command, cwd=repo, text=True, capture_output=True, timeout=600)
        (out / (dtype + "_ptxas.txt")).write_text(build.stdout + build.stderr)
        build.check_returncode()
        (out / (dtype + "_binary.sha256")).write_text(hashlib.sha256(binary.read_bytes()).hexdigest()+"\n")
    # Interleave dtype states inside every round. No performance claim is made.
    for round_id in range(args.processes):
        states = ("bf16", "fp32") if round_id % 2 == 0 else ("fp32", "bf16")
        for dtype in states:
            run = subprocess.run([str(out / ("task_body_" + dtype))], text=True,
                                 capture_output=True, timeout=60)
            (out / f"{dtype}_{round_id:03d}.txt").write_text(run.stdout + run.stderr)
            run.check_returncode()
            assert "cases=24 bit_equal=1 global_intermediate_untouched=1" in run.stdout
        if (round_id+1) % 10 == 0:
            print(f"{round_id+1}/{args.processes} fresh processes per dtype PASS", flush=True)
    status.write_text(f"PASS BF16 {args.processes}/{args.processes}, FP32 {args.processes}/{args.processes}; "
                      "24 body cases/process; full-model lowering not verified\n")
except BaseException as error:
    status.write_text(f"STOP {error}\n")
    raise
