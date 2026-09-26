#!/usr/bin/env python3
"""Solve and retain the 20 full-domain R10 serving plans."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "plans"
WORK = Path("/root/r10_work/plans")
PYTHON = "/root/venvs/tilemega-torch213-cu126/bin/python"
ARTIFACTS = (
    ".plan.json", ".search.tsv", ".timing.tsv", ".phases.tsv",
    ".resources.tsv", ".materializations.tsv", ".top3_measured.tsv",
    ".floor.tsv", ".floor_tensors.tsv", ".build_command.txt", ".ptxas.log",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("llama", "qwen3"),
                        default=["llama", "qwen3"])
    parser.add_argument("--phases", nargs="+", choices=("prefill", "decode"),
                        default=["prefill", "decode"])
    parser.add_argument("--batches", nargs="+", type=int,
                        default=[1, 2, 4, 8, 16])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    failed = False
    for model in args.models:
        for phase in args.phases:
            previous: Path | None = None
            for batch in args.batches:
                cell = f"{model}_{phase}_B{batch}"
                work = WORK / cell
                evidence = EVIDENCE / cell
                library = work / "plan.so"
                manifest = Path(str(library) + ".plan.json")
                if library.exists() and manifest.exists():
                    print(f"{cell}: reuse completed plan", flush=True)
                    evidence.mkdir(parents=True, exist_ok=True)
                    audit = evidence / "fp64_audit.json"
                    if not audit.exists():
                        with (evidence / "fp64_audit.stdout").open("w") as stream:
                            check = subprocess.run(["python3", str(HERE / "audit_sass.py"),
                                str(library), "--out", str(audit)], cwd=ROOT,
                                stdout=stream, stderr=subprocess.STDOUT)
                        if check.returncode:
                            print(f"{cell}: FP64 audit failed", flush=True)
                            failed = True
                    previous = manifest
                    continue
                work.mkdir(parents=True, exist_ok=True)
                evidence.mkdir(parents=True, exist_ok=True)
                bridge = Path(f"/root/r10_work/export/{model}_{phase}/bridge.json")
                command = [
                    str(ROOT / "build-portable/tools/tilemega-compile"),
                    str(bridge), str(library), "--serving", phase,
                    "--batch", str(batch), "--past-range",
                    "64:1086" if phase == "decode" else "0:0",
                    "--capacity", "1088", "--solver", "skeleton",
                    "--solve", str(HERE / "calibration/target_serving.json"),
                    "--hop-curve", str(ROOT / "docs/experiments/SIMULATOR/hop_ns.tsv"),
                    "--variant-cache", "/root/r9_work/variant_resources",
                    "--emit", "serving", "--search-passes", "3",
                    "--top-m", "8", "--incremental-prepare", "1",
                    "--measure-cmd",
                    f"PYTHONPATH={ROOT / 'python'} {PYTHON} -m "
                    f"tilemega.serving.measure_candidate --model "
                    f"/root/models/{'llama3_2_1b' if model == 'llama' else 'qwen3_1_7b'}",
                ]
                if previous is not None:
                    command += ["--serving-warm-start", str(previous)]
                if args.dry_run:
                    print(json.dumps({"cell": cell, "command": command}), flush=True)
                    previous = manifest
                    continue
                print(f"{cell}: solve started", flush=True)
                start = time.perf_counter()
                with (work / "stdout.txt").open("w") as stdout, \
                     (work / "stderr.txt").open("w") as stderr:
                    process = subprocess.run(command, cwd=ROOT,
                                             stdout=stdout, stderr=stderr)
                seconds = time.perf_counter() - start
                result = {"cell": cell, "command": command,
                          "seconds": seconds, "returncode": process.returncode,
                          "library": str(library), "manifest": str(manifest)}
                (evidence / "result.json").write_text(
                    json.dumps(result, indent=2) + "\n")
                for name in ("stdout.txt", "stderr.txt"):
                    shutil.copyfile(work / name, evidence / name)
                for suffix in ARTIFACTS:
                    path = Path(str(library) + suffix)
                    if path.exists():
                        shutil.copyfile(path, evidence / path.name)
                if process.returncode or not library.exists() or not manifest.exists():
                    print(f"{cell}: FAIL after {seconds:.1f}s; stop this batch chain",
                          flush=True)
                    failed = True
                    break
                with (evidence / "fp64_audit.stdout").open("w") as stream:
                    check = subprocess.run(["python3", str(HERE / "audit_sass.py"),
                        str(library), "--out", str(evidence / "fp64_audit.json")],
                        cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                if check.returncode:
                    print(f"{cell}: FP64 audit failed", flush=True)
                    failed = True
                    break
                print(f"{cell}: PASS in {seconds:.1f}s", flush=True)
                previous = manifest
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
