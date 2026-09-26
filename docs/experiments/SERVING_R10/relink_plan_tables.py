#!/usr/bin/env python3
"""Rebuild completed serving plans with library-local GEMM variant tables.

Older generated libraries exported kGemmVariantInfo as a GNU-unique symbol.
When prefill and decode were loaded in one process, the loader coalesced their
different tile tables. This repair uses each plan's recorded nvcc command and
the unchanged generated CUDA source; it never reruns the solver.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
WORK = Path("/root/r10_work/plans")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exports_variant_table(binary: Path) -> bool:
    result = subprocess.run(["nm", "-D", "--defined-only", str(binary)],
                            check=True, capture_output=True, text=True)
    return "kGemmVariantInfo" in result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="*", default=[])
    args = parser.parse_args()
    cells = (args.cells or [f"{model}_{phase}_B{batch}"
                            for model in ("llama", "qwen3")
                            for phase in ("prefill", "decode")
                            for batch in (1, 2, 4, 8, 16)])
    failed = False
    for cell in cells:
        binary = WORK / cell / "plan.so"
        if not binary.exists():
            print(f"{cell}: missing plan", flush=True)
            failed = True
            continue
        evidence = HERE / "plans" / cell
        evidence.mkdir(parents=True, exist_ok=True)
        report_path = evidence / "symbol_isolation.json"
        before = digest(binary)
        exported = exports_variant_table(binary)
        if not exported:
            report_path.write_text(json.dumps({
                "cell": cell, "relinked": False, "binary_sha256": before,
                "variant_table_exported": False,
            }, indent=2) + "\n")
            print(f"{cell}: already isolated", flush=True)
            continue
        command_file = Path(str(binary) + ".build_command.txt")
        if not command_file.exists():
            raise FileNotFoundError(command_file)
        command = shlex.split(command_file.read_text().strip())
        cuda_source = Path(next(item for item in command
                                if item.endswith(".cu") and item.startswith(str(WORK))))
        if digest(cuda_source) != digest(Path(str(binary) + ".cu")):
            raise RuntimeError(f"{cell}: recorded nvcc source differs from final CUDA")
        output_at = command.index("-o") + 1
        temporary = Path(str(binary) + ".relinking")
        command[output_at] = str(temporary)
        start = time.perf_counter()
        log = evidence / "symbol_isolation_build.log"
        with log.open("w") as output:
            result = subprocess.run(command, cwd=ROOT, stdout=output,
                                    stderr=subprocess.STDOUT)
        if result.returncode:
            failed = True
            print(f"{cell}: nvcc failed ({result.returncode})", flush=True)
            continue
        if exports_variant_table(temporary):
            failed = True
            print(f"{cell}: variant table still exported", flush=True)
            temporary.unlink()
            continue
        os.replace(temporary, binary)
        shutil.copyfile(log, Path(str(binary) + ".relink.log"))
        audit = subprocess.run(["python3", str(HERE / "audit_sass.py"),
                                str(binary), "--out",
                                str(evidence / "fp64_audit.json")],
                               cwd=ROOT, capture_output=True, text=True)
        (evidence / "fp64_audit.stdout").write_text(audit.stdout + audit.stderr)
        if audit.returncode:
            failed = True
        report_path.write_text(json.dumps({
            "cell": cell, "relinked": True, "old_binary_sha256": before,
            "binary_sha256": digest(binary),
            "generated_source_sha256": digest(Path(str(binary) + ".cu")),
            "variant_table_exported": False,
            "compile_seconds": time.perf_counter() - start,
            "nvcc_command": command,
            "fp64_audit_pass": audit.returncode == 0,
        }, indent=2) + "\n")
        print(f"{cell}: relinked, FP64 audit {'PASS' if audit.returncode == 0 else 'FAIL'}",
              flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
