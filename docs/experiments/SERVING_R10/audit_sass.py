#!/usr/bin/env python3
"""Count FP64 SASS instructions in serving megakernels."""

import argparse
import json
import re
import subprocess
from pathlib import Path


FP64 = re.compile(r"\b(?:DADD|DMUL|DFMA|DSETP|F2F\.F64|F2F\.F32\.F64|I2F\.F64|F64)\b")
KERNEL = re.compile(r"tilemega_l[12]_kernel")


def audit(binary: Path, cuobjdump: str) -> dict:
    dump = subprocess.run([cuobjdump, "-sass", str(binary)], check=True,
                          capture_output=True, text=True).stdout
    function = ""
    counts = {}
    matches = {}
    for line in dump.splitlines():
        if "Function : " in line:
            function = line.split("Function : ", 1)[1].strip()
            if KERNEL.search(function):
                counts.setdefault(function, 0)
                matches.setdefault(function, [])
            continue
        if KERNEL.search(function) and line.lstrip().startswith("/*"):
            instruction = FP64.search(line.split(";", 1)[0])
            if instruction:
                counts[function] += 1
                matches[function].append(line.strip())
    if not counts:
        raise RuntimeError(f"no serving megakernel found in {binary}")
    return {"binary": str(binary), "fp64_total": sum(counts.values()),
            "fp64_by_kernel": counts, "instructions": matches}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path, nargs="+")
    parser.add_argument("--cuobjdump", default="/usr/local/cuda-12.8/bin/cuobjdump")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    results = [audit(binary, args.cuobjdump) for binary in args.binary]
    report = {"binaries": results,
              "fp64_total": sum(item["fp64_total"] for item in results)}
    rendered = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered, end="")
    if report["fp64_total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
