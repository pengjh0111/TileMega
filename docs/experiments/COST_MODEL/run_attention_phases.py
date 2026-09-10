#!/usr/bin/env python3
"""Numerical primitive validation, not the A7 production-model acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build-portable/attention_phase_test"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve()
    frozen = digest(binary)
    rows = []
    for seed in range(1, 51):
        command = [str(binary), str(seed)]
        run = subprocess.run(command, capture_output=True, text=True, timeout=120)
        log = args.out / f"seed{seed}.txt"
        log.write_text(run.stdout + run.stderr)
        match = re.search(r"ATTENTION_PHASE cases=(\d+) chunk1_bits=(\d+) max_abs=(\S+) PASS", log.read_text())
        if run.returncode or not match or match.group(1, 2) != ("32", "8"):
            (args.out / "status.txt").write_text(f"STOP seed={seed}; original log retained\n")
            raise RuntimeError(f"attention primitive failure: {log}")
        rows.append(dict(seed=seed, command=command, log_sha256=digest(log),
                         max_abs=float(match.group(3))))
    if digest(binary) != frozen:
        raise RuntimeError("binary changed during the process sweep")
    sources = [Path(p) for p in (
        "test/unit/attention_phase_test.cu",
        "include/tilemega/Codegen/AttentionPlan.h",
        "include/tilemega/Codegen/tasks/AttentionPhasedTaskBody.h",
        "include/tilemega/Codegen/tasks/AttentionCombineTaskBody.h")]
    result = dict(scope="BF16 numerical primitive; not full models or persistent queues",
                  binary=str(binary), binary_sha256=frozen, processes=50,
                  cases=1600, chunk1_bit_cases=400, rows=rows,
                  sources={str(p): digest(p) for p in sources})
    (args.out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "status.txt").write_text("PASS 50/50 processes, 1600 numerical primitive cases\n")


if __name__ == "__main__":
    main()
