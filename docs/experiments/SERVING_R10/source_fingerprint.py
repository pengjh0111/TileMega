#!/usr/bin/env python3
"""Identify the compiler/runtime sources behind an R10 experiment."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def source_fingerprint(root: Path = ROOT) -> str:
    digest = hashlib.sha256()
    files = [root / "CMakeLists.txt",
             root / "docs/experiments/SERVING_R10/calibration/target_serving.json",
             root / "docs/experiments/SIMULATOR/hop_ns.tsv"]
    for directory in ("include", "lib", "tools", "python"):
        files.extend(p for p in (root / directory).rglob("*")
                     if p.is_file() and p.suffix in
                     {".h", ".hpp", ".cuh", ".c", ".cpp", ".cu", ".py", ".td"}
                     and "__pycache__" not in p.parts)
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


if __name__ == "__main__":
    print(source_fingerprint())
