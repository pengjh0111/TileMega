#!/usr/bin/env python3
"""Identify the compiler/runtime sources behind an R10 experiment."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


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


def main():
    import argparse
    import json
    import subprocess
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--header', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args()
    value = source_fingerprint(args.root)
    if args.header:
        body = '// Generated at build time; do not edit.\n#pragma once\n#define TILEMEGA_SOURCE_SHA256 "' + value + '"\n'
        args.header.parent.mkdir(parents=True, exist_ok=True)
        if not args.header.exists() or args.header.read_text() != body:
            args.header.write_text(body)
    elif args.check:
        record = json.loads(subprocess.check_output([str(args.check), 'version', '--json'], text=True))
        if record['source_sha256'] != value:
            raise SystemExit('FAIL source fingerprint: binary and source differ')
        print('PASS source fingerprint ' + value)
    else:
        print(value)

if __name__ == '__main__':
    main()
