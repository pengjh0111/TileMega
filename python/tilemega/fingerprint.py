#!/usr/bin/env python3
"""Identify the compiler/runtime sources behind an R10 experiment."""
import hashlib
import json
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


# A TaskBody change invalidates its calibration section, not the export graph.
# Common target/schema and suite code participate in every section.
def calibration_stamps(root: Path = ROOT) -> dict[str, str]:
    common = ['include/tilemega/Target/TargetSpec.h', 'include/tilemega/Target/Calibration.h',
              'include/tilemega/Target/ArchDispatch.h', 'lib/Target/TargetSpec.cpp',
              'tools/commands/calibrate.cpp', 'tools/commands/calibration-suite.cpp']
    sections = {
        'base': ['lib/Target/Calibration.cu'],
        'bf16': ['lib/Target/Calibration.cu', 'lib/Target/GemmCalibration.cu',
                 'include/tilemega/Backend/CutlassGemmCandidate.h'],
        'wait': ['lib/Target/ServingSyncCalibration.cu', 'include/tilemega/Codegen/tasks/EventSync.cuh'],
        'events': ['lib/Target/ServingSyncCalibration.cu', 'include/tilemega/Codegen/tasks/EventSync.cuh'],
        'hop': ['lib/Target/ServingSyncCalibration.cu', 'include/tilemega/Codegen/tasks/EventSync.cuh'],
        'inflight': ['lib/Target/InflightCalibration.cu'],
        'task_bodies': ['lib/Target/ServingTaskCalibration.cu',
                        'include/tilemega/Backend/Serving*.h',
                        'include/tilemega/Codegen/tasks/Serving*.h',
                        'include/tilemega/Codegen/tasks/*AttentionTaskBody.h',
                        'include/tilemega/Codegen/tasks/AttentionMergeTaskBody.h',
                        'include/tilemega/Codegen/executor/*.cuh'],
    }
    result = {}
    for section, patterns in sections.items():
        paths = sorted({p for pattern in common + patterns for p in root.glob(pattern) if p.is_file()})
        digest = hashlib.sha256()
        for path in paths:
            digest.update(str(path.relative_to(root)).encode() + b'\0' + path.read_bytes() + b'\0')
        result[section] = digest.hexdigest()
    return result


def main():
    import argparse
    import json
    import subprocess
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--header', type=Path)
    parser.add_argument('--check', type=Path)
    parser.add_argument('--calibration-stamps', action='store_true')
    args = parser.parse_args()
    value = source_fingerprint(args.root)
    if args.header:
        body = '// Generated at build time; do not edit.\n#pragma once\n#define TILEMEGA_SOURCE_SHA256 "' + value + '"\n'
        body += '#define TILEMEGA_CALIBRATION_STAMPS_JSON R"stamps(' + json.dumps(calibration_stamps(args.root), sort_keys=True) + ')stamps"\n'
        args.header.parent.mkdir(parents=True, exist_ok=True)
        if not args.header.exists() or args.header.read_text() != body:
            args.header.write_text(body)
    elif args.calibration_stamps:
        print(json.dumps(calibration_stamps(args.root), sort_keys=True))
    elif args.check:
        record = json.loads(subprocess.check_output([str(args.check), 'version', '--json'], text=True))
        if record['source_sha256'] != value:
            raise SystemExit('FAIL source fingerprint: binary and source differ')
        print('PASS source fingerprint ' + value)
    else:
        print(value)

if __name__ == '__main__':
    main()
