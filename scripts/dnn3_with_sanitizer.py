#!/usr/bin/env python3
"""Bind an isolated Compute Sanitizer package to an unchanged DNN3 runner."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    if len(sys.argv)<4:
        raise SystemExit('usage: dnn3_with_sanitizer.py TOOL_ROOT scripts/dnn3_RUNNER.py [args]')
    package=Path(sys.argv[1]).resolve(strict=True)
    executable=package/'bin/compute-sanitizer'
    if not executable.is_file():
        raise ValueError('package must contain bin/compute-sanitizer')
    runner=Path(sys.argv[2]).resolve(strict=True)
    if runner.parent!=ROOT/'scripts' or not runner.name.startswith('dnn3_') or runner==Path(__file__).resolve():
        raise ValueError('require a DNN3 runner in this repository')
    argv=[str(runner),*sys.argv[3:]]
    if '--out' not in argv:
        raise ValueError('require an output path for tool provenance')
    out=Path(argv[argv.index('--out')+1]).resolve()
    receipt=out.parent/(out.name+'.sanitizer_dispatch.json')
    if receipt.exists():
        raise ValueError('retain previous dispatch; choose a new output')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    files={str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*')) if p.is_file()}
    version=subprocess.run([str(executable),'--version'],capture_output=True,text=True,check=True).stdout.strip()
    out.parent.mkdir(parents=True,exist_ok=True)
    receipt.write_text(json.dumps(dict(evidence='verified',tool_root=str(package),version=version,
        package_sha256=files,runner=str(runner),runner_sha256=sha(runner),
        wrapper_sha256=sha(Path(__file__)),argv=argv),indent=2)+'\n')
    os.environ['PATH']=str(executable.parent)+os.pathsep+os.environ.get('PATH','')
    sys.argv=argv
    sys.path.insert(0,str(runner.parent))
    try:
        runpy.run_path(str(runner),run_name='__main__')
    finally:
        if any(sha(package/name)!=digest for name,digest in files.items()):
            raise ValueError('sanitizer package changed during dispatch')


if __name__=='__main__':
    main()
