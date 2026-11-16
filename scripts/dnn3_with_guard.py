#!/usr/bin/env python3
"""Dispatch an existing DNN3 runner through the versioned 4 MiB idle guard.

Keep frozen runner bytes intact; the guard receipt binds the selected provider.
"""
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    if len(sys.argv)<2:
        raise SystemExit('usage: dnn3_with_guard.py scripts/dnn3_RUNNER.py [args]')
    script=Path(sys.argv[1]).resolve()
    if script.parent!=ROOT/'scripts' or not script.name.startswith('dnn3_') or script==Path(__file__).resolve():
        raise ValueError('wrapper requires an existing DNN3 runner in this repository')
    old=ROOT/'scripts/dnn1_idle_floor_python.py'
    provider=ROOT/'scripts/dnn3_idle_floor_python.py'
    original=subprocess.run
    def guarded(command,*args,**kwargs):
        if isinstance(command,(list,tuple)) and command and Path(command[0]).resolve()==old:
            command=[str(provider),*command[1:]]
        return original(command,*args,**kwargs)
    argv=[str(script),*sys.argv[2:]]
    if '--out' in argv:
        out=Path(argv[argv.index('--out')+1]).resolve()
        out.parent.mkdir(parents=True,exist_ok=True)
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        receipt=out.parent/(out.name+'.guard_dispatch.json')
        if receipt.exists():
            raise ValueError('retain existing guard dispatch; use a new output directory')
        receipt.write_text(json.dumps(dict(evidence='verified',time=time.time(),argv=argv,
            wrapper=str(Path(__file__).resolve()),wrapper_sha256=sha(Path(__file__)),
            runner_sha256=sha(script),guard_provider=str(provider),guard_provider_sha256=sha(provider)),indent=2)+'\n')
    subprocess.run=guarded
    sys.argv=argv
    sys.path.insert(0,str(script.parent))
    runpy.run_path(str(script),run_name='__main__')


if __name__=='__main__':
    main()
