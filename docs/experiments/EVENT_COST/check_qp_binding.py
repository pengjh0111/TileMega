#!/usr/bin/env python3
"""Isolate early/late QP binding; never rebuild an active projection probe."""
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    build = repo/'build-portable'
    out = here/'runtime_projection/binding_control'
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    def run(command, name):
        command = list(map(str,command))
        result = subprocess.run(command,capture_output=True,text=True,
                                env=dict(os.environ,TILEMEGA_ISL_AUDIT='1'))
        (out/(name+'.stdout')).write_text(result.stdout)
        (out/(name+'.stderr')).write_text(result.stderr)
        manifest.append(dict(command=command,returncode=result.returncode))
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        result.check_returncode()
    try:
        for flag in ('OFF','ON'):
            run(['cmake','-S',repo,'-B',build,f'-DTILEMEGA_EARLY_QP_BINDING={flag}'],flag+'_configure')
            run(['cmake','--build',build,'--target','isl_relation_test','tilemega-event-cost','-j','4'],flag+'_build')
            run([build/'isl_relation_test'],flag+'_unit')
            run([build/'tools/tilemega-event-cost',repo],flag+'_incidence')
            for case in ('unit','incidence'):
                if 'ISL_CONTEXT remaining=0' not in (out/(flag+'_'+case+'.stderr')).read_text():
                    raise RuntimeError('missing explicit zero-reference evidence')
        old = (out/'OFF_incidence.stdout').read_bytes()
        new = (out/'ON_incidence.stdout').read_bytes()
        if old != new: raise RuntimeError('parameter specialization changed imported metrics')
        (out/'status.txt').write_text('PASS OFF/ON unit assertions; 1920 imported rows byte-identical; '
                                    'sha256='+hashlib.sha256(new).hexdigest()+'\n')
    finally:
        # Restore only configuration; callers choose when to rebuild other
        # binaries, so running experiment manifests retain their binary hash.
        subprocess.run(['cmake','-S',str(repo),'-B',str(build),
                        '-DTILEMEGA_EARLY_QP_BINDING=ON'],check=True,capture_output=True)


if __name__ == '__main__':
    main()
