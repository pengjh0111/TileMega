#!/usr/bin/env python3
"""Run the three CPU-only runner checks and bind their logs to input bytes."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]


def main():
    out=HERE/'self_check'
    out.mkdir(exist_ok=True)
    for name,folder in (('fence','FENCE'),('sync','SYNC_V3'),('chain','CHAIN2')):
        script=REPO/f'docs/experiments/{folder}/run_sm120.sh'
        array=re.search(r'^python_sources=\((.*?)\)$',script.read_text(),re.M).group(1)
        inputs=[script,HERE/'sm120_common.sh']
        for word in shlex.split(array):
            word=word.replace('${here}',str(script.parent)).replace('${repo}',str(REPO))
            if '$' in word:raise ValueError(f'unresolved self-check input: {word}')
            inputs.append(Path(word))
        if name=='chain':
            inputs += [REPO/'docs/experiments/SIMULATOR'/p for p in
                ('contention.tsv','contention_load.tsv','hop_ns.tsv')]
        command=['bash',str(script)]
        result=subprocess.run(command,cwd=REPO,env={**os.environ,'SELF_CHECK':'1'},
                              capture_output=True,text=True)
        (out/f'{name}.log').write_text(result.stdout+result.stderr)
        (out/f'{name}.json').write_text(json.dumps(dict(command=command,
            env={'SELF_CHECK':'1'},exit_code=result.returncode,time_ns=time.time_ns(),
            inputs={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in inputs}),indent=2)+'\n')
        print(f'{name}: exit={result.returncode}',flush=True)
        result.check_returncode()


if __name__=='__main__':main()
