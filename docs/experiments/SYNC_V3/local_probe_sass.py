#!/usr/bin/env python3
"""Audit local completion barriers retained by the unsafe neither arm."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent


def main():
    out=HERE/'local_probe_sass'
    out.mkdir(exist_ok=True)
    records=[]
    for config in ('window2','local2'):
        for model in ('gqa2','mha4'):
            binary=HERE/config/'bin'/f'{model}_p0_neither'
            output=out/f'{config}_{model}_neither.sass'
            cmd=['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(binary)]
            with output.open('wb') as f:
                result=subprocess.run(cmd,stdout=f,check=True)
            records.append(dict(configuration=config,model=model,command=cmd,
                exit_code=result.returncode,
                binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                sass_sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
    (out/'commands.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
