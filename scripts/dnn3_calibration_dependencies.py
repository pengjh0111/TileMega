#!/usr/bin/env python3
"""Certify the exact header dependencies of previously measured F3 probes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=False)
    target=json.loads(args.target.read_text())['sm90_f3_constants']
    headers=target['header_sha256']
    for name,digest in headers.items():
        if sha(ROOT/'include'/name)!=digest:
            raise ValueError('create dependency evidence before changing headers: '+name)
    rows=[];sources={}
    for raw in target['raw_sources']:
        result=ROOT/raw['result']
        if sha(result)!=raw['sha256']:raise ValueError('measured result changed')
        d=json.loads(result.read_text());build=Path(d['build'])
        if sha(build)!=raw['build_sha256']:raise ValueError('probe build receipt changed')
        b=json.loads(build.read_text())
        for probe in b['probes']:
            command=probe['command'];index=next(i for i,x in enumerate(command) if Path(x).name=='nvcc')
            command=command[index:];source=next(Path(x) for x in command if x.endswith('.cu'))
            expected=b['inputs'][str(source)];key=(str(source),expected)
            if key not in sources:
                data=source.read_bytes()
                if hashlib.sha256(data).hexdigest()!=expected:
                    relative=str(source.relative_to(ROOT));data=None
                    history=subprocess.check_output(['git','log','--format=%H','--',relative],cwd=ROOT,text=True).splitlines()
                    for commit in history:
                        candidate=subprocess.check_output(['git','show',commit+':'+relative],cwd=ROOT)
                        if hashlib.sha256(candidate).hexdigest()==expected:data=candidate;break
                    if data is None:raise ValueError('cannot reconstruct measured CUDA source: '+relative)
                frozen=args.out/(expected+'-'+source.name);frozen.write_bytes(data);sources[key]=frozen
            frozen=sources[key];dependency=args.out/(str(len(rows))+'.d')
            # Preserve every preprocessing/architecture option of the recorded build.
            options=[];i=1
            while i<len(command):
                value=command[i]
                if value=='-o':i+=2;continue
                options.append(str(frozen) if value==str(source) else value);i+=1
            cmd=[command[0],'-M',*options]
            with dependency.open('w') as stream:
                subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=stream,stderr=subprocess.PIPE,check=True,timeout=120)
            paths=shlex.split(dependency.read_text().replace('\\\n',' ').split(':',1)[1])
            included={str(Path(name).resolve().relative_to(ROOT/'include')) for name in paths if Path(name).resolve().is_relative_to(ROOT/'include')}
            if not included or any(name not in headers for name in included):raise ValueError('unknown probe header dependency')
            if sha(Path(probe['binary']))!=probe['binary_sha256']:raise ValueError('measured binary changed')
            rows.append(dict(build=str(build),build_sha256=sha(build),binary_sha256=probe['binary_sha256'],source_sha256=expected,command=cmd,dependencies=str(dependency),dependencies_sha256=sha(dependency),headers={name:headers[name] for name in sorted(included)}))
    for name,digest in headers.items():
        if sha(ROOT/'include'/name)!=digest:raise ValueError('headers changed during dependency audit')
    receipt=dict(evidence='verified',status='passed',target_sha256=sha(args.target),script_sha256=sha(Path(__file__)),nvcc_version=subprocess.check_output(['nvcc','--version'],text=True),rows=rows)
    (args.out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(probes=len(rows),header_union=len(set().union(*(r['headers'] for r in rows))),status='passed')))


if __name__=='__main__':
    main()
