#!/usr/bin/env python3
"""Compare device instructions/resources, without claiming new GPU processes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def device(binary):
    dump=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(binary)],text=True)
    functions={}
    for section in dump.split('Function : ')[1:]:
        name=section.splitlines()[0].strip()
        code='\n'.join(line.strip() for line in section.splitlines()[1:]
                       if re.match(r'^\s*/\*',line))
        if not code or name in functions: raise RuntimeError('missing or duplicate device function')
        functions[name]=hashlib.sha256(code.encode()).hexdigest()
    resources=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(binary)],text=True)
    usage={}
    for section in resources.split(' Function ')[1:]:
        lines=section.splitlines()
        name=lines[0].strip().removesuffix(':')
        usage[name]=next(line.strip() for line in lines[1:] if 'REG:' in line)
    if not functions or set(functions)!=set(usage):
        raise RuntimeError('device instruction/resource function sets differ')
    return functions,usage


def main():
    p=argparse.ArgumentParser()
    p.add_argument('before',type=Path)
    p.add_argument('after',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise RuntimeError('refusing to overwrite comparison evidence')
    manifests=[json.loads((path/'build_manifest.json').read_text()) for path in (a.before,a.after)]
    builds=[{Path(item['command'][-1]).name:item for item in m['builds']} for m in manifests]
    if set(builds[0])!=set(builds[1]): raise RuntimeError('build sets differ')
    result=[]
    for name in sorted(builds[0]):
        versions=[]
        for i,path in enumerate((a.before,a.after)):
            binary=path/'bin'/name
            if hashlib.sha256(binary.read_bytes()).hexdigest()!=builds[i][name]['binary_sha256']:
                raise RuntimeError('binary SHA differs from manifest')
            versions.append(device(binary))
        result.append(dict(binary=name,instructions_equal=versions[0][0]==versions[1][0],
                           resources_equal=versions[0][1]==versions[1][1],
                           instruction_hashes=[v[0] for v in versions],
                           resources=[v[1] for v in versions]))
    a.out.write_text(json.dumps(dict(
        scope='exact disassembled device instruction words and function resource records; host not compared',
        builds=result),indent=2)+'\n')
    equal=sum(r['instructions_equal'] and r['resources_equal'] for r in result)
    print(f'DEVICE_BUILD_EQUAL {equal}/{len(result)}; not new GPU correctness evidence')
    if equal!=len(result): raise RuntimeError('device builds differ; new controls required')


if __name__=='__main__': main()
