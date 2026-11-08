#!/usr/bin/env python3
"""Compile each unique DNN3 winner stage alone for the DNN4 P-CODE census."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def declaration(text,pattern):
    match=re.search(pattern,text,re.M|re.S)
    if match is None:raise ValueError('missing declaration: '+pattern)
    return match.group(0)


def inventory(winners):
    unique={}
    for winner in winners:
        source=Path(winner['library']+'.cu');text=source.read_text()
        batch,buffers,maps=map(int,re.search(r'using F3ForwardProgram=Program<(\d+),(\d+),(\d+),',text).groups())
        for match in re.finditer(r'^using Body_(\d+)=(.*);$',text,re.M):
            i=int(match[1]);body=match[0]
            bind=declaration(text,rf'^struct Bind_{i} \{{.*?^\}};')
            shape=declaration(text,rf'^struct Shape_{i} \{{.*?^\}};')
            conv=''
            if re.search(rf'\bConv_{i}\b',body):
                conv=declaration(text,rf'^struct Conv_{i} \{{.*?^\}};')
            wrap=declaration(text,rf'^using Wrap_{i}=.*?;')
            offset=declaration(text,rf'^constexpr int kOffset_{i}=.*?;')
            stage=declaration(text,rf'^using Stage_{i}=.*?;')
            signature='\n'.join([body,bind,shape,conv,wrap])
            signature=re.sub(r'\b(Body|Bind|Shape|Conv|Wrap)_\d+\b',r'\1_X',signature)
            # Buffer/map numbers select addresses, not a different body template.
            signature=re.sub(r'=\d+u', '=0u',signature)
            key=hashlib.sha256(signature.encode()).hexdigest()
            use=dict(model=winner['model'],batch=batch,stage=i,source=str(source),source_sha256=sha(source))
            if key in unique:
                unique[key]['uses'].append(use);continue
            code='\n'.join(['// SPDX-License-Identifier: BSD-3-Clause',
                '#define TILEMEGA_STATIC_FORWARD_F3 1','#define TILEMEGA_ARCH_ID 900',
                '#define TILEMEGA_TRACE_STAGE 0',
                '#include <tilemega/Codegen/tasks/Sm90F3Program.cuh>',
                'namespace tilemega::codegen::forward_f3 {','using namespace tilemega::backend;',
                conv,bind,shape,body,wrap,offset,stage,
                f'using ProbeProgram=Program<{batch},{buffers},{maps},Stage_{i}>;',
                'template __global__ void sm90_forward_f3_kernel<ProbeProgram>(',
                'const __grid_constant__ ProbeProgram::Binding,ForwardCounter*,unsigned long long);','}'])+'\n'
            unique[key]=dict(key=key,body=match[2],signature=signature,code=code,uses=[use])
    return list(unique.values())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--winners',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=False)
    winners=json.loads(args.winners.read_text())['rows'];specs=inventory(winners)
    (args.out/'inventory.json').write_text(json.dumps(specs,indent=2)+'\n')
    print(json.dumps(dict(unique_stage_templates=len(specs),stage_instances=sum(len(s['uses']) for s in specs))),flush=True)
    lock=threading.Lock();rows=[]

    def compile_one(item):
        index,spec=item;name=f'{index:03d}_{spec["key"][:16]}'
        source=args.out/(name+'.cu');source.write_text(spec['code'])
        binary=args.out/(name+'.cubin');log=args.out/(name+'.ptxas.log')
        cpu=136+2*(index%4)
        command=['taskset','-c',f'{cpu}-{cpu+1}','nvcc','-rdc=true','--cubin','-std=c++17','-O3',
                 '-gencode','arch=compute_90a,code=sm_90a','--expt-relaxed-constexpr',
                 '-Xptxas','-v','-Xptxas','--device-function-maxrregcount=224',
                 '-I'+str(ROOT/'include'),'-I'+str(ROOT/'third_party/cutlass/include'),
                 str(source),'-o',str(binary)]
        with log.open('w') as stream:
            done=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=1200,
                                env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        if done.returncode:raise RuntimeError('stage code-size compile failed: '+str(log))
        sass=args.out/(name+'.sass')
        with sass.open('w') as stream:
            subprocess.run(['cuobjdump','--dump-sass',str(binary)],stdout=stream,stderr=subprocess.STDOUT,check=True)
        active=False;count=0;kernels=0
        for line in sass.read_text().splitlines():
            if 'Function :' in line:
                active='sm90_forward_f3_kernel' in line
                kernels+=int(active)
            elif active and re.match(r'^\s*/\*[0-9a-fA-F]+\*/\s+(?:@!?[A-Z]+\d+\s+)?[A-Z]',line):
                count+=1
        if kernels!=1 or count==0:raise ValueError('missing or duplicate stage kernel')
        row=dict(key=spec['key'],body=spec['body'],uses=spec['uses'],sass_instructions=count,sass_bytes=16*count,
                 source=str(source),source_sha256=sha(source),binary=str(binary),binary_sha256=sha(binary),
                 sass=str(sass),sass_sha256=sha(sass),log=str(log),log_sha256=sha(log),command=command)
        with lock:
            rows.append(row)
            (args.out/'result.json').write_text(json.dumps(dict(evidence='verified',status='partial',
                expected_templates=len(specs),rows=rows),indent=2)+'\n')
            print(json.dumps(dict(completed=len(rows),expected=len(specs),sass_bytes=row['sass_bytes'],template=name)),flush=True)
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(compile_one,enumerate(specs)))
    (args.out/'result.json').write_text(json.dumps(dict(evidence='verified',status='passed',
        expected_templates=len(specs),rows=rows,scope='single-stage Program kernel; not equal to an in-situ stage instruction slice',
        winners_sha256=sha(args.winners)),indent=2)+'\n')


if __name__=='__main__':main()
