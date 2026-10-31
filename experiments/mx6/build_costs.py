#!/usr/bin/env python3
"""Freeze and build the complete specified MX-6 MoE cost domain on the CPU."""
import argparse
import concurrent.futures
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

ROOT=Path(__file__).resolve().parents[2]
MODELS={
    'qwen3_30b_a3b_l24':dict(hidden=2048,intermediate=768,experts=128,hkv=4,q=8,dim=128,vocabulary=151936,epsilon=1e-6,layers=24),
    'olmoe_1b_7b':dict(hidden=2048,intermediate=1024,experts=64,hkv=16,q=1,dim=128,vocabulary=50304,epsilon=1e-5,layers=16)}
MODE_VALUES={'owner16':0,'owner_tile':1,'unit_la':3}
MODES=tuple(MODE_VALUES)


def save(p,value):
    with Path(p).open('x')as output:json.dump(value,output,indent=2);output.write('\n')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))


def domain(helper,replay):
    requests=[];regular=[]
    for model,m in MODELS.items():
        for b in (1,2,4,8,16):
            for stage in ('qkv','o','router','gate_up','down','head'):
                n,k,op,norm={
                    'qkv':((m['q']+2)*m['hkv']*m['dim'],m['hidden'],0,1),
                    'o':(m['hidden'],m['q']*m['hkv']*m['dim'],1,0),
                    'router':(m['experts'],m['hidden'],0,1),
                    'gate_up':(2*m['intermediate'],m['hidden'],2,1),
                    'down':(m['hidden'],m['intermediate'],0,0),
                    'head':(m['vocabulary'],m['hidden'],0,1)}[stage]
                grouped=stage in ('gate_up','down');partition_n=n*min(b*8,m['experts'])if grouped else n
                for tn,wn,d,part in itertools.product((16,32,64,128),(1,2,4,8),(1,2,3,4),(1,0)):
                    if stage=='router'and(tn!=16 or wn!=1 or part!=1):continue
                    if stage=='o'and part!=1:continue
                    if grouped and part!=1:continue
                    if stage=='head'and(part!=1 or wn not in (4,8)):continue
                    row=dict(model=model,b=b,stage=stage,n=n,k=k,tn=tn,wn=wn,d=d,part=part,op=op,norm=norm,
                        hidden=m['hidden'],intermediate=m['intermediate'],experts=m['experts'],epsilon=m['epsilon'])
                    if grouped:row['replay']=ref(replay/f'{model}_B{b}.json')
                    requests.append(f'{partition_n} {k} {b} {tn} {wn} {d} {op} {part} {int(grouped)}\n');regular.append(row)
    result=subprocess.run([helper],input=''.join(requests),capture_output=True,text=True,check=True)
    flags=[tuple(map(int,line.split()))for line in result.stdout.splitlines()];assert len(flags)==len(regular)
    rows=[]
    for key,(legal,chunks)in zip(regular,flags):
        if not legal:continue
        for mode in MODES if key['stage']=='down'else('owner16',):
            row=dict(key,first_chunks=chunks)
            if key['stage']=='down':row['combine_mode']=mode
            rows.append(row)
    for model,m in MODELS.items():
        for b,prefetch,split in itertools.product((1,2,4,8,16),(0,1,2),(1,2,4)):
            if split>1 and b*m['hkv']*split>104:continue
            rows.append(dict(model=model,b=b,stage='attention',dim=m['dim'],q=m['q'],hkv=m['hkv'],
                qk_norm=1,row_norm_heads=m['hkv']if m['experts']==64 else 0,prefetch=prefetch,split=split,epsilon=m['epsilon']))
    return rows


def definitions(key,pw):
    stage=key['stage'];values=dict(B=key['b'])
    if stage=='attention':
        values.update(HKV=key['hkv'],DIM=key['dim'],Q=key['q'],NORM=key['qk_norm'],ROW=key['row_norm_heads'],PREFETCH=key['prefetch'],SPLIT=key['split'])
        return 'attention_v3_bench.cu',values,'Mx6AttentionBench'
    values.update(TN=key['tn'],WN=key['wn'],D=key['d'],PW=pw)
    if stage in ('gate_up','down'):
        values.update(E=key['experts'],H=key['hidden'],I=key['intermediate'],GU=int(stage=='gate_up'),COMBINE=MODE_VALUES[key.get('combine_mode','owner16')])
        return 'moe_grouped_bench.cu',values,'Mx6GroupedBench'
    values.update(N=key['n'],K=key['k'],PART=key['part'],STAGE={'qkv':0,'o':1,'head':2,'router':3}[stage])
    return 'moe_router_logits_bench.cu'if stage=='router'else'regular_stage_bench.cu',values,'Mx6RegularBench'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--replay',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=24)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();destination=args.out.resolve();destination.mkdir(parents=True,exist_ok=args.resume)
    frozen=destination/'frozen'
    if not frozen.exists():
        frozen.mkdir();shutil.copytree(ROOT/'include',frozen/'include');(frozen/'experiments/mx6').mkdir(parents=True)
        for name in ('regular_stage_bench.cu','moe_router_logits_bench.cu','moe_grouped_bench.cu','attention_v3_bench.cu','cost_domain.cpp','measure_costs.py'):
            shutil.copy2(ROOT/'experiments/mx6'/name,frozen/'experiments/mx6'/name)
        files={str(p.relative_to(frozen)):sha(p)for p in frozen.rglob('*')if p.is_file()}
        compiler=Path('/opt/maca/tools/cu-bridge/bin/cucc')
        source=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files=files,
            source_digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),compiler=ref(compiler),
            date_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),protocol='CPU build only; no default or production acceptance')
        save(destination/'SOURCE.json',source)
        with tarfile.open(destination/'SOURCE.tar.gz','w:gz')as archive:archive.add(frozen,arcname='frozen')
    source=json.loads((destination/'SOURCE.json').read_text())
    for name,digest in source['files'].items():
        if sha(frozen/name)!=digest:raise ValueError('frozen cost source changed')
    helper=destination/'cost_domain'
    if not helper.exists():subprocess.run(['c++','-std=c++17','-I'+str(frozen/'include'),str(frozen/'experiments/mx6/cost_domain.cpp'),'-o',str(helper)],check=True)
    if not(destination/'DOMAIN.json').exists():save(destination/'DOMAIN.json',domain(helper,args.replay.resolve()))
    keys=json.loads((destination/'DOMAIN.json').read_text());builds=destination/'builds';builds.mkdir(exist_ok=True)
    from tilemega.build.identity import parse_resources
    base=[source['compiler']['path'],'-std=c++17','-O3','-shared','-Xcompiler=-fPIC,-resource-usage','-fno-strict-aliasing',
        '-mllvm','--max-mtreg-number=255','-DTILEMEGA_WATCHDOG=0','-DTILEMEGA_MX_DIAG=0','-I'+str(frozen/'include')]

    def compile_one(directory,key,pw):
        if(directory/'BUILD.json').exists():return json.loads((directory/'BUILD.json').read_text())
        directory.mkdir(parents=True);filename,defs,kernel=definitions(key,pw);binary=directory/'library.so'
        argv=base+[f'-DMX6_{name}={value}'for name,value in defs.items()]+[str(frozen/'experiments/mx6'/filename),'-o',str(binary)]
        start=time.time()
        with(directory/'stdout.log').open('x')as output:p=subprocess.run(argv,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,timeout=1800)
        resources=parse_resources((directory/'stdout.log').read_text());matches=[r for n,r in resources.items()if kernel in n]
        resource=matches[0]if len(matches)==1 else{}
        fit=p.returncode==0 and bool(resource)and resource.get('stack_bytes')==resource.get('static_smem_bytes')==0 and 0<resource.get('registers',0)<=255
        row=dict(key=key,argv=argv,returncode=p.returncode,compile_s=time.time()-start,source_digest=source['source_digest'],
            status='resource_fit'if fit else'compile_rejected'if p.returncode else'resource_rejected',resources=resource,
            source=ref(frozen/'experiments/mx6'/filename),binary=ref(binary)if binary.exists()else None)
        save(directory/'BUILD.json',row);return row

    def one(item):
        ordinal,key=item;directory=builds/f'{ordinal:05d}';final=directory/'COMPLETE.json'
        if final.exists():return json.loads(final.read_text())
        if(ROOT/'runs/mx6/v2/A_STOP.json').exists():raise RuntimeError('A stop')
        if key['stage']=='attention':
            result=compile_one(directory/'main',key,None)
            outcome=dict(key=key,status=result['status'],attempts=[ref(directory/'main/BUILD.json')])
            if result['status']=='resource_fit':outcome.update(binary=result['binary'],build=ref(directory/'main/BUILD.json'))
        else:
            base_key=dict(key,prefetch=0,r_next=None)
            zero=compile_one(directory/'pw0',base_key,0)
            outcome=dict(key=key,status=zero['status'],attempts=[ref(directory/'pw0/BUILD.json')])
            if zero['status']=='resource_fit':
                registers=zero['resources']['registers'];width=0 if key['stage']=='gate_up'else max(key['d'],min(key['first_chunks'],min(64,248-registers)//(key['tn']//4)))
                while True:
                    resolved=dict(key,prefetch=width,r_next=registers)
                    if width==0:
                        # Same binary/source, a separate metadata receipt binds resolved R_next.
                        result=dict(zero,key=resolved);receipt=directory/'pw0/RESOLVED_BUILD.json'
                        if not receipt.exists():save(receipt,result)
                    else:
                        result=compile_one(directory/f'pw{width}',resolved,width);receipt=directory/f'pw{width}/BUILD.json'
                    outcome['attempts'].append(ref(receipt))
                    if result['status']=='resource_fit':
                        outcome.update(key=resolved,status='resource_fit',binary=result['binary'],build=ref(receipt),
                            zero_prefetch=dict(binary=zero['binary'],build=ref(directory/'pw0/BUILD.json')));break
                    smaller=max(key['d'],width//2)
                    if smaller>=width:outcome.update(key=resolved,status=result['status']);break
                    width=smaller
        save(final,outcome);print('MX6_COST_BUILD',ordinal,key['model'],key['b'],key['stage'],outcome['status'],flush=True);return outcome
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)as pool:results=list(pool.map(one,enumerate(keys)))
    rows=[];rejections=[]
    for result in results:
        if result['status']!='resource_fit':rejections.append(result);continue
        if result['key']['stage']=='attention':
            for length in (65,256,512,768,1088):rows.append(dict(result,key=dict(result['key'],length=length)))
        else:rows.append(result)
    save(destination/'KEYS_CPU.json',rows);save(destination/'REJECTIONS_CPU.json',rejections)
    save(destination/'COMPLETE.json',dict(compiled_coordinates=len(keys),fitted_keys=len(rows),rejected_coordinates=len(rejections),source=ref(destination/'SOURCE.json'),GPU_measured=False))


if __name__=='__main__':main()
