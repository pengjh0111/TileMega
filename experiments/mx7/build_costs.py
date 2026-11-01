#!/usr/bin/env python3
"""Freeze and build the complete specified MX-7 TP dense cost domain on the CPU."""
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
MODELS={f'qwen3_1_7b_tp{tp}':dict(tp=tp,hidden=2048,intermediate=6144//tp,
    hkv=8//tp,q=2,dim=128,vocabulary=151936//tp,epsilon=1e-6,layers=28)
    for tp in (2,4)}


def save(p,value):
    with Path(p).open('x')as output:json.dump(value,output,indent=2);output.write('\n')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))


def domain(helper):
    requests=[];regular=[]
    for model,m in MODELS.items():
        for b in (1,2,4,8,16):
            for stage in ('qkv','o','gate_up','down','head'):
                n,k,op,norm={
                    'qkv':((m['q']+2)*m['hkv']*m['dim'],m['hidden'],0,1),
                    'o':(m['hidden'],m['q']*m['hkv']*m['dim'],1,0),
                    'gate_up':(2*m['intermediate'],m['hidden'],2,1),
                    'down':(m['hidden'],m['intermediate'],1,0),
                    'head':(m['vocabulary'],m['hidden'],0,1)}[stage]
                for tn,wn,d,part in itertools.product((16,32,64,128),(1,2,4,8),(1,2,3,4),(1,0)):
                    if stage=='o' and part!=1:continue
                    if stage=='head' and (part!=1 or wn not in (4,8)):continue
                    if stage=='gate_up' and tn<32:continue
                    row=dict(model=model,b=b,stage=stage,n=n,k=k,tn=tn,wn=wn,d=d,part=part,op=op,norm=norm,
                             hidden=m['hidden'],intermediate=m['intermediate'],epsilon=m['epsilon'],tp=m['tp'])
                    requests.append(f'{n} {k} {b} {tn} {wn} {d} {op} {part} 0\n');regular.append(row)
    result=subprocess.run([helper],input=''.join(requests),capture_output=True,text=True,check=True)
    flags=[tuple(map(int,line.split()))for line in result.stdout.splitlines()];assert len(flags)==len(regular)
    rows=[dict(key,first_chunks=chunks)for key,(legal,chunks)in zip(regular,flags)if legal]
    for model,m in MODELS.items():
        for b,prefetch,b_min in itertools.product((1,2,4,8,16),(0,1,2),(1,2,4,8)):
            rows.append(dict(model=model,b=b,stage='attention',dim=128,q=2,hkv=m['hkv'],
                             prefetch=prefetch,b_min=b_min,attn_version=2,epsilon=m['epsilon'],tp=m['tp']))
    return rows


def definitions(key,pw):
    values=dict(B=key['b'])
    if key['stage']=='attention':
        values.update(HKV=key['hkv'],DIM=128,Q=2,PREFETCH=key['prefetch'],BMIN=key['b_min'])
        return 'attention_v2_bench.cu',values,'Mx7AttentionBench'
    values.update(TN=key['tn'],WN=key['wn'],D=key['d'],PW=pw,N=key['n'],K=key['k'],PART=key['part'],
                  STAGE={'qkv':0,'o':1,'down':1,'head':2,'gate_up':4}[key['stage']])
    return 'dense_stage_bench.cu',values,'Mx7DenseBench'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=24)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();destination=args.out.resolve();destination.mkdir(parents=True,exist_ok=args.resume)
    frozen=destination/'frozen'
    if not frozen.exists():
        frozen.mkdir();shutil.copytree(ROOT/'include',frozen/'include');(frozen/'experiments/mx7').mkdir(parents=True)
        for name in ('dense_stage_bench.cu','attention_v2_bench.cu','cost_domain.cpp','measure_costs.py'):
            shutil.copy2(ROOT/'experiments/mx7'/name,frozen/'experiments/mx7'/name)
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
    if not helper.exists():subprocess.run(['c++','-std=c++17','-I'+str(frozen/'include'),str(frozen/'experiments/mx7/cost_domain.cpp'),'-o',str(helper)],check=True)
    if not(destination/'DOMAIN.json').exists():save(destination/'DOMAIN.json',domain(helper))
    keys=json.loads((destination/'DOMAIN.json').read_text());builds=destination/'builds';builds.mkdir(exist_ok=True)
    from tilemega.build.identity import parse_resources
    base=[source['compiler']['path'],'-std=c++17','-O3','-shared','-Xcompiler=-fPIC,-resource-usage','-fno-strict-aliasing',
        '-mllvm','--max-mtreg-number=255','-DTILEMEGA_WATCHDOG=0','-DTILEMEGA_MX_DIAG=0','-I'+str(frozen/'include')]

    def compile_one(directory,key,pw):
        if(directory/'BUILD.json').exists():return json.loads((directory/'BUILD.json').read_text())
        directory.mkdir(parents=True);filename,defs,kernel=definitions(key,pw);binary=directory/'library.so'
        argv=base+[f'-DMX7_{name}={value}'for name,value in defs.items()]+[str(frozen/'experiments/mx7'/filename),'-o',str(binary)]
        start=time.time()
        with(directory/'stdout.log').open('x')as output:p=subprocess.run(argv,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,timeout=1800)
        resources=parse_resources((directory/'stdout.log').read_text());matches=[r for n,r in resources.items()if kernel in n]
        resource=matches[0]if len(matches)==1 else{}
        fit=p.returncode==0 and bool(resource)and resource.get('stack_bytes')==resource.get('static_smem_bytes')==0 and 0<resource.get('registers',0)<=255
        row=dict(key=key,argv=argv,returncode=p.returncode,compile_s=time.time()-start,source_digest=source['source_digest'],
            status='resource_fit'if fit else'compile_rejected'if p.returncode else'resource_rejected',resources=resource,
            source=ref(frozen/'experiments/mx7'/filename),binary=ref(binary)if binary.exists()else None)
        save(directory/'BUILD.json',row);return row

    def one(item):
        ordinal,key=item;directory=builds/f'{ordinal:05d}';final=directory/'COMPLETE.json'
        if final.exists():return json.loads(final.read_text())
        if key['stage']=='attention':
            result=compile_one(directory/'main',key,None)
            outcome=dict(key=key,status=result['status'],attempts=[ref(directory/'main/BUILD.json')])
            if result['status']=='resource_fit':outcome.update(binary=result['binary'],build=ref(directory/'main/BUILD.json'))
        else:
            base_key=dict(key,prefetch=0,r_next=None)
            zero=compile_one(directory/'pw0',base_key,0)
            outcome=dict(key=key,status=zero['status'],attempts=[ref(directory/'pw0/BUILD.json')])
            if zero['status']=='resource_fit':
                registers=zero['resources']['registers'];width=max(key['d'],min(key['first_chunks'],min(64,248-registers)//(key['tn']//4)))
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
        save(final,outcome);print('MX7_COST_BUILD',ordinal,key['model'],key['b'],key['stage'],outcome['status'],flush=True);return outcome
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
