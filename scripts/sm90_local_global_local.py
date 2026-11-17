#!/usr/bin/env python3
"""Predeclare and measure complete native operators, never whole-program scores."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SHAPES = dict(qkv=(4096,2048), o=(2048,2048), gate_up=(12288,2048),
              down=(2048,6144), lm_head=(151936,2048))
CONFIG = ROOT / 'configs/serving/sm90_local_global_ae/qwen3_1_7b_b1'
TARGET = ROOT / 'configs/targets/sm_90_h800.json'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+'\n')
def identity(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()[:16]

def prepare(run):
    if (CONFIG/'local_domain.json').exists():raise ValueError('do not overwrite a frozen domain')
    target=json.loads(TARGET.read_text());c=target['sm90_native_constants']
    rows=[];limit=target['resources']['max_dynamic_smem_per_cta']
    for family,(n,k) in SHAPES.items():
        for tn in (16,32,48,64,128,256):
            if n%tn:continue
            for tk in (64,256,512,1024):
                if k%tk:continue
                group=[]
                for wn in (1,2,4):
                    if tn%(8*wn):continue
                    wk=4//wn
                    for slots in range(2,9):
                        shared=1024+2*slots*tk*(8+tn)+64*wk*tn+512
                        if shared>limit:continue
                        for rs in (1,2,4):
                            if rs>tk//(16*wk):continue
                            p=dict(family=family,n=n,k=k,tn=tn,tk=tk,wn=wn,wk=wk,
                                   slots=slots,register_stages=rs,norm=family in ('qkv','gate_up','lm_head'),
                                   head=family=='lm_head',shared_bytes=shared)
                            p['id']=identity(p)
                            # Sm90DenseCostModel::Gemm(pre=false), used only for screening.
                            # Its fused epilogue is an explicitly recorded proxy for the
                            # standalone linear output; it never chooses the final winner.
                            epiname=dict(qkv='qkv',o='residual',gate_up='gate_up',down='residual',lm_head='lm_head')[family]
                            ep=c['epilogue_ns'][epiname]['wk'+str(wk)]
                            if str(tn) in ep:epvalue=ep[str(tn)]['1'];source='native calibrated epilogue proxy'
                            else:
                                nearest=min(ep,key=lambda t:abs(int(t)-tn))
                                epvalue=ep[nearest]['1']*tn/int(nearest);source='scaled nearest-TN epilogue proxy'
                            tasks=n//tn;per=math.ceil(tasks/132);wt=2*n*k;at=tasks*k*2
                            wc=per*tn*k*2;ac=per*k*2
                            stream=max(wt/c['dram_gbps'],(wt+at)/c['l2_gbps'],
                                       (wc+ac)/c['sm_gbps'],wc*c['dram_latency_ns']/(slots*2*tn*tk))
                            p.update(predicted_ns=c['c_bar_ns']['132']+c['dram_latency_ns']+stream+epvalue,
                                     prediction_source=source,screen_group=f'{family}:TN{tn}:TK{tk}')
                            group.append(p)
                group.sort(key=lambda p:(p['predicted_ns'],p['id']))
                for i,p in enumerate(group):p['shortlisted']=i<2;rows.append(p)
    att=c['attention_ns']['d128:q2:norm1']
    for chunks in (1,2,4,8,16,32,64):
        for bk in (64,128):
            for merge in (('direct',) if chunks==1 else ('la','stage')):
                group=[]
                for slots in (2,3,4):
                    payload=4*slots*bk*128+3*32*128+16*2*128+32*2+4*max(chunks,4)*2+4
                    shared=1024+math.ceil(payload/1024)*1024
                    if shared>limit:continue
                    p=dict(family='attention',chunks=chunks,block_keys=bk,slots=slots,
                           merge=merge,shared_bytes=shared,prepared_input=True)
                    p['id']=identity(p)
                    # Existing midpoint formula; missing S64 merge is extrapolated only
                    # for within-decomposition screening (never drops a decomposition).
                    m=0 if chunks==1 else att['merge'].get(str(chunks),att['merge']['32']*2)
                    keys=math.ceil(576/chunks)
                    p.update(predicted_ns=c['c_bar_ns']['132']+c['l2_latency_ns']+
                        max(8*575*128*4/c['dram_gbps'],math.ceil(8*chunks/132)*
                            (keys*128*4/(c['dram_gbps']/132)+math.ceil(keys/64)*att['block']))+m,
                        prediction_source='native attention midpoint proxy; slots/BK ties use identity',
                        screen_group=f'attention:S{chunks}:BK{bk}:{merge}')
                    group.append(p)
                group.sort(key=lambda p:(p['predicted_ns'],p['id']))
                for i,p in enumerate(group):p['shortlisted']=i<2;rows.append(p)
    domain=dict(schema=1,target=str(TARGET.relative_to(ROOT)),target_sha256=sha(TARGET),
        workload='Qwen3-1.7B/B1/TP1/BF16/ISL64/OSL1024',grid=132,residency=1,cluster=1,
        shared_limit=limit,register_stages=[1,2,4],body='mma.sync / TMA / producer-consumer',
        selection='two implementations per family/TN/TK or attention/chunks/BK/merge; local prediction then identity hash',
        tie_break='lexicographic complete candidate identity; no D preference',
        cache='256 MiB cg-store eviction before every warmup/sample; eviction excluded from timing',
        inputs='deterministic finite BF16 hash-generated data; inverse=1; prepared Q/K/V; same protocol for every candidate',
        weights='prepacked in native tile order before timing; no pack cost; no hot repetition',
        timing='CUDA Graph complete operator, CUDA event elapsed; 3 warmups, 15 samples, median',
        attention_intervals=[dict(first=64+i*1023//5,last=63+(i+1)*1023//5,
            past=(127+i*1023//5+(i+1)*1023//5)//2,steps=(i+1)*1023//5-i*1023//5) for i in range(5)],
        candidates=rows)
    CONFIG.mkdir(parents=True,exist_ok=True);write(CONFIG/'local_domain.json',domain)
    selected=[p for p in rows if p['shortlisted']];random.Random(1741).shuffle(selected)
    run.mkdir(parents=True,exist_ok=True)
    shards=[]
    for j in range(0,len(selected),8):
        batch=selected[j:j+8];stem=f'operators-{j//8:03d}';path=run/(stem+'.cu')
        lines=['#include "sm90_local_global_operator.cuh"',
               'int main(){try{cudaDeviceProp prop{};Check(cudaGetDeviceProperties(&prop,0));',
               'if(prop.major!=9 || prop.multiProcessorCount!=132)throw std::runtime_error("requires 132-SM H800");Bench bench;']
        for p in batch:
            if p['family']=='attention':
                args=[p['chunks'],p['block_keys'],p['slots'],str(p['merge']=='stage').lower()]
                name='AttentionCase'
            else:
                args=[p[k] for k in ('n','k','tn','tk','wn','slots','register_stages')]+[str(p['norm']).lower(),str(p['head']).lower()]
                name='GemmCase'
            lines.append(f'{name}<{",".join(map(str,args))}>(bench,"{p["id"]}");')
        lines.append('return 0;}catch(std::exception const& e){fprintf(stderr,"%s\\n",e.what());return 2;}}')
        path.write_text('\n'.join(lines)+'\n')
        shards.append(dict(source=str(path),binary=str(run/stem),candidates=[p['id'] for p in batch]))
    write(run/'shards.json',dict(domain_sha256=sha(CONFIG/'local_domain.json'),shards=shards))
    print(f'{len(rows)} candidates, {len(selected)} shortlisted, {len(shards)} lightweight shards')

def build(run):
    spec=json.loads((run/'shards.json').read_text())
    def one(s):
        binary=Path(s['binary'])
        if binary.exists():raise ValueError(f'refusing to overwrite {binary}')
        command=['/usr/local/cuda/bin/nvcc','-O3','-std=c++17','-gencode','arch=compute_90a,code=sm_90a',
            '--expt-relaxed-constexpr','--split-compile=4','-Xptxas=-v',
            '-I'+str(ROOT/'include'),'-I'+str(ROOT/'scripts'),'-I'+str(ROOT/'third_party/cutlass/include'),
            s['source'],'-o',str(binary)]
        with binary.with_suffix('.build.log').open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        write(binary.with_suffix('.build.json'),dict(command=command,code=result.returncode,
            source_sha256=sha(s['source']),binary_sha256=sha(binary) if result.returncode==0 else None,
            harness_sha256=sha(ROOT/'scripts/sm90_local_global_operator.cuh')))
        print(binary.name,result.returncode,flush=True)
        return result.returncode
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        codes=list(pool.map(one,spec['shards']))
    if any(codes):raise RuntimeError('one or more candidate shards failed; repair before selection')

def measure(run,out):
    spec=json.loads((run/'shards.json').read_text());out.mkdir(parents=True,exist_ok=True)
    if spec['domain_sha256']!=sha(CONFIG/'local_domain.json'):raise ValueError('domain changed')
    for s in spec['shards']:
        binary=Path(s['binary']);record=json.loads(binary.with_suffix('.build.json').read_text())
        if sha(binary)!=record['binary_sha256']:raise ValueError('binary changed')
        path=out/(binary.name+'.jsonl')
        if path.exists():raise ValueError('retain measurements; use a new attempt')
        with path.open('w') as log:
            subprocess.run([binary],stdout=log,check=True)
        print(binary.name,'measured',flush=True)
    write(out/'receipt.json',dict(domain_sha256=spec['domain_sha256'],shards=spec['shards']))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','build','measure'))
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path)
    a=parser.parse_args();run=a.run.resolve()
    if a.action=='prepare':prepare(run)
    elif a.action=='build':build(run)
    else:measure(run,a.out.resolve())
