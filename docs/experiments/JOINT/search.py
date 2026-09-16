#!/usr/bin/env python3
"""R5 finite joint search; every capacity deferral is explicit and reproducible."""
import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
def table(p):
    with p.open() as f:return list(csv.DictReader(f,delimiter='\t'))
def write(p,rows):
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
def source(model):
    if os.getenv('R5_INPUT_ROOT'):return Path(os.environ['R5_INPUT_ROOT'])/'src'/f'{model}.cu'
    return REPO/(f'docs/experiments/PLAN_CONTRACT/legacy_identity/plan/{model}.cu' if model!='real' else 'docs/experiments/REALMODEL/raw/work/r2sim_s4/model.cu')
def exported(model):
    if os.getenv('R5_INPUT_ROOT'):return Path(os.environ['R5_INPUT_ROOT'])/'export'/f'{model}.json'
    return REPO/(f'docs/experiments/SEQSCAN/raw/export/{model}.json' if model!='real' else 'docs/experiments/REALMODEL/raw/work/r2sim_s4/model.json')
def key(c):return f"{c['m']}x{c['n']}x{c['k']}s{c['stages']}k{c['split']}_k{c['kappa']}_r{c['residency']}"
def screen(model,seq,out,geometries):
    target=json.loads(Path(os.getenv('JOINT_TARGET',str(REPO/'configs/targets/sm_89.json'))).read_text())
    sms=target['resources']['num_sms'];smem_limit=target['resources']['max_smem_per_sm']-2048
    sm89=target['arch_tag']=='sm_89'
    phase=Path(os.getenv('JOINT_PHASE_ROOT',str(REPO/'docs/experiments/PHASE/raw')))
    nodes=table(phase/f'trace/{model}_s{seq}_p5/node_phases.tsv')
    grouped=defaultdict(list)
    for r in nodes:grouped[int(r['stage'])].append(r)
    phase_stats={s:(len(rs),sum(float(r['setup_ns'])+float(r['epilogue_ns']) for r in rs)/len(rs),
                    sum(float(r['mainloop_ns']) for r in rs)/len(rs),sum(float(r['load_wait_ns']) for r in rs)/len(rs))
                 for s,rs in grouped.items()}
    gemms=[tuple(map(int,m)) for m in re.findall(r'\{(\d+), (\d+), \d+u,',source(model).read_text().split('constexpr GemmDesc kGemms[] = {')[1].split('};')[0])]
    gemm_stage=[s for s,rs in sorted(grouped.items()) if rs[0]['kind']=='0'];assert len(gemms)==len(gemm_stage)
    op=dict(zip(gemm_stage,gemms))
    dag={s:set() for s in grouped}
    deps=source(model).read_text().split('constexpr StageDependency kDependencies0[] = {')[1].split('};')[0]
    for p,c in re.findall(r'\{(\d+)u,\s*(\d+)u,\s*StageDependency',deps):dag[int(c)].add(int(p))
    shape_regs={}
    if model!='real' and sm89:
        p=REPO/f'docs/experiments/ORACLE/raw_bf16/cost/registers_{model}.tsv'
        for line in p.read_text().splitlines():
            if not line.startswith('#'):
                a,b=line.split('\t');shape_regs[a]=int(b)+8 # screening guard, compile is authoritative
    rows=[];legal=[]
    domains=itertools.product((8,16,32,64,128),(4,8,16,32,64,128),(4,8,16,32,64),(1,2,4,8,16,32))
    configs=[(m,n,k,2,split) for m,n,k,split in domains]+[(128,128,16,3,1)]
    for m,n,k,stages,split in configs:
        shape=f'{m}x{n}x{k}s{stages}'
        admissible=m%32==0 and n%16==0 and k%16==0
        smem=max(2*stages*k*(m+n),16384)
        regs=shape_regs.get(shape,0)
        natural=min(6,smem_limit//smem,65536//(128*((regs+7)//8*8))) if regs else min(2,smem_limit//smem)
        # Residency candidates include low occupancy and the tier-3 seed's
        # natural limit. Recompilation must independently confirm that limit.
        for kap in (1,2,4):
            for resid in sorted({1,2,max(1,natural)}):
                c=dict(m=m,n=n,k=k,stages=stages,split=split,kappa=kap,residency=resid)
                status='eligible' if admissible and resid<=natural else 'backend_or_residency_rejected'
                work=0.;serial=0.;required_flops=0.;path={}
                if status=='eligible':
                    for s,(count,fixed,loop,load) in sorted(phase_stats.items()):
                        ns=fixed+loop+load;combine=0.
                        if s in op:
                            N,K=op[s];chunks=min(split,math.ceil(K/k));tiles=math.ceil(seq/m)*math.ceil(N/n)
                            factor=.7*m*n/16384+.3*(m+n)/256
                            ns=fixed+load*(m+n)/256+loop*factor*(math.ceil(K/(chunks*k))*k/K)
                            count=tiles*chunks;required_flops+=2*seq*N*K
                            if chunks>1:
                                combine=max(1024,fixed)+m*n*min(split,32)*.02
                                work+=tiles*combine
                        work+=count*ns
                        path[s]=max((path[p] for p in dag[s]),default=0)+ns+combine
                    serial=max(path.values())
                # Whole-device arithmetic throughput is a deliberately weak
                # admissible bound, not the phase-scaled priority estimate.
                work_lb=required_flops/(330000.0 if sm89 else 1e9)
                c.update(status=status,work_lb_ns=work_lb,cp_lb_ns=0.,priority_ns=max(work/(sms*resid),serial)+kap*1e-6,
                         estimated_work_ns=work,estimated_stage_cp_ns=serial,register_seed=regs)
                c['key']=key(c);rows.append(c)
                if status=='eligible':legal.append(c)
    # Capacity is applied to geometry, not disguised as a proven dominance
    # prune. Keep three differently shaped high-priority points plus L1 and R4.
    selected=[];seen=set()
    for c in sorted(legal,key=lambda c:c['priority_ns']):
        shape=(c['m'],c['n'],c['k'],c['stages'])
        if shape not in seen:
            selected.append((shape,c['split']));seen.add(shape)
        if len(selected)==geometries:break
    for shape,split in [((32,16,16,2),1),((128,128,16,3),1)]:
        if (shape,split) not in selected:selected.append((shape,split))
    chosen=[c for c in legal if ((c['m'],c['n'],c['k'],c['stages']),c['split']) in selected]
    chosen_keys={c['key'] for c in chosen}
    for c in rows:
        if c['status']=='eligible':c['status']='project' if c['key'] in chosen_keys else 'capacity_deferred'
    out.mkdir(parents=True,exist_ok=True);write(out/'screen.tsv',rows);write(out/'project_manifest.tsv',chosen)
    print('SCREEN',model,seq,'enumerated',len(rows),'project',len(chosen),'geometry',selected,flush=True)
    return chosen

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['screen','project','select']);ap.add_argument('--models',nargs='+',default=['gqa2','mha4','real'])
    ap.add_argument('--seqs',nargs='+',type=int,default=[4,128]);ap.add_argument('--geometries',type=int,default=3)
    ap.add_argument('--raw',type=Path,default=HERE/'raw');ap.add_argument('--driver',type=Path,default=Path('/tmp/r5-project'));ap.add_argument('--rank-driver',type=Path,default=Path('/tmp/r5-rank'));a=ap.parse_args()
    for m in a.models:
        for seq in a.seqs:
            cell=a.raw/f'{m}_s{seq}'
            if a.action=='screen':screen(m,seq,cell,a.geometries);continue
            manifest=table(cell/'project_manifest.tsv')
            if a.action=='project':
                for c in manifest:
                    out=cell/'plans'/c['key'];out.mkdir(parents=True,exist_ok=True)
                    if (out/'process.json').exists():continue
                    cmd=[str(a.driver),str(REPO),str(exported(m)),m,str(seq),*[c[k] for k in ('m','n','k','stages','split','kappa','residency')],str(out.resolve()),'3']
                    begin=time.time_ns()
                    with (out/'project.log').open('w') as f:
                        try:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900);status=r.returncode
                        except subprocess.TimeoutExpired:status=124
                    (out/'process.json').write_text(json.dumps(dict(command=cmd,exit_code=status,started_ns=begin,elapsed_ns=time.time_ns()-begin,driver_sha256=hashlib.sha256(a.driver.read_bytes()).hexdigest()))+'\n')
                    print('PROJECT',m,seq,c['key'],status,flush=True)
            else:
                subprocess.run([str(a.rank_driver),str(cell)],check=True)
                ranked=[]
                for c in manifest:
                    d=cell/'plans'/c['key'];meta=d/'process.json'
                    if not meta.exists() or json.loads(meta.read_text())['exit_code']:continue
                    for p in table(d/'predicted.tsv'):
                        ranked.append(dict(c,**p,source=str(d/(p['placement']+'.cu'))))
                ranked.sort(key=lambda r:(r['status']!='ok' or r['simulated']!='1',float(r['makespan_ns'])))
                core=table(cell/'solver_rank.tsv')
                core_keys=[(r['key'],r['placement']) for r in core if r['status']=='ok' and r['simulated']=='1']
                actual_keys=[(r['key'],r['placement']) for r in ranked if r['status']=='ok' and r['simulated']=='1']
                assert set(core_keys)==set(actual_keys)
                rank_position={k:i for i,k in enumerate(core_keys)}
                ranked.sort(key=lambda r:rank_position.get((r['key'],r['placement']),len(core_keys)))
                write(cell/'ranking.tsv',ranked)
                excluded=set()
                exclusion_file=cell/'numeric_exclusions.tsv'
                if exclusion_file.exists():
                    excluded={tuple(r[k] for k in ('m','n','k','stages','split')) for r in table(exclusion_file)}
                available=[r for r in ranked if r['status']=='ok' and r['simulated']=='1' and tuple(r[k] for k in ('m','n','k','stages','split')) not in excluded]
                top=[];used_shapes=set();used_residency=set()
                while available and len(top)<3:
                    best=min(float(r['makespan_ns']) for r in available)
                    tied=[r for r in available if abs(float(r['makespan_ns'])-best)<=1e-6]
                    shape=lambda r:tuple(r[x] for x in ('m','n','k','stages','split'))
                    chosen=min(tied,key=lambda r:(shape(r) in used_shapes,r['residency'] in used_residency))
                    top.append(chosen);used_shapes.add(shape(chosen));used_residency.add(chosen['residency']);available.remove(chosen)
                write(cell/'top3.tsv',top);print('TOP3',m,seq,[(r['key'],r['placement'],r['makespan_ns']) for r in top],flush=True)
if __name__=='__main__':main()
