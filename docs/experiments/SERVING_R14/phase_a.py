#!/usr/bin/env python3
"""Bounded Phase-A correctness, trace collection and preregistered selection."""
import argparse,csv,json,os,statistics,subprocess,sys
from pathlib import Path
from make_phase0 import HERE,ROOT,PYTHON,write
from tilemega.build.identity import verify,sha
from choose_r14 import baseline
from ledger_r14 import tasks

def run(cmd,out,timeout=3600,allowed=(0,)):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.with_suffix('.command.json').write_text(json.dumps(list(map(str,cmd)),indent=2)+'\n')
    with out.open('w') as log:code=subprocess.run(list(map(str,cmd)),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=timeout).returncode
    if code not in allowed:raise SystemExit(code)

def prepare():
    out=HERE/'raw/Apre';out.mkdir(parents=True,exist_ok=True)
    run(['cmake','--build','build-phase12','--target','tilemega','attention_layout_test','paged_gemm_test','-j','6'],out/'build.log',6000)
    run([PYTHON,'python/tilemega/fingerprint.py','--check','build-phase12/tools/tilemega'],out/'fingerprint.log')
    for script in ['test/python/serving_identity.py','docs/experiments/SERVING_R14/test_selection.py','docs/experiments/SERVING_R14/test_phase_a.py']:
        run([PYTHON,script],out/(Path(script).stem+'.log'))
    nvcc='/usr/local/cuda/bin/nvcc';common=[nvcc,'-std=c++17','-O3','--expt-relaxed-constexpr','-DTILEMEGA_TRACE_TASK=1','-Iinclude','-Ithird_party/cutlass/include']
    binaries={}
    for arch in (80,89,90,100,120):
        dst=out/f'attention_sm{arch}';cmd=common+[f'-arch=sm_{arch}','test/unit/attention_layout_test.cu']
        if arch!=89:cmd+=['-c'];dst=dst.with_suffix('.o')
        run(cmd+['-o',dst],out/f'compile_sm{arch}.log',900);binaries[str(arch)]=dict(path=str(dst),sha256=sha(dst))
    write(out/'binaries.json',binaries)

def smoke():
    out=HERE/'raw/A0s';out.mkdir(parents=True,exist_ok=True)
    run([HERE/'raw/Apre/attention_sm89'],out/'position_coded.log',600)
    rows=[]
    for job in json.loads((HERE/'phase_a_builds.json').read_text()):
        so=Path(job['out'])/'plan.so';identity=verify(so)
        config=json.loads((ROOT/f"configs/e2e/{job['cell'].split('_')[0]}_r13_final.json").read_text())
        dest=out/job['cell']/job['label']
        run([PYTHON,'-m','tilemega.serving.smoke','--so',so,'--model',config['model']['path'],'--batch',job['cell'].split('_B')[1],'--steps','64','--out',dest],dest/'run.log',300)
        report=json.loads((dest/'smoke.json').read_text())
        if not report['pass']:raise ValueError(f"smoke failed {job['cell']}/{job['label']}")
        rows.append(dict(cell=job['cell'],label=job['label'],artifact_id=identity['artifact_id'],passed=True))
    write(out/'results.json',rows)

def trace(cell):
    for arm in json.loads((HERE/'phase_a_trace_arms.json').read_text())[cell]:
        suffix=arm['label'].split('_')[-1]
        if suffix not in ('stage','task'):continue
        out=HERE/'raw'/f'A2_trace_{cell}'/arm['label']
        run([PYTHON,'-m','tilemega.serving.trace','--model',arm['model_path'],'--prefill-so',arm['prefill'],'--decode-so',arm['decode'],
             '--batch',arm['batch'],'--mode','L1','--decode-loop','0','--'+suffix,'--past-list','64,575,1000','--launches','16','--out',out],out/'run.log',900)
        for folder in out.glob('past*'):
            if suffix=='task':
                result=tasks(folder/'task_profile.tsv')
                if set(result)!={'attention','gemm'}:raise ValueError('task instrumentation produced no rows for a required kind')
                write(folder/'task_summary.json',result)
            else:
                from ledger_r14 import stage_ledger
                write(folder/'ledger.json',stage_ledger(folder,arm['batch']))

def metric(cell,label,r):
    return json.loads((HERE/f'raw/A1_{cell}_r{r}/{cell}/round{r}.json').read_text())['arms'][label]

def timed_tokens(record):
    rows=[r['tokens'] for r in record['runs'] if not r['warmup'] and r['N']==1024]
    if len(rows)!=1:raise ValueError('expected one timed full generation')
    return rows[0]

def correctness(cell):
    out=HERE/'raw'/f'A3_{cell}';out.mkdir(parents=True,exist_ok=True)
    arms=json.loads((HERE/'phase_a_references.json').read_text())[cell];old=[timed_tokens(metric(cell,'R13D',r)) for r in range(3)]
    new=[timed_tokens(metric(cell,'R13D_prime',r)) for r in range(3)]
    mismatch=sum(a!=b for xs,ys in zip(old,new) for x,y in zip(xs,ys) for a,b in zip(x,y))
    repeated=all(x==new[0] for x in new) and all(x==old[0] for x in old)
    write(out/'c2.json',dict(r13_tokens_identical=mismatch==0,timed_repeats_identical=repeated,mismatches=mismatch))
    # Baseline token regression is a hard stop, before any later performance matrix.
    if mismatch or not repeated:raise ValueError('R13D reference token regression')
    model=arms[0]['model'];prompts=ROOT/f'docs/experiments/SERVING_R10/prompts/{model}_ids.json'
    v=metric(cell,'vllm',0);vout=Path(v['out'])/('B'+str(arms[0]['batch']))
    timed=[r for r in v['generation_runs'] if r['N']==1024 and not r['warmup']]
    if len(timed)!=1:raise ValueError('expected one timed vLLM generation')
    vt=vout/timed[0]['tokens_file']
    common=[PYTHON,'-m','tilemega.serving.hf_check','--model',arms[0]['model_path'],'--prompt-ids',prompts,'--skip-free-greedy']
    run(common+['--generated',vt,'--out',out/'vllm_hf.json'],out/'vllm_hf.log',1800,allowed=(0,1))
    for arm in arms:
        if arm['label'] not in ('R13D_prime','N1_prime','B0h_prime'):continue
        label=arm['label'];tokens=out/(label+'_tokens.json')
        if label=='B0h_prime':
            import torch
            from tilemega.serving.engine import ServingEngine
            ids=torch.tensor(json.loads(prompts.read_text())[:arm['batch']],dtype=torch.int32)
            with ServingEngine(arm['model_path'],arm['prefill'],arm['decode'],arm['batch'],mode='L1',decode_loop=False,prefill_mode='L1') as engine:
                result=engine.generate(ids,1024)
                write(tokens,result.tokens.tolist())
        else:write(tokens,timed_tokens(metric(cell,label,0)))
        run(common+['--generated',tokens,'--vllm-metrics',out/'vllm_hf.json','--out',out/(label+'_hf.json')],out/(label+'_hf.log'),1800)

def analyze():
    rows=[];selection={}
    for cell,arms in json.loads((HERE/'phase_a_arms.json').read_text()).items():
        selection[cell]={};c2=json.loads((HERE/f'raw/A3_{cell}/c2.json').read_text())
        for arm in arms:
            label=arm['label'];observations=[metric(cell,label,r) for r in range(3)]
            if any(x['exit_code'] for x in observations):raise ValueError('failed anchor arm')
            identities=[x.get('execution_identity') for x in observations]
            if any(identities):
                if any(x is None for x in identities) or len({x['execution_id'] for x in identities})!=1:
                    raise ValueError('anchor artifact or execution changed between rounds')
                if any(x['trace'] for x in identities):raise ValueError('trace artifact cannot enter T1')
            elif arm['kind']=='tm' and 'decode' not in arm.get('binaries',{}):
                raise ValueError('legacy anchor lacks pinned binary identity')
            values=[(x['e2e_seconds']-x['ttft_seconds'])/1023 for x in observations]
            rows.append(dict(cell=cell,label=label,tpot_samples=values,tpot_median=statistics.median(values),tpot_range=max(values)-min(values),
                execution_identity=observations[0].get('execution_identity'),binary=arm.get('binaries',{}).get('decode'),
                raw=[str(HERE/f'raw/A1_{cell}_r{r}/{cell}/round{r}.json') for r in range(3)]))
            if label in ('R13D_prime','N1_prime'):
                report=json.loads((HERE/f'raw/A3_{cell}/{label}_hf.json').read_text());key="R13D'" if label=='R13D_prime' else "N1'"
                selection[cell][key]=dict(tpot=values,c1=report['pass'],r13_tokens_identical=c2['r13_tokens_identical'])
    overhead=[]
    for cell,arms in json.loads((HERE/'phase_a_trace_arms.json').read_text()).items():
        for arm in arms:
            label=arm['label'];suffix=label.split('_')[-1]
            if suffix not in ('stage','task'):continue
            base=label.rsplit('_',1)[0];values=[];ratios=[];execution_ids=[]
            for r in range(3):
                result=json.loads((HERE/f'raw/A2_overhead_{cell}_r{r}/{cell}/round{r}.json').read_text())['arms']
                a,b=result[label],result[base]
                if a['exit_code'] or b['exit_code']:raise ValueError('trace overhead arm failed')
                if timed_tokens(a)!=timed_tokens(b):raise ValueError('trace instrumentation changed tokens')
                if not a['execution_identity']['trace'] or b['execution_identity']['trace']:raise ValueError('trace identity mismatch')
                av=(a['e2e_seconds']-a['ttft_seconds'])/1023;bv=(b['e2e_seconds']-b['ttft_seconds'])/1023
                values.append(av);ratios.append(av/bv-1);execution_ids.append(a['execution_identity']['execution_id'])
            if len(set(execution_ids))!=1:raise ValueError('trace artifact changed between rounds')
            overhead.append(dict(cell=cell,label=label,execution_id=execution_ids[0],paired_overhead=ratios,
                median_overhead=statistics.median(ratios),range=max(ratios)-min(ratios),
                below_two_percent=abs(statistics.median(ratios))<=.02,
                purpose='diagnostic only; sampled stage extrema are estimates'))
    write(HERE/'results/T2_trace_overhead.json',overhead)
    identities=[]
    for job in json.loads((HERE/'phase_a_builds.json').read_text()):
        i=verify(Path(job['out'])/'plan.so')
        identities.append(dict(cell=job['cell'],label=job['label'],artifact_id=i['artifact_id'],trace=i['trace'],
            resources=i['resources'],shared_memory_bytes=i['shared_memory_bytes'],implementations=i['implementations']))
    write(HERE/'results/T3_identities.json',identities)
    write(HERE/'results/T1_anchor.json',rows);write(HERE/'results/baseline_input.json',selection)
    write(HERE/'baselines_r14.json',baseline(selection))
    print('Phase A baseline selection completed')

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','smoke','trace','correctness','analyze'));p.add_argument('--cell');a=p.parse_args()
    {'prepare':prepare,'smoke':smoke,'trace':lambda:trace(a.cell),'correctness':lambda:correctness(a.cell),'analyze':analyze}[a.action]()
if __name__=='__main__':main()
