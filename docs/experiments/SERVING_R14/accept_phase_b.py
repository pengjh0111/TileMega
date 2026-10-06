#!/usr/bin/env python3
"""Review identities, guarded paired rounds and the registered Phase-C gates."""
import csv,json,statistics,tarfile
from collections import defaultdict
from pathlib import Path
from tilemega.build.identity import digest,sha,verify
from choose_r14 import CELLS,phase_c,discernible
HERE=Path(__file__).resolve().parent

def load(p):return json.loads(Path(p).read_text())
def write(p,value):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(value,indent=2)+'\n')
def table(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),delimiter='\t',lineterminator='\n');writer.writeheader();writer.writerows(rows)

def review():
    state=load(HERE/'scheduler/state.json');queue=load(HERE/'queue/queue_phase_b.json')
    guards=[]
    for step in queue:
        status=state[step['name']]
        if status['status']!='done' or status['exit_code']:raise ValueError(step['name'])
        if step.get('gpu'):
            record=load(HERE/'raw'/step['name']/'guard_result.json')
            if record['code']:raise ValueError('guard rejected '+step['name'])
            guards.append(dict(step=step['name'],attempts=status['attempts'],**record))
    identities=[]
    for job in load(HERE/'phase_b_builds.json'):
        identity=verify(Path(job['out'])/'plan.so')
        for name,resource in identity['resources'].items():
            if 'tilemega_l1_kernel' in name:
                identities.append(dict(cell=job['cell'],label=job['label'],artifact_id=identity['artifact_id'],trace=identity['trace'],smem=identity['shared_memory_bytes'],registers=resource['registers'],stack=resource['stack_bytes'],spill=resource['spill'],spill_store=resource['spill_store_bytes'],spill_load=resource['spill_load_bytes']))
    rows=[];matrix={};correctness={};profiles=[];groups={};canaries=[]
    reference=load(HERE/'inputs/r13_loadbench_c.json')
    for cell in CELLS:
        checks=load(HERE/f'raw/B6_{cell}/results.json');correctness[cell]=checks
        if any(not v['c1'] or not v['c2']['pass'] or v['same_as_baseline'] is False for v in checks):raise ValueError('correctness '+cell)
        matrix[cell]={}
        for i in range(1,6):
            arms=defaultdict(list);ids={};paths=defaultdict(list)
            for path in sorted(HERE.glob(f'raw/B{i}_{cell}_r*/{cell}/round*.json')):
                for label,record in load(path)['arms'].items():
                    if record['exit_code']:raise ValueError('failed arm '+str(path))
                    ident=record['execution_identity']
                    if ident['trace'] or ident['execution_id']!=digest({k:v for k,v in ident.items() if k!='execution_id'}):raise ValueError('invalid performance identity')
                    if label in ids and ids[label]!=ident['execution_id']:raise ValueError('mixed identity across rounds')
                    ids[label]=ident['execution_id'];paths[label].append(str(path.relative_to(HERE)))
                    arms[label].append((record['e2e_seconds']-record['ttft_seconds'])*1000/1023)
            if not arms:continue
            base=statistics.median(arms['baseline'])
            for label,values in arms.items():
                if len(values)!=3:raise ValueError('incomplete paired rounds')
                med=statistics.median(values);row=dict(cell=cell,matrix=f'B{i}',label=label,execution_id=ids[label],tpot_ms=med,range_ms=max(values)-min(values),relative=med/base-1,discernible=discernible(values,arms['baseline']),samples_ms=json.dumps(values),raw=json.dumps(paths[label]))
                rows.append(row);matrix[cell].setdefault(f'B{i}',{})[label]=row
            for k,v in enumerate(arms['baseline']):
                if abs(v/base-1)>.02:canaries.append(dict(cell=cell,matrix=f'B{i}',round=k,relative=v/base-1))
        folder=HERE/f'raw/B_trace_{cell}/baseline/past575'
        metadata={int(v['stage']):v for v in csv.DictReader((folder/'runtime_stages.tsv').open(),delimiter='\t')}
        by=defaultdict(list)
        for task in csv.DictReader((folder/'task_profile.tsv').open(),delimiter='\t'):
            kind=task['kind'];desc=metadata[int(task['stage'])]
            cls=next((n for n in ('qkv','o_proj','gate_up','down_proj','lm_head') if n in desc['name']),kind)
            by[cls].append(task)
        manifest=load(HERE/f'raw/B0b/{cell}/baseline/plan.so.plan.json');model='llama' if cell.startswith('llama') else 'qwen3'
        config=load(Path('/root/models')/('llama3_2_1b' if model=='llama' else 'qwen3_1_7b')/'config.json')
        gp=[]
        for cls,tasks in by.items():
            duration=sum(int(t['run_end'])-int(t['run_begin']) for t in tasks)
            row=dict(cell=cell,label='baseline',kind=cls,samples=len(tasks),mean_run_ns=duration/len(tasks),bytes=sum(int(t['bytes']) for t in tasks)/len(tasks),page_wait_fraction=sum(int(t['first_page_wait_ns'])+int(t['later_page_wait_ns']) for t in tasks)/duration,epilogue_fraction=sum(int(t['epilogue_ns']) for t in tasks)/duration)
            if cls!='attention':
                idx={'qkv':0,'o_proj':1,'gate_up':2,'down_proj':3,'lm_head':len(manifest['gemms'])-1}[cls];gemm=manifest['gemms'][idx]
                k=config['intermediate_size'] if cls=='down_proj' else config['hidden_size']
                n=config['hidden_size'] if cls in ('o_proj','down_proj') else (config['vocab_size'] if cls=='lm_head' else config['intermediate_size']*2 if cls=='gate_up' else (config['num_attention_heads']+2*config['num_key_value_heads'])*config.get('head_dim',config['hidden_size']//config['num_attention_heads']))
                active=min(manifest['grid'],(n+gemm['tile_n']-1)//gemm['tile_n']);fraction=round(active/reference['num_sms']*100)
                match=[p for p in reference['points'] if p['method']==2 and p['row']==(manifest['pg']!='pages') and p['tile_n']==gemm['tile_n'] and p['tile_k']==gemm['tile_k'] and p['stages']==gemm['stages'] and p['K']==k and p['active_pct']==fraction]
                # Paged TN32 uses the same 16-B stream, but its two-stage ring
                # differs from the GEMV probe; use the measured 980 ceiling as
                # an explicit lower bound, never call it a matched measurement.
                rate=match[0]['gbps'] if match else 980.0
                row.update(active_ctas=active,pure_load_gbps=rate,pure_load_source='matched MB-1c' if match else '980 GB/s optimistic bound',pure_load_ns=row['bytes']*active/rate,over_load_fraction=row['mean_run_ns']/(row['bytes']*active/rate)-1)
                gp.append(row)
            profiles.append(row)
        groups[cell]=dict(gemm=gp,attention=next(v for v in profiles if v['cell']==cell and v['kind']=='attention'),pg=manifest['pg'])
    inputs={name:{} for name in ('C-AT3b','C-RW1','C-RW2','C-EP2','C-AT4','C-GV2')}
    for cell,data in groups.items():
        paged=data['pg']=='pages';b16=cell.endswith('B16');gemm=data['gemm']
        over=max(r['over_load_fraction'] for r in gemm)
        inputs['C-AT3b'][cell]=dict(applicable=paged,attention_page_wait_fraction=data['attention']['page_wait_fraction'])
        inputs['C-RW1'][cell]=dict(applicable=not paged and b16,gemm_over_load_fraction=over,metric_detail='maximum class mean relative to pure loading, including actual active CTA count')
        pipe=matrix[cell].get('B4',{}).get('RW_pipe',{}).get('relative',0)
        inputs['C-RW2'][cell]=dict(applicable=paged and b16,gemm_over_load_fraction=over,pipe_recovers_fraction_lt_half=(-pipe)<over/(1+max(over,0))/2)
        inputs['C-EP2'][cell]=dict(applicable=b16,epilogue_task_fraction=sum(v['epilogue_fraction']*v['mean_run_ns']*v['samples'] for v in gemm)/sum(v['mean_run_ns']*v['samples'] for v in gemm))
        # A sampled stage span is a lower bound; standalone merge is omitted.
        folder=HERE/f'raw/B_trace_{cell}/AT_pv/past575';spans=defaultdict(list)
        for t in csv.DictReader((folder/'task_profile.tsv').open(),delimiter='\t'):
            if t['kind']=='attention':spans[(int(t['iteration']),int(t['stage']))].append(t)
        lower=statistics.mean((max(int(t['run_end']) for t in ts)-min(int(t['run_begin']) for t in ts))/1000 for ts in spans.values())
        inputs['C-AT4'][cell]=dict(applicable=cell.endswith('B1'),attention_merge_us_per_layer=lower,metric_detail='AT-pv sampled attention frontier lower bound; excludes separate merge')
        inputs['C-GV2'][cell]=dict(applicable=cell.endswith('B1'),nonpaged_gemv_selected=0,paged_faster=False,reason='all GEMV arms slower than same-geometry MMA controls')
    # PlanFamily is decided after D1 because per-past winners do not exist yet.
    decisions=phase_c(inputs)
    protocols={p.parent.parent.name:load(p) for p in HERE.glob('raw/B6_protocol_*/processes/summary.json')}
    if any(not p['complete'] or p['passed']!=50 or p['failed'] for p in protocols.values()):raise ValueError('protocol failed')
    table(HERE/'results/T4_phase_b.tsv',rows);table(HERE/'results/T3_phase_b_resources.tsv',identities);table(HERE/'results/T5_phase_b_tasks.tsv',profiles)
    write(HERE/'results/phase_b_acceptance.json',dict(steps=len(queue),builds=len(identities),correctness={c:dict(arms=len(v),c1=sum(x['c1'] for x in v),c2=sum(x['c2']['pass'] for x in v)) for c,v in correctness.items()},protocols={k:{f:v[f] for f in ('passed','failed','complete','processes_per_case')} for k,v in protocols.items()},guards=guards,canary_rounds=canaries,matrix=matrix))
    write(HERE/'results/phase_c_inputs.json',inputs);write(HERE/'phase_c_decision.json',decisions)
    print(json.dumps({k:dict(status=v['status'],cells=v.get('cells')) for k,v in decisions.items()}))

def archive():
    queue=load(HERE/'queue/queue_phase_b.json');selected=set()
    for step in queue:
        for p in (HERE/'raw'/step['name']).rglob('*'):
            if p.is_file() and p.suffix in ('.json','.jsonl','.tsv','.log','.txt','.patch'):selected.add(p)
        selected.update(p for p in (HERE/'scheduler').glob(step['name']+'.*') if p.suffix in ('.log','.done'))
    selected.update((HERE/'results').glob('*.json'));selected.update((HERE/'results').glob('*.tsv'))
    selected.update([HERE/'scheduler/state.json',HERE/'scheduler/progress.tsv',HERE/'phase_c_decision.json'])
    with (HERE/'raw/phase_b_evidence_manifest.tsv').open('w') as f:
        f.write('path\tsha256\tbytes\n')
        for p in sorted(selected):f.write(f'{p.relative_to(HERE)}\t{sha(p)}\t{p.stat().st_size}\n')
    with tarfile.open(HERE/'raw/phase_b_completed.tar.xz','w:xz',preset=1) as tar:
        for p in sorted(selected):tar.add(p,arcname=str(p.relative_to(HERE)),recursive=False)
    print('Archived Phase B without binaries or generated CUDA')
if __name__=='__main__':
    import sys
    archive() if '--archive' in sys.argv else review()
