"""Audit completed MX8 evidence offline before publishing compact results."""
import argparse
import array
import collections
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))
from tilemega.build.identity import validate_snapshot_inputs, verify

spec=importlib.util.spec_from_file_location('mx8_bw',ROOT/'experiments/mx8/bandwidth.py')
bw=importlib.util.module_from_spec(spec);spec.loader.exec_module(bw)


def checked_ref(reference):
    path=Path(reference['path'])
    if bw.sha(path)!=reference['sha256']:
        raise ValueError('evidence hash changed: '+str(path))
    return path


def read(path):return json.loads(Path(path).read_text())


def equal(actual,expected):
    if not math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError(f'recomputed value differs: {actual} != {expected}')


def audit(queue,out):
    out.mkdir(parents=True,exist_ok=True)
    manifest=read(queue.parent/'ABLATION_QUEUE.json')
    if len(manifest['arms'])!=18 or manifest['errors']:
        raise ValueError('ablation builds incomplete')
    receipts={};counts=collections.Counter();durations=collections.Counter();intervals=[]
    checked=0;receipt_hashes={}
    for path in sorted(queue.glob('jobs/*/definition.json')):
        accepted=path.parent/'ACCEPTED.json'
        if not accepted.exists() or (path.parent/'FAILED.json').exists():
            raise ValueError('job not accepted: '+path.parent.name)
        item=read(accepted)
        for key in ('definition','process','guard','stdout'):
            checked_ref(item[key]);checked+=1
        definition=read(path);process=read(item['process']['path']);guard=read(item['guard']['path'])
        if process['returncode']!=0 or guard['code']!=0:
            raise ValueError('process or guard failed')
        for reference in definition['inputs']:
            checked_ref(reference);checked+=1
        counts[definition['kind']]+=1
        duration=process['end']-process['start']
        if duration<0:raise ValueError('negative process duration')
        durations[process['gpu']]+=duration
        intervals.extend(((process['start'],1,process['gpu']),(process['end'],-1,process['gpu'])))
        receipts[path.parent.name]=(item,definition)
        receipt_hashes[path.parent.name]=bw.sha(accepted)
    if counts!=dict(graph_probe=5,hardware=120,ablation_qualify=92,ablation_timing=54,routing=8):
        raise ValueError('unexpected queue coverage: '+str(counts))
    cards=collections.Counter();active=maximum=0
    for _,delta,gpu in sorted(intervals):
        active+=delta;cards[gpu]+=delta;maximum=max(maximum,active)
        if not 0<=cards[gpu]<=1 or not 0<=active<=4:
            raise ValueError('overlapping jobs violate GPU limits')
    correct=[];resources=[];artifacts={};timing_ranges=[]
    for arm in manifest['arms']:
        cell=read(checked_ref(arm['cell']));binary=checked_ref(arm['binary'])
        identity=verify(binary);validate_snapshot_inputs(ROOT,identity['source'])
        artifacts[str(binary)]=identity['artifact_id']
        checked_ref(cell['decode']);checked_ref(cell['prefill']);checked_ref(cell['tokens']);checked_ref(cell['generated_source'])
        frozen=read(cell['tokens']['path']);token_hashes=set();pids=set();hf_count=0
        for name in arm['qualifications']:
            accepted,definition=receipts[name]
            result_path=Path(accepted['folder'])/'result/RESULT.json';result=read(result_path)
            if result['artifact_id']!=identity['artifact_id'] or result['cell_sha256']!=arm['cell']['sha256']:
                raise ValueError('correctness execution identity mismatch')
            if not all(result[k] for k in ('passed','single_loop_exact_1024','frozen_reference_exact')):
                raise ValueError('correctness gate failed')
            values=read(result_path.parent/'tokens_1024.json')
            if len(values)!=cell['batch'] or any(len(row)!=1024 for row in values):
                raise ValueError('incorrect token shape')
            for a,b in zip(values,frozen):
                if a[:len(b)]!=b:raise ValueError('stored tokens differ from frozen main prefix')
            bits=array.array('i',(token for row in values for token in row))
            if bits.itemsize!=4 or sys.byteorder!='little':raise ValueError('token hash ABI differs')
            token_hash=hashlib.sha256(bits.tobytes()).hexdigest()
            if token_hash!=result['tokens_sha256']:raise ValueError('stored token hash differs')
            token_hashes.add(token_hash);pids.add(result['PID'])
            if 'hf' in result:
                if not result['hf']['pass'] or result['hf']['positions']!=cell['batch']*256:
                    raise ValueError('HF gate failed or has incorrect coverage')
                hf_count+=1
        if len(pids)!=5 or len(token_hashes)!=1 or hf_count!=1:
            raise ValueError('five-process or HF cohort incomplete')
        base=arm['qualifications'][0].rsplit('_check',1)[0]
        timing=[];timing_pids=set()
        for index in range(3):
            accepted,definition=receipts[base+f'_timing{index}']
            result=read(Path(accepted['folder'])/'result/RESULT.json')
            if result['artifact_id']!=identity['artifact_id'] or not result['passed']:
                raise ValueError('timing identity or status differs')
            for count in (1,1024):
                runs=[r for r in result['runs'] if r['count']==count]
                if len(runs)!=4 or sorted(r['run'] for r in runs)!=[0,1,2,3] or sum(r['warmup'] for r in runs)!=1:
                    raise ValueError('timing protocol incomplete')
            full=[r for r in result['runs'] if r['count']==1024 and not r['warmup']]
            first=[r for r in result['runs'] if r['count']==1 and not r['warmup']]
            e2e=statistics.median(r['e2e_ms'] for r in full);n1=statistics.median(r['e2e_ms'] for r in first)
            equal(result['tpot_ms'],(e2e-n1)/1023)
            equal(result['event_tpot_ms'],statistics.median((r['e2e_ms']-r['ttft_ms'])/1023 for r in full))
            equal(result['output_tokens_per_second'],cell['batch']*1024/(e2e/1000))
            timing.append(result['event_tpot_ms']);timing_pids.add(result['PID'])
        if len(timing_pids)!=3 or pids&timing_pids:raise ValueError('timing processes are not fresh')
        timing_ranges.append((max(timing)-min(timing))/statistics.median(timing))
        correct.append(dict(evidence='verified',model=cell['model'],batch=cell['batch'],variant=arm['mode'],
            fresh_processes=5,passes=5,single_loop_exact_1024=True,fresh_exact_1024=True,
            hf_positions_per_row=256,hf_pass=True,main_reference_tokens=len(frozen[0]),
            main_prefix_exact=True,timing_fresh_processes=3,timing_process_spread_fraction=timing_ranges[-1],
            tokens_sha256=next(iter(token_hashes)),artifact_id=identity['artifact_id'],binary_sha256=arm['binary']['sha256'],
            qualifications=json.dumps(arm['qualifications'])))
        shared=identity['shared_memory_bytes']
        if not 32768<shared<=65536:raise ValueError('unexpected sequence LDS domain')
        for name,resource in identity['resources'].items():
            if 'StageKernel' not in name:continue
            if resource['registers']>255 or resource['stack_bytes'] or resource['static_smem_bytes']:
                raise ValueError('stage resource gate failed')
            stage=re.search(r'StageKernelILi(\d+)ELb([01])E',name)
            resources.append(dict(evidence='verified',model=cell['model'],batch=cell['batch'],variant=arm['mode'],
                stage=int(stage[1]) if stage else name,terminal=bool(int(stage[2])) if stage else '',
                threads=identity['plan']['threads'],registers=resource['registers'],stack_bytes=resource['stack_bytes'],
                static_shared_bytes=resource['static_smem_bytes'],dynamic_shared_bytes=shared,
                runtime_resident_query_min=1,creation_passes=5,artifact_id=identity['artifact_id']))
    for row in correct:
        other=next(r for r in correct if r['model']==row['model'] and r['batch']==row['batch'] and r['variant']!=row['variant'])
        if row['tokens_sha256']!=other['tokens_sha256']:raise ValueError('cross-arm 1024-token gate failed')
        row['cross_arm_exact_1024']=True
    routing=[]
    for name,(accepted,definition) in receipts.items():
        if definition['kind']!='routing':continue
        folder=Path(accepted['folder'])/'result';result=read(folder/'RESULT.json')
        for phase in ('decode','prefill'):
            binary=Path(definition['command'][definition['command'].index('--'+phase)+1])
            identity=verify(binary);validate_snapshot_inputs(ROOT,identity['source'])
            if bw.sha(binary)!=result[phase+'_sha256']:raise ValueError('routing binary identity differs')
            artifacts[str(binary)]=identity['artifact_id']
        data=np.load(folder/'routes.npy',allow_pickle=False)
        config=read(Path('/root/models')/(result['model']+'_rand')/'config.json')
        experts=config.get('num_local_experts',config.get('num_experts'))
        if data.shape!=(1024,result['layers'],result['batch'],8) or not np.all(data<experts):
            raise ValueError('invalid routing array')
        if np.any(np.diff(np.sort(data,axis=-1),axis=-1)==0):raise ValueError('duplicate experts in a route')
        active=np.array([[len(np.unique(data[s,l])) for l in range(result['layers'])] for s in range(1024)])
        if bw.sha(folder/'routes.npy')!=result['capture_sha256']:raise ValueError('routing array hash differs')
        for a,b in zip(active.mean(axis=0),result['active_experts_per_layer_mean']):equal(float(a),b)
        equal(float(active.mean()),result['active_experts_mean'])
        routing.append(dict(evidence='verified',model=result['model'],batch=result['batch'],positions=1024,
            layers=result['layers'],active_experts_mean=result['active_experts_mean'],minimum=result['minimum'],maximum=result['maximum'],
            per_layer_mean=json.dumps(result['active_experts_per_layer_mean']),capture_sha256=result['capture_sha256'],
            decode_sha256=result['decode_sha256'],prefill_sha256=result['prefill_sha256'],raw_result=str(folder/'RESULT.json')))
    peaks,best,complete=bw.hardware_summary(queue)
    if not complete or len(peaks)!=80 or not all(r['qualified'] for r in peaks):
        raise ValueError('hardware coverage incomplete')
    bw.write_csv(out/'MX8_correctness_c550.csv',correct)
    bw.write_csv(out/'MX8_resources_c550.csv',resources)
    bw.write_csv(out/'MX8_routing_reference_c550.csv',sorted(routing,key=lambda r:(r['model'],r['batch'])))
    receipt=dict(status='pass',evidence='verified',audited_utc=datetime.now(timezone.utc).isoformat(),
        audit_source_sha256=bw.sha(Path(__file__)),
        completed_jobs=len(receipts),job_kinds=dict(counts),hash_references_checked=checked,
        accepted_receipts_digest=hashlib.sha256(json.dumps(receipt_hashes,sort_keys=True).encode()).hexdigest(),
        maximum_parallel_jobs=maximum,gpu_job_wall_hours={str(g):v/3600 for g,v in sorted(durations.items())},
        ablation_arms=18,correctness_fresh_processes=90,correctness_passes=90,hf_checks=18,
        timing_fresh_processes=54,max_timing_process_spread_fraction=max(timing_ranges),
        timing_arms_with_spread_over_2pct=[dict(model=r['model'],batch=r['batch'],variant=r['variant'],
            spread_pct=100*r['timing_process_spread_fraction']) for r in correct if r['timing_process_spread_fraction']>.02],
        hardware_jobs=120,hardware_configuration_rows=len(peaks),hardware_processes_per_configuration=5,
        measured_hbm_peak_GBs=best['median'],peak_gpu=best['gpu'],peak_config=best['config'],
        routing_captures=8,routing_positions_each=1024,resource_kernels=len(resources),
        max_MTreg=max(r['registers'] for r in resources),max_stack_bytes=0,max_static_shared_bytes=0,
        data_files_sha256={p.name:bw.sha(p) for p in sorted(list((queue.parent/'tables').glob('*.csv'))+
            list(out.glob('MX8_*.csv')))},
        frozen_timing_sources_sha256={r['timing_source']:r['timing_source_sha256'] for r in bw.frozen_timings()},
        artifact_ids=artifacts,external_baselines_remeasured=False,AB4b_remeasured=False)
    (out/'MX8_acceptance_c550.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='artifact_ids'},indent=2))
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,default=ROOT/'runs/mx8/queue')
    parser.add_argument('--out',type=Path,default=ROOT/'runs/mx8/acceptance')
    args=parser.parse_args();audit(args.queue,args.out)
