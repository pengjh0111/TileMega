"""Freeze actual dense controls and count source-level work without NPU access."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import struct
import subprocess

from tilemega.bench.as2_models import REPO, RUNS, WORK, DENSE
from tilemega.bench.as2_queue import controlled
from tilemega.bench.as2_body_profile import logical_bytes


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def stage_counts(s):
    kind,epilogue,tasks,rows,columns,k=s[17:23]
    variant=s[31];result=Counter(tasks=tasks)
    if kind in (0,6):
        tm,tn=(16,128) if variant else (32,64)
        nt=(columns+tn-1)//tn
        if not rows or not columns or not k:raise ValueError('Empty GEMM family')
        if kind==0:
            result['mmad_object_lifetimes']=tasks
            if epilogue==2:
                result['swiglu_tile_calls' if variant in (4,5) else 'pointwise16_calls']=tasks if variant in (4,5) else rows*((columns+31)//32)
            if epilogue==3:
                result['argmax_row_calls']=rows*nt
                if variant==6:result['argmax_batched_load_calls']=tasks
                else:result['argmax_scalar_scan_upper_bound_elements']=rows*columns
            if epilogue==1:result['residual_row_calls']=rows*nt
        else:
            result['combine_row_calls']=rows*nt
            if epilogue==1:result['residual_row_calls']=rows*nt
    if kind==13 and variant in (4,5,6,7,8,9):
        result['mmad_object_lifetimes']=2*tasks
    if kind==15:result['argmax_merge_vector_calls' if variant==1 else 'argmax_merge_scalar_reads']=tasks if variant==1 else tasks*s[24]
    return dict(result)


def phase_ledger(plan,phase,past):
    metadata=struct.unpack('<12I3Q',(plan/'metadata.bin').read_bytes())
    stages=list(struct.iter_unpack('<17Q16I',(plan/'stages.bin').read_bytes()))
    tasks=list(struct.iter_unpack('<7I',(plan/'tasks.bin').read_bytes()))
    offsets=struct.unpack('<'+'I'*(metadata[2]+1),(plan/'offsets.bin').read_bytes())
    queue=struct.unpack('<'+'I'*metadata[6],(plan/'queue.bin').read_bytes())
    if metadata[3]!=len(tasks) or metadata[4]!=len(stages) or sum(s[19] for s in stages)!=len(tasks):
        raise ValueError('Plan task/stage domain is inconsistent')
    totals=Counter();families=Counter();stage_rows=[]
    for i,s in enumerate(stages):
        counts=stage_counts(s);totals.update(counts)
        traffic=sum(logical_bytes(s,metadata,t,past) for t in range(s[19]))
        totals['source_logical_payload_bytes']+=traffic
        if s[17] in (0,6):
            families[(s[17],s[18],s[20],s[21],s[22],s[23],s[31])]+=1
        stage_rows.append(dict(stage=i,kind=s[17],variant=s[31],tasks=s[19],counts=counts,
                               source_logical_payload_bytes=traffic))
    mix=sum(bool(t[6]) for t in tasks)
    attention=sum(t[6] and stages[t[0]][17]==13 and stages[t[0]][31] in (4,5,6,7,8,9) for t in tasks)
    totals.update(queue_references=metadata[6],lowered_waits=metadata[5],mix_tasks=mix,
                  mix_ready_done_pairs=mix+2*attention,
                  acquire_data_calls=metadata[6]+mix+4*attention,
                  frontier_publications=metadata[6]+sum(a==b for a,b in zip(offsets,offsets[1:])),
                  body_timer_start_reads=metadata[6])
    recipes=[]
    for worker in range(metadata[1]):
        last=None;switches=0;owned=0
        for slot in range(offsets[worker],offsets[worker+1]):
            t=tasks[queue[slot]];s=stages[t[0]]
            recipe=('bf16' if s[17]==0 and s[23]<=1 else 'partial-fp32') if s[17]==0 else 'attention-fp32'
            switches+=last is not None and recipe!=last;last=recipe;owned+=1
        recipes.append(dict(worker=worker,owner_tasks=owned,coarse_dtype_switches=switches))
    return dict(phase=phase,past=past,groups=metadata[1],totals=dict(totals),stages=stage_rows,
                aic_recipes=recipes,gemm_families=[dict(kind=k[0],epilogue=k[1],m=k[2],n=k[3],k=k[4],
                split=k[5],variant=k[6],stages=v) for k,v in sorted(families.items())])


def freeze(out,prompt):
    out=controlled(out);prompt=Path(prompt).absolute()
    if out.exists():raise FileExistsError(out)
    if prompt.resolve()!=prompt or not prompt.is_relative_to(WORK/'Prompt'):raise ValueError('Use the private saved prompt')
    resume=json.loads((RUNS/'resume.json').read_text());out.mkdir()
    controls=[]
    for model in DENSE:
        for batch in (1,16):
            cell=f'{model}_b{batch}';binary=Path(resume['selected_images'][cell]['path'])
            binding=RUNS/resume['selected_bindings'][cell];origin='declared_default_historical_snapshot'
            if batch==16:
                name='opt_swiglu_b16_request_priced_v2' if model=='qwen3_1_7b' else 'opt_swiglu_b16_swiglu4_request_priced_v2'
                binding=RUNS/'models'/model/name
                study='opt_swiglu_qwen17_final_v1' if model=='qwen3_1_7b' else 'opt_swiglu_dense_priced_pilot_v1'
                state_path=RUNS/'bench'/study/'state.json';state=json.loads(state_path.read_text())
                matching=[v for v in state['jobs'].values() if v.get('status')=='collected' and
                          v['request']['argv'][2]==str(binding)]
                if not matching:raise ValueError('Latest declared study has no collected measurement')
                binary=Path(matching[0]['request']['argv'][0]);origin='reported_run_local_five_pairs' if model=='qwen3_1_7b' else 'reported_run_pilot'
                expected_binary=matching[0]['receipt']['executable_sha256']
            else:expected_binary=resume['selected_images'][cell]['sha256']
            if digest(binary)!=expected_binary:raise ValueError('Control image changed after its measurement')
            identity=json.loads((binding/'request.json').read_text())
            if identity['model_id']!=model or identity['batch']!=batch:raise ValueError('Control identity differs')
            config=REPO/'configs/ascend/as2_models'/f'{model}.json'
            if digest(config)!=identity['config_sha256']:raise ValueError('Changed model config')
            if digest(WORK/'weights'/model/'manifest.json')!=identity['weight_manifest_sha256']:
                raise ValueError('Control weights differ from the canonical manifest')
            phases=[];hashes={}
            for phase in ('prefill','decode'):
                plan=binding/phase
                hashes[phase]={n:digest(plan/n) for n in ('metadata.bin','stages.bin','tasks.bin','waits.bin','queue.bin','offsets.bin')}
                source=Path(identity['phases'][phase]['source_plan']).parent/'persistent_llm.cpp'
                hashes[phase]['source_sha256']=digest(source)
                for past in ((0,) if phase=='prefill' else (64,575,1086)):
                    phases.append(phase_ledger(plan,phase,past))
            controls.append(dict(cell=cell,binding=str(binding),declared_default=str(RUNS/resume['selected_bindings'][cell]),
                measurement_status=origin,request_sha256=digest(binding/'request.json'),binary=str(binary),binary_sha256=digest(binary),
                config_sha256=identity['config_sha256'],weight_manifest_sha256=identity['weight_manifest_sha256'],
                input_sha256=identity['input_sha256'],plan_hashes=hashes,ledgers=phases))
    manifest=dict(schema='tilemega.as2.taskbody_control_inventory.v1',status='verified_code_and_plan_counts_only',
        source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        prompt=str(prompt),prompt_sha256=digest(prompt),controls=controls,
        definitions={'counts':'Static source/Plan calls or labelled upper bounds, not measured cycles or TPOT shares',
                     'bytes':'Existing logical payload contract; neither measured HBM nor inferred physical GM traffic',
                     'dtype_switches':'Coarse consecutive owner recipe changes; Attention internally creates QK and PV objects',
                     'stage_acquire':'one per queue reference plus participant1/Attention-owner2/Attention-participant3',
                     'body_timer':'Unconditional source call in current executor; compiler instruction presence is not asserted'},
        unsupported_new_candidates_enabled=False,default_changed=False,device_jobs=0)
    (out/'controls.json').write_text(json.dumps(manifest,indent=2)+'\n')
    rows=[]
    for c in controls:
        for phase in c['ledgers']:
            if phase['phase']=='decode' and phase['past']!=64:continue
            rows.extend(dict(cell=c['cell'],phase=phase['phase'],**f) for f in phase['gemm_families'])
    with (out/'projection_families.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(controls=len(controls),projection_family_rows=len(rows),out=str(out),device_jobs=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prompt',type=Path,required=True);a=p.parse_args();freeze(a.out,a.prompt)
