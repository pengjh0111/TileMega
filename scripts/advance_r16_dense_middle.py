#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Advance dense middle batches only after both representative winners qualify."""
from __future__ import annotations
import argparse,collections,copy,ctypes,hashlib,json,os,select,shutil,sys,time,traceback
from pathlib import Path
from collect_r16_candidates import (read_states,inspect_measurement,summarize_samples,load_driver,validate_numeric_tokens)
from r16_measurement_validation import file_sha,native_identity,json_digest

MODELS=('qwen3_0_6b','qwen3_1_7b','llama3_2_1b','llama3_2_3b','llama3_1_8b')
TERMINAL={'done','failed','skipped','not_run','cancelled'}
COORDS=('arch_paths','attention_impl','attention_page','activation_tma','qkv_epilogue',
        'cluster_dim','combine','nonpaged_la','gemv_load','moe_activation_bulk')

def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)

def family(candidate):
    identity=native_identity(candidate['decode']);plan=identity['plan']
    coordinates={k:plan.get(k)for k in COORDS}
    for key in ('activation_tma','qkv_epilogue','moe_activation_bulk'):
        coordinates[key]=int(coordinates[key]or identity.get('defines',{}).get('TILEMEGA_'+key.upper(),0))
    coordinates['cluster_dim']=int(coordinates['cluster_dim']or 1)
    coordinates['nonpaged_la']=bool(coordinates['nonpaged_la'])
    if not coordinates['nonpaged_la']:coordinates['combine']='stage'
    elif coordinates['combine']is None:coordinates['combine']='legacy_la'
    value=dict(arm=candidate['arm'],mode=candidate['mode'],loop=candidate['loop'],
               coordinates=coordinates,
               gemm_implementations=[g['impl']for g in plan['gemms']])
    return candidate['arm']+'-'+json_digest(value)[:10],value

def packing_pair(prefill,decode):
    """Mirror weight_recipes' shared-name contract using hash-bound manifests."""
    maps=[]
    for binary in (prefill,decode):
        native_identity(binary)
        buffers=read(str(binary)+'.plan.json')['buffers'];weights={}
        for b in buffers:
            if not b.get('recipe'):continue
            value=(b['dtype'],b['constant'],b['recipe'])
            if b['name']in weights and weights[b['name']]!=value:
                raise ValueError('conflicting weight recipe within phase: '+b['name'])
            weights[b['name']]=value
        if not weights:raise ValueError('phase contains no checkpoint recipes')
        maps.append(weights)
    common=maps[0].keys()&maps[1].keys()
    if not common:raise ValueError('phase weights have no shared recipe names')
    conflicts=[n for n in common if maps[0][n]!=maps[1][n]]
    if conflicts:raise ValueError('prefill/decode packing conflict: '+','.join(conflicts[:5]))
    return dict(shared_names=len(common),phase_recipe_sha256=[json_digest(x)for x in maps])

def token_digest(sample):
    path=Path(sample['measurement']);value=read(path)
    row=next(r for r in value['generation_runs']if r['N']==1024 and r['run']==3)
    return json_digest(read(path.parent/row['tokens_file']))

def selected_rows(report,model):
    chosen=[]
    for batch in (1,16):
        matches=[c for c in report['cells']if c['model']==model and c['batch']==batch]
        if len(matches)!=1 or matches[0].get('status')!='complete':return None
        cell=matches[0];rows=[r for r in cell['candidates']if r['candidate']==cell['selected']]
        if len(rows)!=1:raise ValueError('representative winner absent or ambiguous')
        row=rows[0];samples=row['samples']
        if len(samples)!=3 or len({s['pid']for s in samples})!=3:
            raise ValueError('representative winner lacks three fresh processes')
        if len({token_digest(s)for s in samples})!=1:
            raise ValueError('representative final processes disagree on tokens')
        chosen.append((batch,cell['selected'],samples))
    return chosen

def gpu_command(root,candidate,stage,rnd=0):
    expected=candidate['physical_device_expected'];node=expected['numa_node']
    return ['python3',str(root/'driver/run_r16_with_device_identity.py'),'--out','{attempt_dir}',
            '--expected-uuid',expected['uuid'],'--expected-pci',expected['pci'],'--numa-node',str(node),'--',
            'numactl',f'--cpunodebind={node}',candidate['python'],candidate['script'],'execute',
            '--candidate',candidate['file'],'--stage',stage,'--round-index',str(rnd),'--out','{attempt_dir}/measurements']

def job(root,c,name,command,gpu,timing=False,after=()):
    return dict(name=name,command=command,gpu=gpu,timing=timing,after=list(after),priority=60,
                timeout_s=14400,needs_free_mib=30000 if c['model']=='llama3_1_8b'else 24000,
                cwd=c['root'],env=c['env'])

def chain(root,c,scheduler):
    jobs=[];previous=[]
    for stage in ('smoke','longcheck','pilot'):
        name=c[stage+'_job'];jobs.append(job(root,c,name,gpu_command(root,c,stage),True,stage=='pilot',previous))
        if stage!='pilot':
            gate=name+'_numeric_gate';cmd=['python3',str(root/'validator/validate_r16_numeric_gate.py'),'--root',c['root'],
                '--scheduler-root',str(scheduler),'--candidate',c['file'],'--job',name,'--stage',stage,
                '--expected-uuid',c['physical_device_expected']['uuid'],'--out','{attempt_dir}/numeric-gate.json']
            jobs.append(job(root,c,gate,cmd,False,after=[name]));previous=[gate]
    return jobs

class Campaign:
    def __init__(self,repo,out):
        self.repo=repo.resolve();self.out=out.resolve();self.rep=self.repo/'runs/r16/rf2-final-v1'
        self.scheduler=self.repo/'runs/r16/scheduler-rf2-final-v1';self.queues=self.repo/'runs/r16/queue-rf2-final-v1'
        self.master=read(self.repo/'runs/r16/native30-gpu7-master-v1/plan.json')
        self.runtime=read(self.rep/'runtime-source.json')
        sys.path.insert(0,str(self.repo/'python'))
        from tilemega.bench.queue_check import check
        self.queue_check=check
        self.expected={2:dict(uuid='GPU-701b606f-5e57-e434-5e2b-d5cc774c3c9e',pci='0000:41:00.0',numa_node=0),
                       4:dict(uuid='GPU-1f2fad96-e938-be19-e775-94e8ae7501a4',pci='0000:86:00.0',numa_node=1)}
    def freeze(self):
        if (self.out/'frozen.json').exists():return
        self.out.mkdir(parents=True,exist_ok=True)
        for p in (self.rep/'driver').glob('*.py'):
            d=self.out/'driver'/p.name;d.parent.mkdir(exist_ok=True);shutil.copy2(p,d)
        shutil.copy2(self.repo/'runs/r16/device-identity-wrapper-v1/run_r16_with_device_identity.py',self.out/'driver/run_r16_with_device_identity.py')
        validator=self.out/'validator';validator.mkdir(exist_ok=True)
        for name in ['validate_r16_numeric_gate.py','collect_r16_candidates.py','r16_measurement_validation.py',
                     'summarize_r16.py','r16_runtime_contract.py','r16_argument_contract.py','r16_kernel_resources.py']:
            shutil.copy2(self.repo/'scripts'/name,validator/name)
        write(self.out/'frozen.json',dict(at=time.time(),runtime_source=self.runtime,
            files={str(p.relative_to(self.out)):file_sha(p)for d in ['driver','validator']for p in (self.out/d).glob('*.py')}))
    def bind(self,base,decode,arm,key):
        c=copy.deepcopy(base);model=c['model'];batch=c['batch'];gpu=2 if MODELS.index(model)%2==0 else 4;node=self.expected[gpu]['numa_node']
        if arm=='sm80':c['prefill']=str(Path(base['decode']).parent/'prefill-sm80.so')
        c.update(arm=arm,id=key,gpu=gpu,decode=str(decode),system='TileMega',native_system_label='TileMega',family=arm,base_variant=arm=='mma',
                 mode='L2',loop=0,script=str(self.out/'driver/run_r16_native_candidate.py'),
                 script_sha256=file_sha(self.out/'driver/run_r16_native_candidate.py'),runtime_source=self.runtime,
                 runtime_contract_sha256=file_sha(self.out/'driver/r16_runtime_contract.py'),
                 argument_contract_sha256=file_sha(self.out/'driver/r16_argument_contract.py'),
                 kernel_resource_helper_sha256=file_sha(self.out/'driver/r16_kernel_resources.py'),
                 physical_device_expected=self.expected[gpu],identity_sha256={},phase_sources={},
                 variant=dict(campaign='final dense middle after qualified B1/B16 winners',family=key,selection='pilot1 then at most three finalists'))
        c['exports']={phase:str(self.repo/f'runs/r16/exports/{model}/{phase}/export.json')for phase in ['prefill','decode']}
        c['export_sha256']={phase:file_sha(path)for phase,path in c['exports'].items()}
        for phase in ('prefill','decode'):
            identity=native_identity(c[phase]);c['identity_sha256'][phase]=file_sha(c[phase]+'.identity.json')
            c['phase_sources'][phase]={k:identity['source'][k]for k in ['root','head','compiler_sha256']}
            if phase=='decode':c.update({k:identity['plan'].get(k)for k in COORDS})
        source=c['phase_sources']['decode'];c.update(snapshot_source=source['root'],snapshot_head=source['head'],compiler_sha256=source['compiler_sha256'])
        c['packing_contract']=packing_pair(c['prefill'],c['decode'])
        c['env']=dict(CUDA_VISIBLE_DEVICES=str(gpu),TILEMEGA_DEVICE_INDEX=str(gpu),TILEMEGA_NUMA_NODE=str(node),TILEMEGA_GPU_PID_MODE='namespace_limited',
            PYTHONPATH=self.runtime['root']+'/python',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8',
            TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{gpu}.lock',TMPDIR=str(self.repo/'runs/r16/tmp/compiler'))
        directory=self.out/'templates'/model/f'B{batch}';c['file']=str(directory/(key+'.json'))
        stem=f'final_middle_v1_{model}_B{batch}_{key}'
        for stage in ('smoke','longcheck','pilot'):c[stage+'_job']=stem+'_'+stage
        c['numeric_jobs']=[c['smoke_job'],c['longcheck_job']]
        write(c['file'],c);load_driver(c).check_pair(c);return c
    def prepare(self):
        if (self.out/'published.json').exists():raise ValueError('cannot replace prepared definitions after GPU publication')
        self.freeze();templates=[];rejected=[]
        for cell in self.master['cells']:
            if cell['model']not in MODELS or cell['batch']not in [2,4,8]:continue
            base=cell['candidates'][0];directory=Path(base['decode']).parent
            for arm in ['mma','h1','h1h4','sm80']:
                decode=directory/f'decode-{arm}.so';prototype=dict(base,arm=arm,decode=str(decode));key,signature=family(prototype)
                try:c=self.bind(base,decode,arm,key)
                except Exception as error:
                    rejected.append(dict(model=base['model'],batch=base['batch'],family=key,arm=arm,reason=str(error)));continue
                definition=dict(model=c['model'],batch=c['batch'],family=key,signature=signature,candidate=c)
                write(Path(c['file']).with_suffix('.chain.json'),chain(self.out,c,self.scheduler));templates.append(definition)
        write(self.out/'templates.json',templates)
        write(self.out/'plan.json',dict(root=str(self.repo),run_id='final_dense_middle_v1',protocol=dict(isl=64,osl=1024,warmup=1,repeats=3,pilot_rounds=1,final_fresh_rounds=3,maximum_candidates=3),cells=[]))
        write(self.out/'preparation.json',dict(cells=15,available_template_candidates=len(templates),jobs_defined=sum(len(chain(self.out,t['candidate'],self.scheduler))for t in templates),rejected_templates=rejected,gpu_jobs_published=0,representative_gate='Both B1/B16 selected three fresh final samples and token equality required before model publication'))
    def publish(self,gpu,name,jobs):
        q=self.queues/f'gpu{gpu}';existing=[j for p in q.glob('queue_*.json')for j in read(p)];state=read(self.scheduler/f'gpu{gpu}/state.json')
        if (q/name).exists():
            if read(q/name)!=jobs:raise ValueError('immutable queue publication differs')
            return
        check=self.queue_check(existing,jobs,state);write(self.out/'queue-checks'/(name+'.json'),check)
        if not check['pass_check']:raise ValueError(check)
        write(q/name,jobs)
    def verify_representatives(self,model,state,cards):
        report=read(self.rep/'selection-report.json');selection=selected_rows(report,model)
        if selection is None:return None
        definitions=[c for p in ['qwen06-plan.json','dense-plan.json']for cell in read(self.rep/p)['cells']for c in cell['candidates']]
        winners=[]
        for batch,name,samples in selection:
            c=next(c for c in definitions if c['model']==model and c['batch']==batch and c['id']==name)
            if c['mode']!='L2' or c['loop']!=0:raise ValueError('selected executor unsupported; no substitution allowed')
            verified=[]
            for sample in samples:verified.append(inspect_measurement(self.repo,state,cards,sample['job'],sample['round'],c))
            summary=summarize_samples(verified)
            if not summary['protocol_complete']or len({token_digest(s)for s in verified})!=1:raise ValueError('representative fresh protocol differs')
            validate_numeric_tokens(self.repo,state,c,verified)
            for n in c['numeric_jobs']:
                gate=n+'_numeric_gate';record=state[gate];result=read(Path(record['out'])/'numeric-gate.json')
                if record['status']!='done'or record['exit_code']!=0 or not result['passed']or result['candidate_sha256']!=file_sha(c['file']):raise ValueError('representative numeric gate not valid')
            key,signature=family(c);winners.append(dict(candidate=c,family=key,signature=signature,samples=verified))
        return dict(report_sha256=file_sha(self.rep/'selection-report.json'),winners=winners)
    def missing(self,base,winner):
        # Commands are proposals only. Missing geometry/import graphs are explicit prerequisites.
        from run_r16_hopper_candidates import compile_command
        c=copy.deepcopy(winner['candidate']);c['batch']=base['batch'];c['layout']=base['layout'];c['exports']=base.get('exports',c['exports'])
        c['prefill']=str(Path(base['decode']).parent/'prefill-sm80.so')if c['arm']=='sm80'else base['prefill']
        output=self.out/'missing-artifacts'/base['model']/f"B{base['batch']}"/(winner['family']+'.so')
        command=compile_command(c,'decode',output)
        plan=native_identity(c['decode'])['plan'];requirements=[]
        try:native_identity(c['prefill'])
        except Exception as error:requirements.append('Prefix-safe middle prefill must be built or adopted: '+str(error))
        if c['arm'].startswith('gemv'):
            receipt=Path(c['decode']).parent/f"build-{plan['gemv_load']}-receipt.json"
            if not receipt.exists():
                requirements.append('Exact official GEMV ImportPlan compile receipt missing; generic dense compile command is not sufficient')
                command=None
            else:
                command=read(receipt)['command'];command[3]=str(output)
                command[command.index('--batch')+1]=str(base['batch']);c['compiler']=command[0]
                requirements.append('Verify the immutable official ImportPlan graph supports dynamic batch before rebinding --batch; a static B1 graph is ineligible')
        if plan.get('nonpaged_la'):
            command+=['--nonpaged-la','1','--combine',plan['combine'],'--cluster-dim',str(plan['cluster_dim'])]
        if not Path(c['compiler']).is_file() or file_sha(c['compiler'])!=c['compiler_sha256']:requirements.append('Recorded compiler path SHA changed or absent; resolve an immutable official compiler receipt before building')
        return dict(model=base['model'],batch=base['batch'],family=winner['family'],required_signature=winner['signature'],representative_candidate=c,
                    proposed_official_cpu_compile_command=command,prerequisites=requirements,output=str(output),fallback_substituted=False)
    def advance(self):
        state,cards=read_states(self.repo,[self.scheduler]);plan=read(self.out/'plan.json');templates=read(self.out/'templates.json');published=read(self.out/'published.json')if(self.out/'published.json').exists()else{};missing=[];waiting=[]
        adoptions=read(self.out/'adoptions.json')if(self.out/'adoptions.json').exists()else{}
        for model in MODELS:
            wanted_keys=[f'{model}_B{b}'for b in [2,4,8]]
            if all(k in published for k in wanted_keys):continue
            try:proof=self.verify_representatives(model,state,cards)
            except Exception as error:
                waiting.append(dict(model=model,error=str(error)));continue
            if proof is None:waiting.append(model);continue
            choices={w['family']:w for w in proof['winners']}
            for batch in [2,4,8]:
                try:
                    key=f'{model}_B{batch}'
                    if key in published:continue
                    base=next(c['candidates'][0]for c in self.master['cells']if c['model']==model and c['batch']==batch)
                    mma=next(t for t in templates if t['model']==model and t['batch']==batch and t['candidate']['arm']=='mma')
                    pool={mma['family']:mma['candidate']};blocked=[]
                    for family_key,w in choices.items():
                        found=next((t['candidate']for t in templates if t['model']==model and t['batch']==batch and t['family']==family_key),None)
                        adoption=adoptions.get(key+'_'+family_key)
                        if adoption:
                            proposed=read(adoption);actual_key,_=family(proposed)
                            if actual_key!=family_key:raise ValueError('adopted family silently differs')
                            found=self.bind(base,proposed['decode'],proposed['arm'],family_key)
                        if found:pool[family_key]=found
                        else:blocked.append(self.missing(base,w))
                    if blocked:missing.extend(blocked);continue
                    if len(pool)>3:raise ValueError('middle family pool exceeds two representative winners plus MMA')
                    candidates=list(pool.values());gpu=candidates[0]['gpu'];jobs=[j for c in candidates for j in chain(self.out,c,self.scheduler)]
                    cell=dict(model=model,batch=batch,gpu=gpu,candidates=candidates,representative_selection=proof)
                    # Publish explicit plan before the queue; no job can bypass the model-level gate.
                    prior=next((x for x in plan['cells']if x['model']==model and x['batch']==batch),None)
                    if prior is None:plan['cells'].append(cell)
                    elif prior['candidates']!=cell['candidates']:raise ValueError('immutable middle cell candidates changed')
                    write(self.out/'plan.json',plan)
                    name=f'queue_final_middle_v1_{key}.json';self.publish(gpu,name,jobs)
                    published[key]=dict(queue=name,candidates=[c['id']for c in candidates],representative_report_sha256=proof['report_sha256']);write(self.out/'published.json',published)
                except Exception as error:
                    missing.append(dict(model=model,batch=batch,phase='publication',error=str(error)))
        write(self.out/'missing-builds.json',missing);self.finals(state,cards,plan)
        write(self.out/'progress.json',dict(at=time.time(),published_cells=len(published),waiting_representative_models=waiting,missing_artifacts=missing,final_report=str(self.out/'selection-report.json')))
    def finals(self,state,cards,plan):
        decisions=read(self.out/'finalists.json')if(self.out/'finalists.json').exists()else{};report=[]
        for cell in plan['cells']:
            try:
                key=f"{cell['model']}_B{cell['batch']}";pool=cell['candidates']
                if key not in decisions and all(state.get(c['pilot_job'],{}).get('status')in TERMINAL for c in pool):
                    valid=[];rejected=[]
                    for c in pool:
                        try:s=inspect_measurement(self.repo,state,cards,c['pilot_job'],0,c);valid.append((s['e2e_seconds'],c,s))
                        except Exception as e:rejected.append(dict(candidate=c['id'],reason=str(e)))
                    valid.sort(key=lambda x:(x[0],x[1]['id']));chosen=valid[:3];jobs=[]
                    for _,c,s in chosen:
                        for rnd in range(3):
                            name=f"rf2_final_dense_middle_v1_{cell['model']}_B{cell['batch']}_final{rnd}_{c['id']}"
                            jobs.append(job(self.out,c,name,gpu_command(self.out,c,'final',rnd),True,True,[c['pilot_job']]))
                    if jobs:self.publish(cell['gpu'],f'queue_final_middle_finals_v1_{key}.json',jobs)
                    decisions[key]=dict(selected_for_final=[c['id']for _,c,_ in chosen],rejected=rejected);write(self.out/'finalists.json',decisions)
                d=decisions.get(key);rows=[];pending=[];rejected=list(d['rejected'])if d else[]
                if d:
                    for c in pool:
                        if c['id']not in d['selected_for_final']:continue
                        samples=[]
                        for rnd in range(3):
                            name=f"rf2_final_dense_middle_v1_{cell['model']}_B{cell['batch']}_final{rnd}_{c['id']}"
                            if state.get(name,{}).get('status')not in TERMINAL:pending.append(name);continue
                            try:samples.append(inspect_measurement(self.repo,state,cards,name,rnd,c))
                            except Exception as e:rejected.append(dict(candidate=c['id'],job=name,reason=str(e)))
                        if len(samples)==3:
                            try:
                                summary=summarize_samples(samples)
                                if not summary['protocol_complete']or len({token_digest(s)for s in samples})!=1:raise ValueError('middle final processes disagree')
                                validate_numeric_tokens(self.repo,state,c,samples)
                                rows.append(dict(candidate=c['id'],samples=samples,**summary))
                            except Exception as e:rejected.append(dict(candidate=c['id'],reason=str(e)))
                selected=min(rows,key=lambda r:(r['e2e_seconds'],r['candidate']))['candidate']if rows and not pending and d else None
                report.append(dict(model=cell['model'],batch=cell['batch'],status='complete'if selected else'rejected'if d and not pending else'pending',selected=selected,candidates=rows,rejected=rejected,default_promoted=False))
            except Exception as error:
                report.append(dict(model=cell['model'],batch=cell['batch'],status='blocked',selected=None,candidates=[],rejected=[dict(reason=str(error))],default_promoted=False))
        write(self.out/'selection-report.json',dict(cells=report,default_promoted=False))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--out',type=Path,required=True);p.add_argument('--prepare',action='store_true');a=p.parse_args();c=Campaign(a.root,a.out)
    if a.prepare:c.prepare();return
    lib=ctypes.CDLL(None);fd=lib.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC);assert fd>=0
    for directory in [c.rep]+[c.scheduler/f'gpu{i}'for i in range(5)]:assert lib.inotify_add_watch(fd,os.fsencode(directory),0x8|0x80)>=0
    while True:
        try:c.advance()
        except Exception:
            with(c.out/'events.jsonl').open('a')as f:f.write(json.dumps(dict(at=time.time(),error=traceback.format_exc()))+'\n')
        result=read(c.out/'selection-report.json')if(c.out/'selection-report.json').exists()else{}
        if len(result.get('cells',[]))==15 and all(x['status']in ['complete','rejected']for x in result['cells']):break
        if select.select([fd],[],[],55)[0]:
            try:
                while os.read(fd,1<<20):pass
            except BlockingIOError:pass
    os.close(fd)
if __name__=='__main__':main()
