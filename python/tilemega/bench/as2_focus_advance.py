"""CPU follow-up for the reviewed B16 plans; writes jobs for the sole submitter."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tilemega.bench.as2_queue import REPO,RUNS,WORK,controlled,write
from tilemega.bench.as1_run import process_identity


def advance(directory):
    directory=controlled(directory);queue=json.loads((directory/'state.json').read_text())
    journal=directory/'cpu_followup.json'
    state=json.loads(journal.read_text()) if journal.exists() else dict(steps={})
    image=WORK/'build/retained/focus_generic_v20/as2_llm'

    def step(name,argv):
        old=state['steps'].get(name)
        if old:return old['status']=='pass'
        if name.endswith('_solve') or '_pack' in name or '_bind' in name:
            disk=subprocess.run([sys.executable,str(REPO/'scripts/as2_disk_status.py')],capture_output=True,text=True)
            (directory/(name+'_disk.json')).write_text(disk.stdout)
            if disk.returncode:
                state['resource_wait']=dict(step=name,reason='Project disk budget preflight');write(journal,state)
                return False
        log=directory/(name+'.log')
        state['steps'][name]=dict(status='running',argv=list(map(str,argv)),log=str(log));write(journal,state)
        with log.open('x') as stream:
            try:
                result=subprocess.run(['nice','-n','10',*map(str,argv)],stdout=stream,stderr=stream,timeout=900)
                state['steps'][name].update(status='pass' if result.returncode==0 else 'fail',exit_code=result.returncode)
            except subprocess.TimeoutExpired:
                state['steps'][name].update(status='blocked',reason='Finite CPU follow-up budget exhausted; partial artifacts retained')
        write(journal,state);return state['steps'][name]['status']=='pass'

    def enqueue(name,argv,out,batch,model,kind='functional',priority=10,admission=None):
        path=directory/'requests'/(name+'.json')
        if path.exists():return
        write(path,dict(id=name,argv=list(map(str,argv)),output=str(out),environment='as2',
            kind=kind,system='tilemega' if kind=='tilemega' else 'functional',batch=batch,model_id=model,
            physical_device=0,required_free_mib=16384,
            timeout_s=900 if kind=='tilemega' else 600,require_idle=True,priority=priority,prior_correctness=admission))

    controls=[v['job_directory'] for k,v in queue['jobs'].items()
              if k.startswith('focus_control0_v20_') and v['status']=='passed']
    costs=RUNS/'search/focus_control0_costs_v20.json'
    if len(controls)>=5 and not costs.exists():
        step('control_prices',[sys.executable,'-m','tilemega.bench.as2_native_prices','costs',
             '--jobs',*controls[:5],'--out',costs])
    for model,tag,initial,reference,decomposition in [
        ('qwen3_0_6b','qwen06','native_request_b16_fold_v1','qwen06_b16_fused1024_reference_v1',None),
        ('llama3_2_3b','llama3b','native_b16_request_decode_split2_v20','llama3b_b16_sdpa1024_reference_v20',
         REPO/'configs/ascend/as2_llama3b_b16_decode_split2.json')]:
        source=RUNS/'models'/model/initial;binding=json.loads((source/'request.json').read_text())
        profile_job='qwen06_b16_body_profile_v20' if tag=='qwen06' else 'llama3b_b16_split2_profile1024_v20'
        job=queue['jobs'].get(profile_job,{})
        if job.get('status')!='passed':continue
        for phase in ('prefill','decode'):
            profile=RUNS/'search'/f'{tag}_b16_{phase}_profile_v20.json'
            if not profile.exists():
                if not step(tag+'_'+phase+'_profile',[sys.executable,'-m','tilemega.bench.as2_body_profile',
                    '--binding',source,'--job',job['job_directory'],'--phase',phase,'--out',profile]):break
        else:
            if not costs.exists():continue
            selected={}
            for phase in ('prefill','decode'):
                profile=RUNS/'search'/f'{tag}_b16_{phase}_profile_v20.json'
                price=RUNS/'search'/f'{tag}_b16_{phase}_prices_v20.json'
                if not price.exists():
                    if not step(tag+'_'+phase+'_prices',[sys.executable,'-m','tilemega.bench.as2_native_prices','prices',
                        '--plan',binding['phases'][phase]['source_plan'],'--profile',profile,'--costs',costs,'--out',price]):break
                out=RUNS/'models'/model/f'focus_b16_{phase}_priced_v21';selected[phase]=out/'plan'
                if not (out/'import.json').exists():
                    split=decomposition and phase=='decode'
                    cmd=[WORK/'build/host/as2_compile_llm','emit-priced-decomposed' if split else 'emit-priced',
                         RUNS/'models'/model/('export_'+phase),out,'16','8','1088','128','cube-fp32-fold-softmax',price]
                    if split:cmd.append(decomposition)
                    if not step(tag+'_'+phase+'_solve',cmd):break
            else:
                packed=RUNS/'models'/model/'focus_b16_inputs_priced_v22'
                if not (packed/'inputs.json').exists():
                    rope=RUNS/'probes'/('qwen06_rope_table_v1' if tag=='qwen06' else 'llama3b_rope_table_v1')
                    if not step(tag+'_pack_v22',[sys.executable,'-m','tilemega.bench.as2_native',
                        '--model',model,'--plan',selected['prefill'],'--out',packed,'--past','0',
                        '--reuse',binding['source_inputs'],'--reuse-plan',binding['phases']['prefill']['source_plan'],
                        '--rope-table',rope]):continue
                final=RUNS/'models'/model/'focus_b16_request_priced_v22'
                if not (final/'request.json').exists():
                    if not step(tag+'_bind_v22',[sys.executable,'-m','tilemega.bench.as2_request','--prefill',selected['prefill'],
                        '--decode',selected['decode'],'--inputs',packed,'--out',final]):continue
                name=tag+'_b16_priced1024_v22';out=RUNS/'correctness'/name
                enqueue(name,[image,'request',final,out,1024,0,1,0],out,16,model)
                actual=queue['jobs'].get(name,{})
                numeric=RUNS/'correctness'/(name+'_numeric.json');ref=RUNS/'correctness'/reference
                if actual.get('status')=='passed' and (ref/'result.json').exists():
                    if not numeric.exists():
                        step(tag+'_priced_numerical',[sys.executable,'-m','tilemega.bench.as2_compare_request',
                             '--binding',final,'--native',out,'--reference',ref,'--out',numeric])
                    if numeric.exists() and json.loads(numeric.read_text())['status']=='pass':
                        name=model+'_b16_tilemega';timed=directory/'results'/name
                        enqueue(name,[image,'bench',final,timed,1024,1,3],timed,16,model,'tilemega',300,str(numeric))
    registered=state.setdefault('registered',[])
    artifacts=[costs]
    for model,tag in [('qwen3_0_6b','qwen06'),('llama3_2_3b','llama3b')]:
        artifacts.extend((RUNS/'models'/model).glob('focus_b16_*'))
        artifacts.extend((RUNS/'search').glob(tag+'_b16_*_v20.json'))
        artifacts.extend((RUNS/'correctness').glob(tag+'_b16_priced1024_v22_numeric.json'))
    for artifact in artifacts:
        if str(artifact) in registered or not artifact.exists():continue
        if artifact.is_dir() and not any((artifact/n).exists() for n in ('request.json','import.json')):continue
        subprocess.run([sys.executable,str(REPO/'scripts/as2_register.py'),'--path',str(artifact),
            '--stage','AS2-focus','--category','source','--completed'],check=True,stdout=subprocess.DEVNULL)
        registered.append(str(artifact))
    write(journal,state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--watch-hours',type=float,default=0)
    a=p.parse_args()
    if not 0<=a.watch_hours<=24:p.error('Finite CPU watch budget up to 24 hours required')
    deadline=time.monotonic()+a.watch_hours*3600
    stopping=[]
    def stop_after_current(signum,frame):stopping.append(signum)
    signal.signal(signal.SIGTERM,stop_after_current);signal.signal(signal.SIGINT,stop_after_current)
    write(controlled(a.directory)/'cpu_supervisor.json',dict(**process_identity(os.getpid()),finite_watch_hours=a.watch_hours))
    while True:
        advance(a.directory)
        if time.monotonic()>=deadline or stopping:break
        time.sleep(30)
