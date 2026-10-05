#!/usr/bin/env python3
"""One cell, one rotated round, one new process per registered arm."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path

def measurement_policy(root):
    supplied=os.environ.get('TILEMEGA_MEASUREMENT_POLICY')
    policy=Path(supplied) if supplied else root/'docs/experiments/SERVING_R11/ev2/measurement_policy.json'
    if supplied and not policy.is_file():raise RuntimeError('explicit measurement policy is missing: '+str(policy))
    if not policy.exists():policy=Path(__file__).resolve().parents[3]/'docs/experiments/SERVING_R11/ev2/measurement_policy.json'
    return policy

def loop_required(arm):
    return arm['kind'] == 'tm' and arm.get('decode_loop') in (True, 1, '1')

def replay_vllm_record(record):
    """Recover only the erroneous TM loop postcondition, preserving raw files."""
    if record.get('exit_code') != 3 or record.get('error') != 'requested loop was not used':
        return record
    out = Path(record['out'])
    command = json.loads((out/'command.json').read_text())['command']
    if not any(str(x).endswith('/vllm_baseline.py') for x in command):
        return record
    # The historical anchor applies this error only after subprocess exit 0.
    # Recheck the successful metrics and the enclosing guard before replay.
    metrics = json.loads((out/f"B{record['batch']}"/'measurements.json').read_text())
    guard = json.loads((out.parents[2]/'guard_result.json').read_text())
    if guard.get('code') != 0 or any(metrics.get(k) != record.get(k) for k in
            ('e2e_seconds','ttft_seconds','vllm_version','batch')):
        raise ValueError('vLLM loop-postcondition replay lacks matching clean evidence')
    return dict(record, exit_code=0, validation_replay='TM loop check excluded for vLLM',
                original_wrapper_exit_code=3)

def main():
    p=argparse.ArgumentParser();p.add_argument('--arms',type=Path,required=True);p.add_argument('--cell',required=True);p.add_argument('--round',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    data=json.loads(a.arms.read_text());arms=data[a.cell];arms=[r for r in arms if r.get('available',True)]
    offset=a.round%max(1,len(arms));arms=arms[offset:]+arms[:offset]
    index=['llama_B1','qwen3_B1','llama_B16','qwen3_B16'].index(a.cell)
    if index%2:arms.reverse()
    if not arms:
        print('no available arms for '+a.cell);return 3
    result={'cell':a.cell,'round':a.round,'order':[r['label'] for r in arms],'arms':{}}
    cell=a.out/a.cell;cell.mkdir(parents=True,exist_ok=True)
    for arm in arms:
        root=Path(arm['root']);out=cell/arm['label']/f'round{a.round}';out.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,**arm.get('env',{}),PYTHONPATH=str(root/'python'))
        for phase,record in arm.get('binaries',{}).items():
            if record.get('sha256') and hashlib.sha256(Path(arm[phase]).read_bytes()).hexdigest()!=record['sha256']:
                raise RuntimeError('binary changed: '+arm[phase])
        model=arm['model_path'];batch=str(arm['batch']);prompts=root/f"docs/experiments/SERVING_R10/prompts/{arm['model']}_ids.json"
        policy=measurement_policy(root)
        if arm['kind']=='vllm':
            libs=list((Path(arm['python']).parent.parent/'lib').glob('python*/site-packages/nvidia/cu*/lib'))
            env['LD_LIBRARY_PATH']=':'.join(map(str,libs))+':'+env.get('LD_LIBRARY_PATH','')
            cmd=[arm['python'],str(root/'python/tilemega/serving/vllm_baseline.py'),'--model',model,'--prompt-ids',str(prompts),'--batch',batch,'--out',str(out),'--max-tokens','1024','--policy',str(policy),'--warmup','1','--repeats','1']
        else:
            cmd=[arm['python'],'-m','tilemega.serving.measure','--model',model,'--prefill-so',arm['prefill'],'--decode-so',arm['decode'],'--prompt-ids',str(prompts),'--batch',batch,'--out',str(out),'--mode',arm['mode'],'--max-new-tokens','1024']
            if arm['kind']=='tm_old':
                env['TILEMEGA_GPU_LOCK']=str(out/'old-tool.lock')
                source=(root/'python/tilemega/serving/measure.py').read_text()
                for key,value in [('warmup','1'),('repeats','1'),('policy',str(policy))]:
                    if f'"--{key}"' in source:cmd+=['--'+key,value]
            else:cmd+=['--decode-loop',str(arm['decode_loop']) if arm['decode_loop']=='auto' else str(int(arm['decode_loop'])),'--prefill-mode',arm.get('prefill_mode','L1'),'--step-events',str(arm.get('step_events',1)),'--warmup','1','--repeats','1','--policy',str(policy)]
        (out/'command.json').write_text(json.dumps(dict(command=cmd,env=arm.get('env',{})),indent=2)+'\n')
        with (out/'stdout.txt').open('w') as stdout,(out/'stderr.txt').open('w') as stderr:
            code=subprocess.run(cmd,cwd=root,env=env,stdout=stdout,stderr=stderr).returncode
        metrics=out/('B'+batch)/'measurements.json' if arm['kind']=='vllm' else out/'measurements.json'
        record=dict(exit_code=code,out=str(out))
        if code==0 and metrics.exists():
            record.update(json.loads(metrics.read_text()))
            if loop_required(arm) and not record.get('decode_loop_used'):
                record.update(exit_code=3,error='requested loop was not used')
        result['arms'][arm['label']]=record
        if code==75:
            result['invalidated']=True;(cell/f'round{a.round}.json').write_text(json.dumps(result,indent=2)+'\n');return 75
    (cell/f'round{a.round}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(cell=a.cell,round=a.round,arms={k:v['exit_code'] for k,v in result['arms'].items()})))
    return 0
if __name__=='__main__':raise SystemExit(main())
