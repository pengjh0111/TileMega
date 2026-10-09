#!/usr/bin/env python3
"""Bounded correctness-only completion; no calibration or timing acceptance."""
import argparse,json,os,subprocess,time
from pathlib import Path
from make_phase0 import HERE,ROOT,PYTHON,write
from phase_a import run
from tilemega.build.identity import verify,sha
from ledger_r14 import tasks

OUT=HERE/'raw/logic_completion'
JOBS=HERE/'logic_completion_builds.json'
ARMS=HERE/'logic_completion_arms.json'

def step(name,action):
    begin=time.time()
    def status(value):
        with (OUT/'progress.tsv').open('a') as f:f.write(f'{name}\t{value}\t{time.time()}\n')
    status('running')
    try:action();result=dict(name=name,status='done',elapsed_s=time.time()-begin)
    except (Exception,SystemExit) as error:
        if isinstance(error,SystemExit) and error.code==75:reason='allocation/occupancy failure, not a correctness verdict'
        else:reason='inspect the retained command log'
        result=dict(name=name,status='failed',error=repr(error),reason=reason,elapsed_s=time.time()-begin)
    status(result['status']);write(OUT/(name+'.json'),result)
    print(json.dumps(result),flush=True);return result

def prepare():
    run(['cmake','--build','build-phase12','--target','tilemega','tilemega-unit','serving_gemv_test',
         'serving_task_profile_test','attention_layout_test','serving_pvswap_test','serving_mma_pipeline_test',
         'serving_argmax_rows_test','monotonic_last_arriver_test','paged_gemm_test','-j','6'],OUT/'prepare/build.log',6000)
    run([PYTHON,'python/tilemega/fingerprint.py','--check','build-phase12/tools/tilemega'],OUT/'prepare/fingerprint.log')

def host(folder='host'):
    for case in json.loads((HERE/'logic_completion_host_cases.json').read_text()):
        if sha(case['path'])!=case['sha256']:raise ValueError('host fixture changed')
        run([ROOT/'build-phase12/tilemega-unit','serving_model_plan_test',case['path'],case['phase']],
            OUT/f"{folder}/{case['model']}_{case['phase']}.log",600)
    for script in ('test/python/integrated_selection.py','test/python/test_attention_selection.py',
                   'test/python/serving_identity.py','test/python/validation_guard.py','test/python/serving_build_orchestration.py',
                   'docs/experiments/SERVING_R14/test_selection.py',
                   'docs/experiments/SERVING_R14/test_phase_d.py'):
        run([PYTHON,script],OUT/folder/(Path(script).stem+'.log'),300)
    run(['ctest','--test-dir','build-phase12','-R','^(stage_flow|serving_pruning|operator_classes|skeleton_search_isolation|serving_search_rejection)$',
         '--output-on-failure'],OUT/folder/'ctest.log',600)

def arch():
    common=['/usr/local/cuda/bin/nvcc','-std=c++17','-O3','--expt-relaxed-constexpr',
            '-Iinclude','-Ithird_party/cutlass/include','-Ithird_party/cutlass/tools/util/include']
    rows=[]
    for arch in (80,89,90,100,120):
        for name in ('serving_gemv_dispatch_test','serving_task_profile_test'):
            obj=OUT/f'arch/{name}_sm{arch}.o'
            run(common+[f'-arch=sm_{arch}',f'-DTILEMEGA_ARCH_ID={arch*10}',
                '-DTILEMEGA_DN_VECTOR_SUMS=1','-DTILEMEGA_SWIGLU_U=4',
                '-c',ROOT/f'test/unit/{name}.cu','-o',obj],obj.with_suffix('.log'),900)
            rows.append(dict(arch=arch,name=name,path=str(obj),sha256=sha(obj),executed=False))
    write(OUT/'arch/binaries.json',rows)

def numeric():
    run(['ctest','--test-dir','build-phase12','-R',
        '^(attention_layout|serving_pvswap|serving_mma_pipeline|serving_argmax_rows|monotonic_last_arriver|serving_gemv|serving_task_profile|paged_gemm)$',
        '--output-on-failure'],OUT/'numeric/tests.log',1200)

def build():
    run([PYTHON,HERE/'builds_r14.py','--jobs',JOBS,'--out',OUT/'build/results.json','--keep-going'],OUT/'build/run.log',14400)
    rows=json.loads((OUT/'build/results.json').read_text())
    if any(r['exit_code'] for r in rows):raise ValueError('one or more builds failed; successful siblings remain usable')

def smoke(job,arm):
    library=str(Path(job['out'])/'plan.so');out=OUT/'smoke'/job['cell']/job['label']
    run([PYTHON,'-m','tilemega.serving.smoke','--so',library,'--model',arm['model_path'],
         '--batch',job['batch'],'--steps','64','--out',out],out/'run.log',900)
    if not json.loads((out/'smoke.json').read_text())['pass']:raise ValueError('token/KV mismatch')
    write(out/'identity.json',verify(library))

def real_model(job,arm):
    library=str(Path(job['out'])/'plan.so');out=OUT/'real'/job['label']
    prompts=ROOT/f"docs/experiments/SERVING_R10/prompts/{arm['model']}_ids.json"
    run([PYTHON,'-m','tilemega.serving.check_modes','--model',arm['model_path'],
         '--prefill-so',arm['prefill'],'--decode-so',library,'--prompt-ids',prompts,
         '--batch','1','--steps','64','--allow-shared-gpu','--out',out/'modes'],out/'modes.log',1800)
    run([PYTHON,'-m','tilemega.serving.hf_check','--model',arm['model_path'],
         '--prompt-ids',prompts,'--generated',out/'modes/tokens_L1_separate.json',
         '--vllm-metrics',HERE/f"raw/A3_{job['cell']}/vllm_hf.json",'--skip-free-greedy',
         '--out',out/'hf.json'],out/'hf.log',1800)
    write(out/'identity.json',verify(library))

def trace(job,arm):
    library=str(Path(job['out'])/'plan.so');out=OUT/'trace'/job['label']
    run([PYTHON,'-m','tilemega.serving.trace','--model',arm['model_path'],
         '--prefill-so',arm['prefill'],'--decode-so',library,'--batch','1','--mode','L1',
         '--decode-loop','0','--'+job['label'],'--past','575','--launches','1',
         '--allow-shared-gpu','--out',out],out/'run.log',900)
    if job['label']=='task':
        values=tasks(out/'past575/task_profile.tsv')
        if not values.get('gemm') or any(not r['operand_ready_observed'] for r in values['gemm']['samples'] if r['bytes']):
            raise ValueError('missing nonpaged GEMM operand readiness')
        write(out/'summary.json',values)

def main():
    p=argparse.ArgumentParser();p.add_argument('--host-only',action='store_true');args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.host_only:
        row=step('host_v2',lambda:host('host_v2'));raise SystemExit(0 if row['status']=='done' else 1)
    rows=[];rows.append(step('prepare',prepare))
    if rows[0]['status']!='done':raise SystemExit(1)
    for name,action in (('host',host),('arch',arch),('numeric',numeric),('build',build)):
        rows.append(step(name,action))
    arms=json.loads(ARMS.read_text())
    for job in json.loads(JOBS.read_text()):
        record=Path(job['out'])/'record.json'
        if not record.exists() or json.loads(record.read_text()).get('exit_code')!=0:continue
        arm=arms[job['cell']]
        if job.get('base_so'):rows.append(step('trace_'+job['label'],lambda j=job,a=arm:trace(j,a)))
        else:
            rows.append(step('smoke_'+job['cell']+'_'+job['label'],lambda j=job,a=arm:smoke(j,a)))
            if '_tn' in job['label']:rows.append(step('real_'+job['label'],lambda j=job,a=arm:real_model(j,a)))
    write(OUT/'result.json',dict(checks=rows,pass_=all(r['status']=='done' for r in rows),
          scope='logic correctness only; shared GPU allowed; no performance acceptance',
          timing_eligible=False,real_model_steps=64))
    raise SystemExit(0 if all(r['status']=='done' for r in rows) else 1)
if __name__=='__main__':main()
