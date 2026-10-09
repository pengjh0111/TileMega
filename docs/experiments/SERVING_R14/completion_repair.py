#!/usr/bin/env python3
"""Validate the three implementation repairs without reusing old D1 choices."""
import argparse,json,statistics
from pathlib import Path
from make_phase0 import HERE,ROOT,PYTHON,write
from phase_a import run
from phase_d_r14 import prepare as prepare_d
from tilemega.build.identity import verify,sha

TAG='_v3'
JOBS=HERE/'completion_builds.json'
ARMS=HERE/'phase_d_baseline_arms_v3.json'

def jobs():return json.loads(JOBS.read_text())
def arms():return json.loads(ARMS.read_text())
def destination(action):return HERE/'raw'/('R_complete_'+action)

def prepare():
    import phase_d_r14
    phase_d_r14.TAG=TAG;prepare_d()
    out=destination('prepare')
    run(['cmake','--build','build-phase12','--target','serving_task_profile_test','serving_gemv_test','-j','6'],out/'build_tests.log',1800)
    fixture=next(j['export'] for j in jobs() if j.get('cell')=='llama_B1' and not j.get('base_so'))
    run([ROOT/'build-phase12/tilemega-unit','serving_model_plan_test',fixture,'decode'],out/'model_plan.log',300)
    common=['/usr/local/cuda/bin/nvcc','-std=c++17','-O3','--expt-relaxed-constexpr',
            '-I'+str(ROOT/'include'),'-I'+str(ROOT/'third_party/cutlass/include'),
            '-I'+str(ROOT/'third_party/cutlass/tools/util/include')]
    records=[]
    for arch in (80,89,90,100,120):
        for name in ('serving_gemv_dispatch_test','serving_task_profile_test'):
            obj=out/f'{name}_sm{arch}.o'
            command=common+[f'-arch=sm_{arch}',f'-DTILEMEGA_ARCH_ID={arch*10}',
                '-DTILEMEGA_DN_VECTOR_SUMS=1','-DTILEMEGA_SWIGLU_U=4',
                '-c',ROOT/f'test/unit/{name}.cu','-o',obj]
            run(command,out/f'{name}_sm{arch}.log',900)
            records.append(dict(arch=arch,test=name,path=str(obj),sha256=sha(obj),compiled=True,executed=False))
    write(out/'architecture_checks.json',records)

def numeric():
    run(['ctest','--test-dir','build-phase12','-R','^(serving_gemv|serving_task_profile)$','--output-on-failure'],destination('numeric')/'tests.log',300)

def smoke():
    result=[]
    for job in jobs():
        if job.get('base_so'):continue
        library=str(Path(job['out'])/'plan.so');arm=arms()[job['cell']]
        out=destination('smoke')/job['cell']/job['label']
        run([PYTHON,'-m','tilemega.serving.smoke','--so',library,'--model',arm['model_path'],
             '--batch',job['batch'],'--steps','64','--out',out],out/'run.log',600)
        report=json.loads((out/'smoke.json').read_text())
        if not report['pass']:raise ValueError('repair smoke failed: '+library)
        result.append(dict(cell=job['cell'],label=job['label'],artifact_id=verify(library)['artifact_id'],pass_=True))
    write(destination('smoke')/'results.json',result)

def correctness(label):
    job=next(j for j in jobs() if j['label']==label);arm=arms()[job['cell']]
    library=str(Path(job['out'])/'plan.so');out=destination('c1')/label
    prompts=ROOT/f"docs/experiments/SERVING_R10/prompts/{arm['model']}_ids.json"
    run([PYTHON,'-m','tilemega.serving.check_modes','--model',arm['model_path'],
         '--prefill-so',arm['prefill'],'--decode-so',library,'--prompt-ids',prompts,
         '--batch','1','--steps','1024','--out',out/'modes'],out/'modes.log',2400)
    run([PYTHON,'-m','tilemega.serving.hf_check','--model',arm['model_path'],
         '--prompt-ids',prompts,'--generated',out/'modes/tokens_L1_separate.json',
         '--vllm-metrics',HERE/f"raw/A3_{job['cell']}/vllm_hf.json",'--skip-free-greedy',
         '--out',out/'hf.json'],out/'hf.log',2400)
    write(out/'identity.json',verify(library))

def compare():
    rows=[]
    for cell,arm in arms().items():
        old=json.loads((HERE/'phase_d_baseline_arms_v2.json').read_text())[cell]['decode']
        out=destination('compare')/cell
        run([PYTHON,ROOT/'docs/experiments/SERVING_R13/compare_kernels.py',
             '--reference',old,'--candidate',arm['decode'],'--out',out],out/'run.log',600,allowed=(0,1))
        rows.append(dict(cell=cell,**json.loads((out/'comparison.json').read_text())))
    write(destination('compare')/'results.json',rows)
    if not all(r[k] for r in rows for k in ('source_equal','sass_equal','resources_equal')):
        raise ValueError('default-path identity comparison differs; inspect before timing')

def trace():
    arm=arms()['llama_B1'];report=[]
    for kind in ('stage','task'):
        library=str(HERE/f'raw/R_complete_build/llama_B1/{kind}/plan.so')
        out=destination('trace')/kind
        run([PYTHON,'-m','tilemega.serving.trace','--model',arm['model_path'],
             '--prefill-so',arm['prefill'],'--decode-so',library,'--batch','1','--mode','L1',
             '--decode-loop','0','--'+kind,'--past-list','64,575,1000','--launches','16','--out',out],out/'run.log',900)
        if kind=='task':
            from ledger_r14 import tasks
            for folder in out.glob('past*'):
                values=tasks(folder/'task_profile.tsv')
                if not values.get('gemm') or any(not r['operand_ready_observed'] for r in values['gemm']['samples'] if r['bytes']):
                    raise ValueError('nonpaged GEMM has missing operand-ready observations')
                write(folder/'task_summary.json',values);report.append(str(folder/'task_summary.json'))
    write(destination('trace')/'results.json',dict(task_summaries=report,trace_only=True))

def overhead():
    rows={}
    for r in range(3):
        source=HERE/f'raw/R_complete_overhead_r{r}/llama_B1/round{r}.json'
        for label,record in json.loads(source.read_text())['arms'].items():
            if record['exit_code']:raise ValueError('failed overhead arm: '+label)
            identity=record['execution_identity']
            if bool(identity['trace'])!=(label!='baseline'):raise ValueError('overhead identity mismatch')
            group=rows.setdefault(label,dict(samples_ms=[],execution_id=identity['execution_id']))
            if group['execution_id']!=identity['execution_id']:raise ValueError('mixed overhead artifacts')
            group['samples_ms'].append((record['e2e_seconds']-record['ttft_seconds'])*1000/1023)
    base=statistics.median(rows['baseline']['samples_ms'])
    for row in rows.values():
        row['median_ms']=statistics.median(row['samples_ms']);row['relative']=row['median_ms']/base-1
    write(destination('overhead')/'results.json',dict(arms=rows,stage_within_two_percent=abs(rows['stage']['relative'])<=.02,
          semantics='trace time is diagnostic; never enters performance selection'))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','numeric','smoke','correctness','compare','trace','overhead'))
    p.add_argument('--label');a=p.parse_args()
    if a.action=='correctness':correctness(a.label)
    else:globals()[a.action]()
if __name__=='__main__':main()
