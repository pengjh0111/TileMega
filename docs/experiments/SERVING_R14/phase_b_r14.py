#!/usr/bin/env python3
"""Numerical gates, identity-bound traces and correctness for the R14 matrix."""
import argparse,json,os,subprocess
from pathlib import Path
from make_phase0 import HERE,ROOT,PYTHON,write
from tilemega.build.identity import verify,sha
from phase_a import run

def declared(cell):return json.loads((HERE/'phase_b_arms.json').read_text())[cell]
def accepted(cell):return json.loads((HERE/f'B0c_{cell}_accepted.json').read_text())
def prepare():
    out=HERE/'raw/Bpre'
    run(['cmake','--build','build-phase12','--target','tilemega','tilemega-unit','-j','6'],out/'build.log',6000)
    run(['ctest','--test-dir','build-phase12','-R','serving_pruning|serving_task_index|serving_lag|serving_attention_legality|operator_classes|variant_resource|handoff_|stage_flow','--output-on-failure'],out/'host.log',1200)
    run([PYTHON,'python/tilemega/fingerprint.py','--check','build-phase12/tools/tilemega'],out/'fingerprint.log')
    for script in ('test/python/test_attention_selection.py','test/python/integrated_selection.py','test/python/serving_identity.py','docs/experiments/SERVING_R14/test_selection.py'):
        run([PYTHON,script],out/(Path(script).stem+'.log'),300)
    # Body numerical/architecture gates already passed; rerun arithmetic on
    # the frozen sources before admitting any newly built performance arm.
    for script in ('check_gemv.py','check_pvswap.py','check_ep_ra.py'):
        run([PYTHON,HERE/script,'compile'],out/(script+'.compile.log'),3600)

def numerical():
    for script in ('check_gemv.py','check_pvswap.py','check_ep_ra.py'):
        run([PYTHON,HERE/script,'numerical'],HERE/'raw/Bpre_numeric'/(script+'.log'),1200)

def smoke(cell):
    rows=[];passed=[]
    for arm in declared(cell):
        out=HERE/'raw'/('B0c_'+cell)/arm['label'];out.mkdir(parents=True,exist_ok=True)
        record=Path(arm['decode']).parent/'record.json'
        if not record.exists() or json.loads(record.read_text())['exit_code']:
            rows.append(dict(label=arm['label'],status='build_failed'));continue
        verify(arm['decode'])
        run([PYTHON,'-m','tilemega.serving.smoke','--so',arm['decode'],'--model',arm['model_path'],
             '--batch',arm['batch'],'--steps',64,'--out',out],out/'run.log',300,allowed=(0,1,3))
        result=json.loads((out/'smoke.json').read_text()) if (out/'smoke.json').exists() else {'pass':False}
        rows.append(dict(label=arm['label'],status='passed' if result['pass'] else 'smoke_failed'))
        if result['pass']:passed.append(arm)
    write(HERE/f'raw/B0c_{cell}/results.json',rows)
    if not any(a['label']=='baseline' for a in passed):raise ValueError('baseline failed; no timing for this cell')
    write(HERE/f'B0c_{cell}_accepted.json',passed)
    labels={a['label'] for a in passed}
    for i in range(1,6):
        source=json.loads((HERE/f'B{i}_arms.declared.json').read_text())[cell]
        write(HERE/f'B{i}_{cell}_arms.json',{cell:[a for a in source if a['label'] in labels]})
    print(cell,'accepted',len(passed),'excluded',len(rows)-len(passed))

def trace(cell):
    for arm in accepted(cell):
        so=Path(arm['decode']).parent.with_name(Path(arm['decode']).parent.name+'_task')/'plan.so'
        record=so.parent/'record.json'
        if not record.exists() or json.loads(record.read_text())['exit_code']:continue
        verify(so);out=HERE/'raw'/('B_trace_'+cell)/arm['label']
        run([PYTHON,'-m','tilemega.serving.trace','--model',arm['model_path'],'--prefill-so',arm['prefill'],
            '--decode-so',so,'--batch',arm['batch'],'--mode','L1','--decode-loop',0,'--task',
            '--past-list','575','--launches',16,'--out',out],out/'run.log',900)
        from ledger_r14 import tasks
        for folder in out.glob('past*'):write(folder/'task_summary.json',tasks(folder/'task_profile.tsv'))

def correctness(cell):
    rows=[];reference=None
    for arm in accepted(cell):
        out=HERE/'raw'/('B6_'+cell)/arm['label'];out.mkdir(parents=True,exist_ok=True)
        prompts=ROOT/f"docs/experiments/SERVING_R10/prompts/{arm['model']}_ids.json"
        run([PYTHON,'-m','tilemega.serving.check_modes','--model',arm['model_path'],'--prefill-so',arm['prefill'],
            '--decode-so',arm['decode'],'--prompt-ids',prompts,'--batch',arm['batch'],'--steps',1024,'--out',out],out/'modes.log',1200,allowed=(0,1))
        report=json.loads((out/'mode_check.json').read_text()) if (out/'mode_check.json').exists() else {'pass':False}
        tokens=out/'tokens_L1_separate.json';same=None
        if tokens.exists():
            values=json.loads(tokens.read_text())
            if arm['label']=='baseline':reference=values
            if arm['label'] in ('EP_arg','RA_noinline','RW_pipe'):same=values==reference
            run([PYTHON,'-m','tilemega.serving.hf_check','--model',arm['model_path'],'--prompt-ids',prompts,
                '--generated',tokens,'--vllm-metrics',HERE/f'raw/A3_{cell}/vllm_hf.json','--skip-free-greedy','--out',out/'hf.json'],out/'hf.log',1800,allowed=(0,1))
        hf=json.loads((out/'hf.json').read_text()) if (out/'hf.json').exists() else {'pass':False}
        rows.append(dict(label=arm['label'],c2=report,same_as_baseline=same,c1=hf.get('pass',False),hf=str(out/'hf.json')))
    write(HERE/f'raw/B6_{cell}/results.json',rows)
    if any(not r['c1'] or not r['c2']['pass'] or r['same_as_baseline'] is False for r in rows):raise ValueError('C-2 failed; preserve all evidence')

def protocol(cell,label):
    arm=next((a for a in accepted(cell) if a['label']==label),None)
    if not arm:raise ValueError('protocol candidate failed build or smoke')
    out=HERE/'raw'/('B6_protocol_'+cell);out.mkdir(parents=True,exist_ok=True)
    reference_label={'llama_B1':'baseline','qwen3_B1':'AT_la_ref','llama_B16':'SK_fill_ref'}[cell]
    reference=next(a for a in accepted(cell) if a['label']==reference_label)
    case=dict(model=arm['model_path'],batch=arm['batch'],steps=64,
        prompt_ids=str(ROOT/f"docs/experiments/SERVING_R10/prompts/{arm['model']}_ids.json"),
        prefill=arm['prefill'],decode=arm['decode'],reference_decode=reference['decode'],
        binary_sha256=dict(prefill=sha(arm['prefill']),decode=sha(arm['decode']),reference_decode=sha(reference['decode'])),
        arms=[dict(label='L1_reference',prefill=arm['prefill'],decode=reference['decode'],modes=['L1'],decode_loop=False),
              dict(label='L1_repeat',prefill=arm['prefill'],decode=arm['decode'],modes=['L1','L1'],decode_loop=False)])
    write(out/'cases.json',[case])
    run([PYTHON,ROOT/'docs/experiments/SERVING_R11/check_protocol.py','--cases',out/'cases.json',
         '--out',out/'processes','--processes',50,'--resume'],out/'run.log',14000)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','numerical','smoke','trace','correctness','protocol'));p.add_argument('--cell');p.add_argument('--label');a=p.parse_args()
    if a.action in ('prepare','numerical'):globals()[a.action]()
    elif a.action=='protocol':protocol(a.cell,a.label)
    else:globals()[a.action](a.cell)
if __name__=='__main__':main()
