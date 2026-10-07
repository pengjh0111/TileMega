#!/usr/bin/env python3
"""Bounded conditional builds, numerical gates and registered retention."""
import argparse,json,os,re,shlex,subprocess
from pathlib import Path
from make_phase0 import HERE,ROOT,PYTHON,write
from phase_a import run
from tilemega.build.identity import sha,verify

def arms(cell):return json.loads((HERE/'phase_c_arms.json').read_text())[cell]
def accepted(cell):return json.loads((HERE/f'C_{cell}_accepted.json').read_text())
def prepare():
    out=HERE/'raw/Cpre';out.mkdir(parents=True,exist_ok=True)
    run(['cmake','--build','build-phase12','--target','tilemega','tilemega-unit','-j','6'],out/'build.log',6000)
    run(['ctest','--test-dir','build-phase12','-R','serving_pruning|serving_attention_legality|variant_resource|handoff_|stage_flow','--output-on-failure'],out/'host.log',1200)
    for name in ('integrated_selection','test_attention_selection','serving_identity'):
        run([PYTHON,f'test/python/{name}.py'],out/(name+'.log'))
    run([PYTHON,'python/tilemega/fingerprint.py','--check','build-phase12/tools/tilemega'],out/'fingerprint.log')
    bins=[]
    for arch in (80,89,90,100,120):
        for name,flags in [('attention_layout',['-DTILEMEGA_ATTENTION_BUFFERS=1']),('serving_fragment_epilogue',['-DTILEMEGA_EP_DIRECT=1'])]:
            dst=out/f'{name}_sm{arch}';cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-O3','--expt-relaxed-constexpr',f'-arch=sm_{arch}','-Iinclude','-Ithird_party/cutlass/include','-Xptxas=-v']+flags+[f'test/unit/{name}_test.cu']
            if arch!=89:cmd+=['-c'];dst=dst.with_suffix('.o')
            run(cmd+['-o',dst],out/f'{name}_sm{arch}.log',1200)
            if arch==89:bins.append(dict(path=str(dst),sha256=sha(dst)))
    write(out/'binaries.json',bins)
def numerical():
    for b in json.loads((HERE/'raw/Cpre/binaries.json').read_text()):
        if sha(b['path'])!=b['sha256']:raise ValueError('changed numerical binary')
        run([b['path']],HERE/'raw/C_numeric'/(Path(b['path']).name+'.log'),600)

def smoke(cell):
    passed=[];results=[]
    for arm in arms(cell):
        out=HERE/'raw'/('C_smoke_'+cell)/arm['label'];out.mkdir(parents=True,exist_ok=True)
        record=Path(arm['decode']).parent/'record.json'
        if not record.exists() or json.loads(record.read_text())['exit_code']:
            results.append(dict(label=arm['label'],status='build_failed'));continue
        identity=verify(arm['decode'])
        run([PYTHON,'-m','tilemega.serving.smoke','--so',arm['decode'],'--model',arm['model_path'],'--batch',arm['batch'],'--steps',64,'--out',out],out/'run.log',300,allowed=(0,1,3))
        result=json.loads((out/'smoke.json').read_text());ok=result['pass']
        if arm['label']=='C_RW1':
            plan=identity['plan'];ok=ok and plan['residency']==2 and identity['shared_memory_bytes']<=50176
        results.append(dict(label=arm['label'],status='passed' if ok else 'smoke_or_resource_failed',artifact_id=identity['artifact_id'],smem=identity['shared_memory_bytes']))
        if ok:passed.append(arm)
    write(HERE/f'raw/C_smoke_{cell}/results.json',results)
    if not any(a['label']=='baseline' for a in passed):raise ValueError('C baseline failed')
    write(HERE/f'C_{cell}_accepted.json',passed);write(HERE/f'C_{cell}_arms.json',{cell:passed})

def architecture():
    cell='qwen3_B1';arm=next((a for a in accepted(cell) if a['label']=='C_AT4'),None)
    if arm is None:
        write(HERE/'raw/C_arch/result.json',dict(status='excluded',reason='frontier build or smoke failed'));return
    base=Path(arm['decode']);command=shlex.split(Path(str(base)+'.build_command.txt').read_text());cu=str(base)+'.cu'
    out=HERE/'raw/C_arch_v2';out.mkdir(parents=True,exist_ok=True)
    # Compile the actual frontier executor, not merely an isolated TaskBody.
    for arch in (80,90,100,120):
        opts=[a for a in command if a.startswith(('-D','-I'))]
        # A production CU pins its solved architecture. Change only that
        # declaration for the compile-only portability specimen.
        source=Path(cu).read_text()
        source=re.sub(r'(TILEMEGA_ARCH_ID(?: != | ))890',lambda m:m[1]+str(arch*10),source)
        source=source.replace('#define TILEMEGA_ARCH_TAG "sm_89"',f'#define TILEMEGA_ARCH_TAG "sm_{arch}"')
        specimen=out/f'frontier_sm{arch}.cu';specimen.write_text(source)
        run([command[0],'-std=c++17','-O3','--expt-relaxed-constexpr',f'-arch=sm_{arch}']+opts+['-c',specimen,'-o',out/f'frontier_sm{arch}.o'],out/f'sm{arch}.log',1800)
    write(out/'result.json',dict(pass_=True,source_sha256=sha(cu),architectures=[80,89,90,100,120],sm89='original production build; executed',other_architectures='compiled only'))

def correctness(cell):
    rows=[]
    for arm in accepted(cell):
        out=HERE/'raw'/('C_correct_'+cell)/arm['label'];out.mkdir(parents=True,exist_ok=True)
        prompts=ROOT/f"docs/experiments/SERVING_R10/prompts/{arm['model']}_ids.json"
        run([PYTHON,'-m','tilemega.serving.check_modes','--model',arm['model_path'],'--prefill-so',arm['prefill'],'--decode-so',arm['decode'],'--prompt-ids',prompts,'--batch',arm['batch'],'--steps',1024,'--out',out],out/'modes.log',1200,allowed=(0,1))
        c2=json.loads((out/'mode_check.json').read_text()) if (out/'mode_check.json').exists() else {'pass':False}
        tokens=out/'tokens_L1_separate.json'
        if tokens.exists():
            run([PYTHON,'-m','tilemega.serving.hf_check','--model',arm['model_path'],'--prompt-ids',prompts,'--generated',tokens,'--vllm-metrics',HERE/f'raw/A3_{cell}/vllm_hf.json','--skip-free-greedy','--out',out/'hf.json'],out/'hf.log',1800,allowed=(0,1))
        c1=json.loads((out/'hf.json').read_text()).get('pass',False) if (out/'hf.json').exists() else False
        same=None
        if arm['label'] in ('C_EP2','C_AT4') and tokens.exists():same=json.loads(tokens.read_text())==json.loads((out.parent/'baseline/tokens_L1_separate.json').read_text())
        rows.append(dict(label=arm['label'],c1=c1,c2=c2['pass'],same_as_baseline=same))
    write(HERE/f'raw/C_correct_{cell}/results.json',rows)
    if any(not r['c1'] or not r['c2'] or r['same_as_baseline'] is False for r in rows):raise ValueError('C correctness failed')

def protocol(cell,label):
    arm=next((a for a in accepted(cell) if a['label']==label),None)
    if arm is None:
        write(HERE/f'raw/C_protocol_{cell}/result.json',dict(status='excluded',reason='candidate build or smoke failed'));return
    reference=next(a for a in accepted(cell) if a['label']=='baseline') if label=='C_AT4' else arm
    out=HERE/'raw'/('C_protocol_'+cell)
    case=dict(model=arm['model_path'],batch=arm['batch'],steps=64,prompt_ids=str(ROOT/f"docs/experiments/SERVING_R10/prompts/{arm['model']}_ids.json"),prefill=arm['prefill'],decode=arm['decode'],reference_decode=reference['decode'],binary_sha256=dict(prefill=sha(arm['prefill']),decode=sha(arm['decode']),reference_decode=sha(reference['decode'])),arms=[dict(label='reference',prefill=arm['prefill'],decode=reference['decode'],modes=['L1'],decode_loop=False),dict(label='candidate',prefill=arm['prefill'],decode=arm['decode'],modes=['L1','L1'],decode_loop=False)])
    write(out/'cases.json',[case])
    run([PYTHON,ROOT/'docs/experiments/SERVING_R11/check_protocol.py','--cases',out/'cases.json','--out',out/'processes','--processes',50,'--resume'],out/'run.log',12000)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','numerical','smoke','architecture','correctness','protocol'));p.add_argument('--cell');p.add_argument('--label');a=p.parse_args()
    if a.action in ('prepare','numerical','architecture'):globals()[a.action]()
    elif a.action=='protocol':protocol(a.cell,a.label)
    else:globals()[a.action](a.cell)
if __name__=='__main__':main()
