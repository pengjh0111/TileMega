#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Prepare fixed-shape H7a model pairs with guarded, budgeted real occupancy queries."""
import argparse
import collections
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import run_r16_dnn_candidates as base
from r16_measurement_validation import native_identity, completed_guard


def audit_sass(lines):
    functions=[];name=None;instructions=[];counts=collections.Counter()
    def flush():
        if name is None:return
        mma=[a for a,s in instructions if 'HGMMA' in s]
        if not mma:return
        backedges=[]
        for a,s in instructions:
            match=re.search(r'\bBRA(?:\.\w+)*\s+(0x[0-9a-fA-F]+)',s)
            if match and int(match[1],16)<a:backedges.append((int(match[1],16),a))
        regions=set()
        for a in mma:
            candidates=[(end-begin,begin,end) for begin,end in backedges if begin<=a<=end]
            regions.add(min(candidates)[1:] if candidates else (min(mma),max(mma)))
        rows=[]
        for begin,end in sorted(regions):
            local=[dict(address=hex(a),instruction=s) for a,s in instructions if begin<=a<=end and re.search(r'\b(?:LDL|STL)(?:\.|\s)',s)]
            rows.append(dict(begin=hex(begin),end=hex(end),local_instructions=local))
        functions.append(dict(function=name,hgmma=len(mma),regions=rows,whole_function_local_instructions=sum(bool(re.search(r'\b(?:LDL|STL)(?:\.|\s)',s)) for a,s in instructions)))
    for line in lines:
        match=re.search(r'Function : (\S+)',line)
        if match:flush();name=match[1];instructions=[];continue
        match=re.search(r'/\*([0-9a-fA-F]+)\*/\s+([^;]+);',line)
        if match:
            instruction=match[2].strip();instructions.append((int(match[1],16),instruction))
            for op in ('HGMMA','UTMALDG','UBLKCP','UTMALDG.4D'):
                counts[op]+=int(op in instruction)
    flush()
    return dict(scope='conservative smallest direct-BRA backedge intervals containing HGMMA; straight-line issue interval otherwise; not a full CFG proof; outer-loop cold traffic may conservatively reject',counts=dict(counts),functions=functions,hot_local_instructions=sum(len(region['local_instructions']) for f in functions for region in f['regions']))


def verified(path):
    c=base.verified_config(path)
    for key in ('recovery_driver','baseline_driver','validation_helper','class_helper'):
        if base.sha(c[key])!=c[key+'_sha256']:raise ValueError('frozen recovery dependency changed: '+key)
    return c


def build(path,attempt):
    c=verified(path);base.require_lease(c);out=Path(attempt).absolute();out.mkdir(parents=True,exist_ok=True)
    command=[c['cli'],'compile',c['bridge'],str(Path(c['directory'])/'plan.so')]+c['flags'];start=time.time();events=[]
    with (out/'build.log').open('w') as log,(out/'cpu-solver-stage-events.jsonl').open('w') as stamps:
        p=subprocess.Popen(command,cwd=c['run_root'],env=os.environ,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        stamps.write(json.dumps(dict(event='official_compiler_started',time=start,pid=p.pid,scope='CPU solver and NVCC then real guarded occupancy query; no assumed phase duration'))+'\n');stamps.flush()
        for line in p.stdout:
            log.write(line);log.flush()
            stamps.write(json.dumps(dict(event='compiler_output',time=time.time(),line=line.rstrip()))+'\n');stamps.flush()
        code=p.wait();stamps.write(json.dumps(dict(event='official_compiler_ended',time=time.time(),code=code))+'\n')
    receipt=dict(command=command,started=start,ended=time.time(),child_pid=p.pid,exit_code=code,source_head=c['source_head'],cli_sha256=c['cli_sha256'],resource_probe_min_free_mib=2048,resource_query='real guarded CUDA occupancy; serialized low-budget query',compiler_log=str(out/'build.log'),cpu_phase_events=str(out/'cpu-solver-stage-events.jsonl'),guard_timeout_seconds=c['build_timeout_s'])
    base.write(Path(c['directory'])/'guarded-build-result.json',receipt)
    if code:raise ValueError('official guarded single-case build failed')


def accept(path):
    c=verified(path);d=Path(c['directory']);receipt=base.read(d/'guarded-build-result.json')
    state=base.read(Path(c['run_root'])/f"runs/r16/scheduler/gpu{c['gpu']}/state.json");event=state[c['build_job']]
    folder=Path(event['out']);folder=folder if folder.is_absolute() else Path(c['run_root'])/folder
    completed_guard(folder,event)
    if receipt['exit_code']:raise ValueError('guarded compiler did not succeed')
    identity=native_identity(d/'plan.so');manifest=identity['plan']
    if identity['source']['head']!=c['source_head'] or identity['source']['compiler_sha256']!=c['cli_sha256']:raise ValueError('artifact is not bound to source15 compiler')
    if int(identity['defines'].get('TILEMEGA_ARCH_ID',0))!=900:raise ValueError('actual TaskBody architecture is not native Hopper')
    for key,want in c['expected'].items():
        if manifest.get(key)!=want:raise ValueError('requested coordinate silently differs: '+key)
    log=(d/'plan.so.ptxas.log').read_text(errors='replace')
    if 'C7510' in log or re.search(r'wgmma[^\n]*serializ',log,re.I):raise ValueError('WGMMA serialization warning')
    proc=subprocess.Popen(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(d/'plan.so')],stdout=subprocess.PIPE,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
    def lines():
        with gzip.open(d/'sass.txt.gz','wt',compresslevel=1) as saved:
            for line in proc.stdout:saved.write(line);yield line
    audit=audit_sass(lines());base.write(d/'sass-audit.json',audit)
    if proc.wait():raise ValueError('SASS disassembly failed')
    actual_wgmma=sum(g.get('impl')=='wgmma' for g in manifest['gemms'])
    if c['arm']=='tma-im2col':
        if not actual_wgmma or not all(audit['counts'].get(op,0)>0 for op in ['HGMMA','UTMALDG.4D']):raise ValueError('actual GEMM dispatch or 4D im2col instruction missing')
        if audit['hot_local_instructions']:raise ValueError('conservative WGMMA compute-loop local-memory gate failed')
    elif actual_wgmma or audit['counts'].get('HGMMA',0):raise ValueError('MMA control contains WGMMA')
    result=dict(passed=True,candidate_id=c['candidate_id'],artifact_id=identity['artifact_id'],library_sha256=identity['binary_sha256'],source_head=c['source_head'],cli_sha256=c['cli_sha256'],sass_counts=audit['counts'],actual_wgmma_gemms=actual_wgmma,gemms=manifest['gemms'],resource_query='guarded real occupancy, explicit 2048MiB disk reserve',resources_sha256=identity['resources_sha256'],source_checkpoint_sha256=base.read(Path(c['export'])/'manifest.json')['checkpoint']['sha256'],hot_local_gate=audit['hot_local_instructions'],whole_kernel_spill_claim='not asserted; full ptxas and conservative loop audit retained')
    base.write(d/'cpu-build-result.json',result)


def prepare(args):
    r=args.root.absolute();receipt=base.read(args.receipt);out=args.out.absolute()
    if out.exists():raise FileExistsError('new recovery run-id required')
    if not receipt.get('build_completed') or base.sha(receipt['cli'])!=receipt['cli_sha256']:raise ValueError('completed immutable source15 CLI required')
    out.mkdir(parents=True);driver=out/'run_r16_h7a_recovery.py';shutil.copy2(__file__,driver)
    baseline=out/'run_r16_dnn_candidates.py';shutil.copy2(Path(base.__file__),baseline)
    validation=out/'r16_measurement_validation.py';shutil.copy2(Path(__file__).with_name('r16_measurement_validation.py'),validation)
    target=out/'target.json';shutil.copy2(args.target,target);helper=args.class_helper.absolute();jobs=[];configs=[];resnet_names=[]
    for model in ['resnet18','nafnet']:
        export=r/('runs/r16/dnn/nafnet-nonzero-scales-v1/export' if model=='nafnet' else 'runs/r16/dnn/export/resnet18');parent=out/model;parent.mkdir();bridge=parent/'export.json';shutil.copy2(export/'bridge_b1.json',bridge)
        cases=parent/'single-point-cases.json'
        with (parent/'classes.log').open('w') as log:code=subprocess.run([str(helper),str(bridge),str(target),str(cases)],env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT).returncode
        if code:raise ValueError('source15 semantic class preparation failed: '+model)
        domain=parent/'domain.json';base.write(domain,{'geometries':[dict(tile_m=64,tile_n=128,tile_k=64,stages=2)]})
        for arm in ['mma','tma-im2col']:
            directory=parent/arm;directory.mkdir();expected=dict(conv_impl='cpasync' if arm=='mma' else 'tma_im2col',forward_gemm_impl='mma16' if arm=='mma' else 'wgmma')
            flags=['--emit','serving','--serving','forward','--frontend','dnn','--batch','1','--runtime-target',str(target),'--pg','l2','--resource-probes','1','--resource-probe-min-free-mib','2048','--nonpaged-weight-layout','tiled','--deferred-ln','auto','--reuse','l2','--global-la','auto','--dwpw-fuse','auto','--forward-gemm-impl',expected['forward_gemm_impl'],'--conv-impl',expected['conv_impl'],'--solve',str(target),'--selection','predicted','--evaluate-configs',str(cases),'--search-domain',str(domain),'--search-passes','1','--search-jobs','1','--variant-cache',str(r/f'runs/r16/dnn/h7a-candidates-source10-v1/{model}/h7a-tma-im2col/seed.cu.resources/variant_cache')]
            c=dict(candidate_id=out.name+':'+model+':'+arm+':B1',arm=arm,source=receipt['source_dir'],source_head=receipt['snapshot_head'],run_root=str(r),cli=receipt['cli'],cli_sha256=receipt['cli_sha256'],directory=str(directory),export=str(export),bridge=str(bridge),python=str(args.python.absolute()),gpu=1,target=str(target),target_sha256=base.sha(target),single_point_case_file=str(cases),single_point_case_sha256=base.sha(cases),flags=flags,expected=expected,export_files={f:base.sha(export/f) for f in ['bridge_b1.json','manifest.json','inputs_b1.safetensors','exported_program.pt2']},build_timeout_s=1800 if model=='resnet18' else 7200,checkpoint_scope='shared full synthetic exported model; not pretrained dataset C-1',search_scope='one predeclared per-class M64N128K64S2split1 point; explicit 2048MiB disk reserve; no performance/numerical threshold change')
            for key,value in [('recovery_driver',driver),('baseline_driver',baseline),('validation_helper',validation),('class_helper',helper)]:c[key]=str(value);c[key+'_sha256']=base.sha(value)
            stem='h7a-'+out.name+'-'+model+'-'+arm;c['build_job']=stem+'-build'
            path=directory/'candidate-gpu1.json';base.write(path,c);configs.append(str(path));prefix=[c['python'],str(driver)];env=dict(PYTHONPATH=c['source']+'/python',CUDA_VISIBLE_DEVICES='1',TILEMEGA_DEVICE_INDEX='1',TILEMEGA_NUMA_NODE='0',TILEMEGA_GPU_PID_MODE='namespace_limited',OMP_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');common=dict(cwd=str(r),env=env,priority=2 if model=='resnet18' else 9,needs_free_mib=4096)
            build_name=stem+'-build';jobs.append(dict(common,name=build_name,gpu=True,timing=False,timeout_s=c['build_timeout_s'],after_any=list(resnet_names) if model=='nafnet' else [],command=['numactl','--cpunodebind=0']+prefix+['build','--candidate',str(path),'--attempt','{attempt_dir}']))
            jobs.append(dict(common,name=stem+'-accept',gpu=False,timing=False,timeout_s=300,after=[build_name],command=prefix+['accept','--candidate',str(path)]))
            checks=[]
            for fresh in range(5):
                name=stem+f'-check-f{fresh}';checks.append(name);jobs.append(dict(common,name=name,gpu=True,timing=False,timeout_s=900,after=[stem+'-accept'],command=['numactl','--cpunodebind=0']+prefix+['check','--candidate',str(path),'--fresh',str(fresh),'--attempt','{attempt_dir}']))
            jobs.append(dict(common,name=stem+'-gate',gpu=False,timing=False,timeout_s=120,after=checks,command=prefix+['aggregate','--candidate',str(path)]))
            for rnd in range(3):jobs.append(dict(common,name=stem+f'-measure-r{rnd}',gpu=True,timing=True,timeout_s=1800,after=[stem+'-gate'],command=['numactl','--cpunodebind=0']+prefix+['measure','--candidate',str(path),'--round-index',str(rnd),'--attempt','{attempt_dir}']))
        if model=='resnet18':resnet_names=[j['name'] for j in jobs]
    base.write(out/'queue-preview.json',jobs);base.write(out/'preparation.json',dict(source_receipt=str(args.receipt.absolute()),gpu=1,configurations=configs,jobs=len(jobs),cpu_bind_only_deviation='container membind unavailable',resnet_first=True,naf_independent_timeout_s=7200,thresholds_unchanged=True,enqueued=False))
    print(json.dumps(dict(queue=str(out/'queue-preview.json'),jobs=len(jobs))))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='operation',required=True)
    a=sub.add_parser('prepare');a.add_argument('--root',type=Path,default=Path('/root/TileMega'));a.add_argument('--receipt',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--target',type=Path,required=True);a.add_argument('--class-helper',type=Path,required=True);a.add_argument('--python',type=Path,default=Path('/root/TileMega/runs/r16/env/dnn-export/bin/python'))
    for operation in ['build','accept','check','aggregate','measure']:
        a=sub.add_parser(operation);a.add_argument('--candidate',type=Path,required=True)
        if operation in ['build','check','measure']:a.add_argument('--attempt',type=Path,required=True)
        if operation=='check':a.add_argument('--fresh',type=int,required=True)
        if operation=='measure':a.add_argument('--round-index',type=int,required=True)
    args=p.parse_args()
    if args.operation=='prepare':prepare(args)
    elif args.operation=='build':build(args.candidate,args.attempt)
    elif args.operation=='accept':accept(args.candidate)
    else:
        verified(args.candidate)
        if args.operation=='check':base.check(args.candidate,args.fresh,args.attempt)
        elif args.operation=='aggregate':base.aggregate(args.candidate)
        else:base.measure(args.candidate,args.round_index,args.attempt)

if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError) as e:print('H7a recovery rejected: '+str(e),file=sys.stderr);raise SystemExit(3)
