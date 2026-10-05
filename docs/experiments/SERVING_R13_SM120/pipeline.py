#!/usr/bin/env python3
"""Declare E2c–E6 ahead of measurement; arm-local evidence gates replace manual pauses."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from bootstrap import HERE, ROOT, FRAME, WORK, PY, write, sha, run

CELLS=('llama_B1','qwen3_B1','llama_B16','qwen3_B16')
MATRICES={
    'E4a':('vllm','B0','R13F'),
    'E4b':('B0','NL2e','NL2g','NL2r'),
    'E4c':('B0-noev','B0p','B0p1','NL2g-noev','NL2gp1','PR_L2-noev','PRp_L2','B0l','PR_L2l'),
    'E4d':('B0h','PS_L1','PS_L2','PS_L2l','PSA_L2','PSA_L2l','PR_L1','PR_L2','PR_L2l'),
}
GROUPS={
    'bulk':('PR_L2','PR_L2l'),
    'pdl':('B0-noev','B0p','B0p1','NL2gp1'),
    'l1loop':('B0-noev','B0l'),
    'seed_fx21':('PS_L2','PS_L2l','PSA_L2','PSA_L2l'),
}


def read(path,default=None):
    path=Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def arm_recipe(label):
    key='N-R12b-120'
    if label=='B0h':key='N-R12bh-120'
    elif label=='B0p':key='N-R12b-120-pdl0'
    elif label in ('B0p1','NL2gp1'):key='N-R12b-120-pdl1'
    elif label=='PRp_L2':key='P-R12bN-120-pdl'
    elif label.startswith('PR_'):key='P-R12bN-120'
    elif label.startswith('PSA_'):key='PSA-120'
    elif label.startswith('PS_'):key='PS-120'
    mode='L2' if label.startswith('NL2') or '_L2' in label else 'L1'
    loop=label=='B0l' or label.endswith('_L2l')
    env={}
    if label in ('NL2g','NL2g-noev','NL2gp1'):env['TILEMEGA_PLACEMENT_ABLATION']='grid_stride'
    elif label=='NL2r':env['TILEMEGA_PLACEMENT_ABLATION']='rotate'
    return key,mode,loop,env


def frozen_arm(label,cell,rows,smoke):
    model,batch=cell.split('_B')
    batch=int(batch)
    config=read(HERE/f'{model}_r13_sm120_r3.json')
    row=dict(label=label+'-120',base_label=label,model=model,batch=batch,root=str(ROOT),python=PY,
             model_path=config['model']['path'],env={},available=False,binaries={},prefill_mode='L1')
    if label=='vllm':
        import importlib.metadata
        cuda=Path('/root/toolchains/cuda-13.0-env/lib/python3.12/site-packages/nvidia/cu13')
        installed=importlib.metadata.version('vllm')
        row.update(kind='vllm',available=installed=='0.29.0',installed_version=installed,env=dict(CUDA_HOME=str(cuda),
                    PATH=str(cuda/'bin')+':'+os.environ['PATH'],CUDACXX=str(cuda/'bin/nvcc')))
        if not row['available']:row['unavailable_reason']='installed vLLM changed after registration; no automatic version switch'
        return row
    row['kind']='tm'
    if label=='R13F':
        plans=read(HERE/f'raw/E3_R13F_{model}_r4/run/plans.json',{})
        pair=plans.get(str(batch))
        if not pair:
            row['unavailable_reason']='R13F plan was not produced'
            return row
        row.update(prefill=pair['prefill'],decode=pair['decode'],mode='auto',decode_loop='auto',prefill_mode='auto')
        from tilemega.serving.execution import read_execution
        execution=read_execution(row['decode'])
        if execution is None:
            row['unavailable_reason']='joint selector did not provide an execution sidecar; no silent auto fallback'
            return row
        row['selected_execution']=execution
        key='R13F-120'
    else:
        key,mode,loop,env=arm_recipe(label)
        decode=rows.get((cell,key))
        prefill=rows.get((cell,'PF-120'))
        if not decode or not prefill or decode['exit_code'] or prefill['exit_code']:
            row['unavailable_reason']='fixed geometry or selected prefill rejected/missing'
            return row
        row.update(prefill=prefill['so'],decode=decode['so'],mode=mode,decode_loop=loop,env=env)
    checked=smoke.get((cell,key))
    if not checked or not checked.get('report'):
        row['unavailable_reason']='decode smoke did not complete safely'
        return row
    report=checked['report']
    selected_mode=row['mode'] if row['mode']!='auto' else row['selected_execution']['decode_mode']
    selected_loop=row['decode_loop'] if row['decode_loop']!='auto' else row['selected_execution']['decode_loop']
    name=selected_mode+('_loop' if selected_loop else '_separate')
    if name not in report.get('arms',{}):
        row['unavailable_reason']='requested execution mode/loop was not exercised'
        return row
    row['smoke_correctness_pass']=report.get('pass',False)
    for phase in ('prefill','decode'):
        if not Path(row[phase]).is_file():
            row['unavailable_reason']='runtime binary disappeared'
            return row
        metadata={suffix:sha(row[phase]+suffix) for suffix in ('.plan.json','.serving.json') if Path(row[phase]+suffix).exists()}
        row['binaries'][phase]=dict(path=row[phase],sha256=sha(row[phase]),metadata_sha256=metadata)
    row['available']=True
    row['step_events']=0 if label in MATRICES['E4c'] else 1
    return row


def catalog():
    builds=read(HERE/'builds_sm120_r3.json',[])
    rows={(row['cell'],row['label']):row for row in builds}
    smoke={(row['cell'],row['label']):row for row in read(HERE/'raw/E2b_smoke_r4/results.json',[])}
    labels=sorted({label for group in MATRICES.values() for label in group})
    arms={cell:[frozen_arm(label,cell,rows,smoke) for label in labels] for cell in CELLS}
    write(HERE/'catalog.json',arms)
    records=[]
    sys.path.insert(0,str(FRAME))
    from bootstrap import BUILD
    for cell in CELLS:
        for phase,key in (('decode','N-R12b-120'),('prefill','PF-120')):
            row=rows.get((cell,key))
            if not row or row['exit_code']:
                records.append(dict(cell=cell,phase=phase,status='unavailable'))
                continue
            graph=Path(row['out'])/'selected.mlir'
            folder=HERE/'raw/E3_catalog_r5/floors'/cell/phase
            lo,hi=(64,1086) if phase=='decode' else (0,0)
            command=[BUILD/'tools/tilemega','inspect','request-floor',graph,HERE/'target_sm120.json',
                     cell.split('_B')[1],str(lo),str(hi),folder/'floor.json',folder/'steps.tsv']
            code=run(command,folder/'stdout.log')
            records.append(dict(cell=cell,phase=phase,exit_code=code,path=str(folder/'floor.json')))
    write(HERE/'raw/E3_catalog_r5/floors.json',records)
    # An unavailable plan is data, not permission to replace its geometry.
    return 0


def original_arm(cell,label):
    return next(row for row in read(HERE/'catalog.json',{}).get(cell,[]) if row['base_label']==label)


def protocol(group):
    folder=HERE/f'raw/E2c_{group}_r5'
    requested=[original_arm('llama_B16',label) for label in GROUPS[group]]
    missing=[arm['base_label'] for arm in requested if not arm['available']]
    selected=[arm for arm in requested if arm['available']]
    if not requested[0]['available'] or len(selected)<2:
        write(folder/'status.json',dict(status='unavailable',group=group,complete=False,
              missing=missing,
              requested=50,passed=None))
        return 0
    arms=[]
    for arm in selected:
        item={key:arm[key] for key in ('prefill','decode','mode','decode_loop','env')}
        item.update(label=arm['base_label'],sha256={phase:arm['binaries'][phase]['sha256'] for phase in ('prefill','decode')})
        arms.append(item)
    case=dict(group=group,model=selected[0]['model_path'],batch=16,steps=64,
              prompt_ids=str(ROOT/'docs/experiments/SERVING_R10/prompts/llama_ids.json'),arms=arms)
    write(folder/'case.json',case)
    code=run([PY,HERE/'protocol.py','--case',folder/'case.json','--out',folder/'processes'],folder/'protocol.log')
    if code==75:return 75
    summary=read(folder/'processes/summary.json',{})
    write(folder/'status.json',dict(group=group,status='collected' if summary.get('complete') else 'incomplete',
                                   exit_code=code,covered_labels=[arm['base_label'] for arm in selected],
                                   missing_labels=missing,coverage_complete=not missing,**summary))
    return code if not summary.get('complete') else 0


def protocol_group(label):
    if label in ('B0p','B0p1','NL2gp1'):return 'pdl'
    if label=='PR_L2l':return 'bulk'
    if label=='B0l':return 'l1loop'
    if label.endswith('_L2l') and label.startswith(('PS_','PSA_')):return 'seed_fx21'
    return None


def runnable(cell,label):
    row=copy.deepcopy(original_arm(cell,label))
    group=protocol_group(label)
    if row['available'] and group:
        status=read(HERE/f'raw/E2c_{group}_r5/status.json',{})
        if not status.get('complete') or label not in status.get('covered_labels',[]):
            row.update(available=False,unavailable_reason='required fresh-process execution is incomplete: '+group)
        else:
            row['protocol_pass_rate']=status['pass_rate']
            row['protocol_correctness_pass']=status['passed']==50
    for phase in ('prefill','decode') if row['available'] and row['kind']=='tm' else ():
        if sha(row[phase])!=row['binaries'][phase]['sha256']:
            raise RuntimeError('frozen binary changed: '+row[phase])
        for suffix,expected in row['binaries'][phase].get('metadata_sha256',{}).items():
            if sha(row[phase]+suffix)!=expected:raise RuntimeError('frozen execution metadata changed: '+row[phase]+suffix)
    return row


def anchor(matrix,cell,round_,replacement=False):
    folder=HERE/f"raw/{matrix}_{cell}_r{round_}_{'canary_' if replacement else ''}r5"
    arms={cell:[runnable(cell,label) for label in MATRICES[matrix]]}
    for arm in arms[cell]:
        if arm['kind']=='tm':arm['step_events']=0 if matrix=='E4c' else 1
    write(folder/'arms.json',arms)
    write(folder/'eligibility.json',arms)
    code=run([PY,FRAME/'anchor.py','--arms',folder/'arms.json','--cell',cell,'--round',str(round_),
              '--out',folder],folder/'anchor.log')
    return code


def round_path(matrix,cell,number):
    replacements=read(HERE/'raw/E4_canary_r5/replacements.json',{})
    key=f'{matrix}:{cell}:{number}'
    if key in replacements:return Path(replacements[key])
    replay=read(HERE/'raw/acceptance_03/anchor_replacements.json',{}).get(key)
    if replay:
        if sha(replay['original_path'])!=replay['original_sha256']:
            raise RuntimeError('original round changed after validation replay: '+replay['original_path'])
        return Path(replay['path'])
    return HERE/f'raw/{matrix}_{cell}_r{number}_r5/{cell}/round{number}.json'


def canaries():
    import statistics
    folder=HERE/'raw/E4_canary_r5'
    replacements=read(folder/'replacements.json',{})
    flags=[]
    bases={'E4a':'vllm','E4b':'B0','E4c':'B0-noev','E4d':'B0h'}
    for matrix in MATRICES:
        for cell in CELLS:
            labels=(bases[matrix],) if not (matrix=='E4d' and cell=='qwen3_B16') else ('PR_L1',)
            label=labels[0]+'-120'
            values={}
            for number in range(3):
                data=read(round_path(matrix,cell,number),{})
                row=data.get('arms',{}).get(label,{})
                if not data.get('invalidated') and row.get('exit_code')==0 and 'e2e_seconds' in row:
                    values[number]=(row['e2e_seconds']-row['ttft_seconds'])/1023
            if len(values)!=3:continue
            median=statistics.median(values.values())
            for number,value in values.items():
                if abs(value/median-1)<=.02:continue
                key=f'{matrix}:{cell}:{number}'
                if key in replacements:continue
                flags.append(dict(matrix=matrix,cell=cell,round=number,canary=label,relative=value/median-1))
                code=anchor(matrix,cell,number,replacement=True)
                if code==75:return 75
                candidate=HERE/f'raw/{matrix}_{cell}_r{number}_canary_r5/{cell}/round{number}.json'
                # Failed reruns are retained, never used to replace valid data.
                candidate_data=read(candidate,{})
                original_data=read(round_path(matrix,cell,number),{})
                required=[name for name,row in original_data.get('arms',{}).items() if row.get('exit_code')==0 and 'e2e_seconds' in row]
                complete=all(candidate_data.get('arms',{}).get(name,{}).get('exit_code')==0 and
                             'e2e_seconds' in candidate_data['arms'][name] for name in required)
                if not code and required and complete and not candidate_data.get('invalidated'):
                    replacements[key]=str(candidate)
                    write(folder/'replacements.json',replacements)
    write(folder/'flags.json',flags)
    return 0


def token_file(cell,label):
    for matrix in ('E4a','E4d','E4c','E4b'):
        for number in range(3):
            path=round_path(matrix,cell,number)
            data=read(path,{})
            if data.get('invalidated'):continue
            record=data.get('arms',{}).get(label+'-120',{})
            if record.get('exit_code')!=0:continue
            matches=sorted(Path(record['out']).rglob('tokens_N1024_run1.json'))
            if matches:return matches[0]
    return None


def correctness(cell):
    folder=HERE/f'raw/E4f_{cell}_r5'
    sys.path.insert(0,str(FRAME))
    from report import measurements
    _,tokens=measurements()
    records=[]
    for matrix in MATRICES:
        for number in range(3):
            # Expected bitwise groups do not compare arbitrary geometries.
            if matrix=='E4b':groups=[MATRICES[matrix]]
            elif matrix=='E4c':groups=[('B0-noev','B0p','B0p1','NL2g-noev','NL2gp1','B0l'),('PR_L2-noev','PRp_L2','PR_L2l')]
            elif matrix=='E4d':groups=[('PS_L1','PS_L2','PS_L2l','PSA_L2','PSA_L2l'),('PR_L1','PR_L2','PR_L2l')]
            else:groups=[]
            for group in groups:
                present=[(label,tokens.get((matrix,cell,label+'-120',number))) for label in group]
                present=[pair for pair in present if pair[1] is not None]
                if not present:continue
                reference=present[0][1]
                for label,value in present:
                    same_shape=[len(row) for row in value]==[len(row) for row in reference]
                    mismatch=sum(a!=b for left,right in zip(reference,value) for a,b in zip(left,right))
                    records.append(dict(check='C-2',matrix=matrix,round=number,label=label,
                          reference=present[0][0],shape_equal=same_shape,mismatches=mismatch,
                          pass_=same_shape and mismatch==0))
                for label in group:
                    if not any(item[0]==label for item in present):
                        records.append(dict(check='C-2',matrix=matrix,round=number,label=label,status='unavailable'))
    for label in ('vllm','B0h','R13F'):
        arm=runnable(cell,label)
        generated=token_file(cell,label) if arm['available'] else None
        if not generated:
            records.append(dict(check='C-1',label=label,status='unavailable/no measured tokens'))
            continue
        report=folder/(label+'_hf.json')
        command=[PY,'-m','tilemega.serving.hf_check','--model',arm['model_path'],'--prompt-ids',
                 ROOT/f"docs/experiments/SERVING_R10/prompts/{arm['model']}_ids.json",'--generated',generated,
                 '--skip-free-greedy','--out',report]
        baseline=folder/'vllm_hf.json'
        if label!='vllm' and baseline.exists():command+=['--vllm-metrics',baseline]
        code=run(command,folder/(label+'_hf.log'))
        if code==75:return 75
        records.append(dict(check='C-1',label=label,exit_code=code,report=read(report)))
    write(folder/'checks.json',records)
    return 0


def trace_requests(stage,cell):
    if stage=='E5a':return [('B0','N-R12b-120-trace','L1',False),('PR_L1','P-R12bN-120-trace','L1',False)]
    if stage=='E5b':return [('NL2e','N-R12b-120-v2','L2',False),('NL2g','N-R12b-120-v2','L2',False)]
    if stage=='E5c':return [('B0-noev','N-R12b-120-trace','L1',False),('B0p1','N-R12b-120-pdl1-step','L1',False),
                            ('B0l','N-R12b-120-trace','L1',True),('PR_L2','P-R12bN-120-trace','L2',False),
                            ('PR_L2l','P-R12bN-120-trace','L2',True)]
    return [('PR_L2','P-R12bN-120-pages','L2',False),('PS_L2','PS-120-pages','L2',False),('PSA_L2','PSA-120-pages','L2',False)]


def trace(stage,cell):
    builds={(row['cell'],row['label']):row for row in read(HERE/'builds_sm120_r3.json',[])}
    folder=HERE/f'raw/{stage}_{cell}_r5'
    results=[]
    for label,variant,mode,loop in trace_requests(stage,cell):
        arm=runnable(cell,label)
        build=builds.get((cell,variant))
        if not arm['available'] or not build or build['exit_code']:
            results.append(dict(label=label,variant=variant,status='unavailable'))
            continue
        destination=folder/label
        command=[PY,'-m','tilemega.serving.trace','--model',arm['model_path'],'--prefill-so',arm['prefill'],
                 '--decode-so',build['so'],'--batch',str(arm['batch']),'--past','575','--out',destination,
                 '--mode',mode,'--decode-loop',str(int(loop))]
        if stage=='E5a':command+=['--stage','--past-list','64,575,1000','--launches','16']
        elif stage=='E5c':command+=['--step','--steps','16','--launches','1']
        else:command+=['--launches','1']
        environment=dict(os.environ,**arm['env'])
        if stage=='E5d':environment['TILEMEGA_PAGE_TRACE_OUT']=str(destination/'page_trace.tsv')
        destination.mkdir(parents=True,exist_ok=True)
        write(destination/'command.json',dict(command=list(map(str,command)),env=arm['env'],
              trace_binary_sha256=sha(build['so']),origin_binary_sha256=arm['binaries']['decode']['sha256']))
        with (destination/'stdout.log').open('w') as log:
            code=subprocess.run(['timeout','--kill-after=15s','1800s',*map(str,command)],env=environment,
                                stdout=log,stderr=subprocess.STDOUT).returncode
        if code==75:return 75
        results.append(dict(label=label,variant=variant,exit_code=code,path=str(destination)))
        write(folder/'trace_results.json',results)
        from device_health import check
        check()
    write(folder/'trace_results.json',results)
    return 0


def audit():
    from bootstrap import BUILD
    records=[]
    for row in read(HERE/'builds_sm120_r3.json',[]):
        if row['exit_code'] or row['phase']!='decode' or row['label'].endswith(('-trace','-v2','-pages','-step')):continue
        folder=HERE/'raw/E3_codegen_r5'/row['cell']/row['label']
        code=run(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',row['so']],folder/'sass.txt')
        sass=(folder/'sass.txt').read_text() if not code else ''
        source=Path(row['so']+'.cu')
        records.append(dict(cell=row['cell'],label=row['label'],binary_sha256=sha(row['so']),
          source_sha256=sha(source) if source.exists() else None,exit_code=code,
          sass_mnemonics={name:sass.count(name) for name in ('LDGSTS','UTMA','GRIDDEPCONTROL','SYNCS','BAR.','HMMA','MMA')},
          qualification='compiled instructions only; not a hardware execution count'))
        embedded=folder/'embedded.ptx'
        ptx_code=run(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-ptx',row['so']],embedded)
        text=embedded.read_text() if not ptx_code else ''
        records[-1]['ptx_mnemonics']={name:text.count(name) for name in (
            'cp.async.bulk.shared::cluster','cp.async.bulk.tensor.2d','cp.async.bulk.prefetch',
            'griddepcontrol','barrier.cluster','mma.sync')}
        records[-1]['metadata']=read(row['so']+'.plan.json',{})
        if row['label'] in ('N-R12b-120-pdl0','N-R12b-120-pdl1'):
            sys.path.insert(0,str(FRAME))
            from arch_checks import command
            trigger=int(row['label'].endswith('pdl1'))
            ptx=folder/'native.ptx'
            code=run(command(row['so'],120,ptx,'ptx',[f'-DTILEMEGA_PDL_TRIGGER={trigger}']),folder/'ptx_build.log')
            if not code:
                records[-1]['pdl_position_exit']=run([PY,FRAME/'ptx_pdl_check.py',ptx,'--trigger',str(trigger),
                                                    '--out',folder/'positions.json'],folder/'position_check.log')
                records[-1]['pdl_position_report']=read(folder/'positions.json')
    write(HERE/'raw/E3_codegen_r5/audit.json',records)
    return 0


def make_queue():
    env=read(HERE/'launch_env_r3.json')
    nodes=[]
    def add(name,action,gpu=False,args=(),after=(),after_any=(),priority=40,timeout=43200):
        command=[PY,str(HERE/'pipeline.py'),action,*map(str,args)]
        if not gpu:command=['flock',str(WORK/'gpu.lock'),'env','TILEMEGA_GPU_LOCK_HELD=1',*command]
        nodes.append(dict(name=name,command=command,cwd=str(ROOT),env=env,gpu=gpu,
          after=list(after),after_any=list(after_any),priority=priority,timeout_s=timeout,
          needs_free_mib=28740 if name.startswith(('E4a','E4_canary')) else 12288,out=str(HERE/'raw'/name)))
    add('E3_catalog_r5','catalog',after_any=['E3_fixed_r4','E2b_smoke_r4','E3_R13F_llama_r4','E3_R13F_qwen3_r4'],priority=30)
    add('E3_codegen_r5','audit',after_any=['E3_fixed_r4'],priority=31)
    protocols=[]
    for group in GROUPS:
        name=f'E2c_{group}_r5';protocols.append(name)
        add(name,'protocol',gpu=True,args=['--group',group],after=['E3_catalog_r5'],priority=35)
    previous=protocols
    matrices=[]
    for index,matrix in enumerate(MATRICES):
        current=[]
        for cell in CELLS:
            for number in range(3):
                name=f'{matrix}_{cell}_r{number}_r5';current.append(name)
                add(name,'anchor',gpu=True,args=['--matrix',matrix,'--cell',cell,'--round',number],
                    after=['E3_catalog_r5'],after_any=previous,priority=40+index)
        previous=current
        matrices+=current
    checks=[]
    add('E4_canary_r5','canaries',gpu=True,after=['E3_catalog_r5'],after_any=matrices,priority=45)
    for cell in CELLS:
        name=f'E4f_{cell}_r5';checks.append(name)
        add(name,'correctness',gpu=True,args=['--cell',cell],after=['E3_catalog_r5'],after_any=['E4_canary_r5'],priority=46)
    traces=[]
    previous=checks
    for index,stage in enumerate(('E5a','E5b','E5c','E5d')):
        current=[]
        for cell in CELLS:
            if stage=='E5d' and not cell.endswith('B16'):continue
            name=f'{stage}_{cell}_r5';current.append(name)
            add(name,'trace',gpu=True,args=['--stage',stage,'--cell',cell],after=['E3_catalog_r5'],
                after_any=previous,priority=50+index)
        traces+=current
        previous=current
    add('E6_collect_r5','collect',after_any=matrices+protocols+checks+traces+['E3_catalog_r5','E3_codegen_r5','E4_canary_r5'],priority=60)
    return nodes


def register():
    nodes=make_queue()
    write(HERE/'queue_complete_r5.json',nodes)
    destination=WORK/'queue/queue_complete_r5.json'
    if destination.exists():raise RuntimeError('complete queue already published')
    temp=destination.with_suffix('.tmp')
    write(temp,nodes)
    temp.replace(destination)
    write(HERE/'implementation_status.json',dict(dependency_free_implementation_complete=True,
          pending_hardware_discovered_fixes=True,queued_nodes=len(nodes),
          groups={key:len(value) for key,value in MATRICES.items()},
          optional_omitted=['E4e cluster end-to-end','E5e ncu'],
          no_solver_or_cost_model_changes=True,no_push=True))
    print('published complete dependency queue:',len(nodes),'nodes',flush=True)
    return 0


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('catalog','protocol','anchor','correctness','trace','audit','collect','register','canaries'))
    parser.add_argument('--group',choices=GROUPS)
    parser.add_argument('--matrix',choices=MATRICES)
    parser.add_argument('--stage',choices=('E5a','E5b','E5c','E5d'))
    parser.add_argument('--cell',choices=CELLS)
    parser.add_argument('--round',type=int,default=0)
    args=parser.parse_args()
    if args.action=='collect':
        from report import collect
        return collect()
    return {'catalog':catalog,'protocol':lambda:protocol(args.group),'anchor':lambda:anchor(args.matrix,args.cell,args.round),
            'correctness':lambda:correctness(args.cell),'trace':lambda:trace(args.stage,args.cell),
            'audit':audit,'register':register,'canaries':canaries}[args.action]()


if __name__=='__main__':
    raise SystemExit(main())
