#!/usr/bin/env python3
"""S1–S10 from frozen evidence, including the final sm89 measurement limitations."""
from collections import defaultdict
import csv
import json
import statistics
import sys
from pathlib import Path

from bootstrap import HERE, FRAME, write
from pipeline import CELLS, MATRICES, GROUPS, read, round_path

PAIRS=(
 ('E4b','NL2e','B0','B2','NL2e','B0'),('E4b','NL2g','B0','B2','NL2g','B0'),
 ('E4d','PR_L1','B0h','B4','PR_L1','B0h'),('E4d','PS_L1','PR_L1','B4','PS_L1','PR_L1'),
 ('E4d','PSA_L2','PS_L2','B4','PSA_L2','PS_L2'),('E4d','PSA_L2l','PS_L2l','B4','PSA_L2l','PS_L2l'),
 ('E4c','B0l','B0-noev','B3','B0l','B0-noev'),('E4c','PR_L2l','PR_L2-noev','B3','PR_L2l','PR_L2'),
 ('E4c','B0p','B0-noev',None,None,None),('E4c','B0p1','B0-noev',None,None,None),
 ('E4c','NL2gp1','NL2g-noev',None,None,None),('E4c','PRp_L2','PR_L2-noev',None,None,None),
)


def table(name,rows):
    sys.path.insert(0,str(FRAME))
    from ledger import write as tsv
    encoded=[]
    for row in rows:
        encoded.append({key:json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list)) else value
                        for key,value in row.items()})
    tsv(HERE/'results'/name,encoded)


def measurements():
    rows=[]
    tokens={}
    for matrix in MATRICES:
        for cell in CELLS:
            for number in range(3):
                path=round_path(matrix,cell,number)
                data=read(path,{})
                if data.get('invalidated'):continue
                for label,record in data.get('arms',{}).items():
                    if record.get('exit_code') or 'e2e_seconds' not in record:continue
                    arm=label.removesuffix('-120')
                    rows.append(dict(matrix=matrix,cell=cell,arm=arm,round=number,
                          tpot_s=(record['e2e_seconds']-record['ttft_seconds'])/1023,
                          ttft_s=record['ttft_seconds'],e2e_s=record['e2e_seconds'],source=str(path)))
                    files=sorted(Path(record['out']).rglob('tokens_N1024_run1.json'))
                    if files:tokens[matrix,cell,label,number]=read(files[0])
    return rows,tokens


def paired(rows,matrix,cell,candidate,control):
    first={row['round']:row for row in rows if row['matrix']==matrix and row['cell']==cell and row['arm']==candidate}
    second={row['round']:row for row in rows if row['matrix']==matrix and row['cell']==cell and row['arm']==control}
    rounds=sorted(set(first)&set(second))
    result=dict(matrix=matrix,cell=cell,candidate=candidate,control=control,paired_rounds=rounds,
                complete=rounds==[0,1,2],relative=None,saving_us=None)
    if rounds:
        c=[first[number]['tpot_s'] for number in rounds]
        b=[second[number]['tpot_s'] for number in rounds]
        result.update(candidate_tpot_s=statistics.median(c),control_tpot_s=statistics.median(b),
                      per_round_relative=[x/y-1 for x,y in zip(c,b)])
        if result['complete']:
            result.update(relative=statistics.median(c)/statistics.median(b)-1,
                          saving_us=(statistics.median(b)-statistics.median(c))*1e6,
                          speedup_tpot=statistics.median(y/x for x,y in zip(c,b)))
            if all('e2e_s' in first[number] and 'e2e_s' in second[number] for number in rounds):
                result['speedup_e2e']=statistics.median(second[number]['e2e_s']/first[number]['e2e_s'] for number in rounds)
    return result


def medians(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['matrix'],row['cell'],row['arm']].append(row)
    result=[]
    for (matrix,cell,arm),items in sorted(groups.items()):
        record=dict(matrix=matrix,cell=cell,arm=arm,rounds=len(items),complete=len(items)==3)
        for metric in ('tpot_s','ttft_s','e2e_s'):
            values=[item[metric] for item in items]
            record.update({metric+'_median':statistics.median(values),metric+'_range':max(values)-min(values),
                           metric+'_samples':values})
        result.append(record)
    return result


def old_tables():
    path=FRAME/'results/measurements.tsv'
    if path.exists():
        with path.open() as stream:
            return [dict(row,round=int(row['round']),tpot_s=float(row['tpot_s']),e2e_s=float(row['e2e_s']))
                    for row in csv.DictReader(stream,delimiter='\t')]
    rows=[]
    for name in ('T2','T5','T6','T10'):
        path=FRAME/f'results/{name}.tsv'
        if path.exists():
            with path.open() as stream:rows.extend(dict(row,table=name) for row in csv.DictReader(stream,delimiter='\t'))
    return rows


def old_effect(rows,matrix,cell,candidate,control):
    if matrix is None:return None,'n/a: sm89 has no PDL'
    if any('round' in row for row in rows):
        result=paired(rows,matrix,cell,candidate,control)
        return result['relative'],('final sm89 evidence, same group and rounds 0/1/2; original canary flags retained'
                                  if result['complete'] else 'final sm89 paired group missing/incomplete')
    first=next((row for row in rows if row.get('matrix')==matrix and row.get('cell')==cell and row.get('arm')==candidate),None)
    second=next((row for row in rows if row.get('matrix')==matrix and row.get('cell')==cell and row.get('arm')==control),None)
    if not first or not second:return None,'checkpoint evidence missing; sm89 final HEAD not aligned'
    if int(first['rounds'])!=3 or int(second['rounds'])!=3:return None,'checkpoint paired group incomplete'
    return float(first['tpot_s_median'])/float(second['tpot_s_median'])-1,'checkpoint only; same matrix, three rounds'


def flatten(value,prefix=''):
    result={}
    if isinstance(value,dict):
        for key,item in value.items():result.update(flatten(item,prefix+'.'+key if prefix else key))
    else:result[prefix]=value
    return result


def collect():
    out=HERE/'results'
    out.mkdir(exist_ok=True)
    rows,tokens=measurements()
    table('measurements.tsv',rows)
    median=medians(rows)
    table('medians.tsv',median)
    alignment=read(HERE/'baseline_alignment.json',{})
    aligned=alignment.get('compiler_runtime_aligned',False)
    provenance='final sm89 evidence; device/compiler sources aligned with allowed native-profile port' if aligned else 'checkpoint; final common baseline not aligned'
    current=read(HERE/'target_sm120.json',{})
    previous=read(FRAME/'raw/inputs/target_r12b.json',{})
    flat,old=flatten(current),flatten(previous)
    table('S2.tsv',[dict(field=key,sm120=value,sm89=old.get(key),
          comparison=provenance+'; fixed target retained, sustained TL-2 ceiling separate') for key,value in flat.items()])
    table('S1.tsv',[dict(kind='environment',value=read(HERE/'env_sm120.json')),
          dict(kind='capabilities',value=read(HERE/'caps_sm120.json')),
          dict(kind='native_profile_port',commit='17860b184',synchronization_modified=False),
          dict(kind='paged_loader',value='fixed tiled-weight builds: LoadTile -> PublishBulk/PublishBulkHint; Tensor2D is in the row-layout LoadRow path',
               evidence='source and manifest inference; compiled instruction audit is separate'),
          dict(kind='paged_attention_TMA',value='PagedAttentionTaskBody KV loader also uses tensor maps and Tensor2D; tensor.2d exists even in tiled-weight builds',
               evidence='verified compiled PTX; runtime branch frequency not measured'),
          dict(kind='codegen',value=read(HERE/'raw/E3_codegen_r5/audit.json',[])),
          dict(kind='baseline_alignment',value=alignment),
          dict(kind='PDL_position_replay',value=read(HERE/'raw/acceptance_03/pdl_replay.json',[])),
          dict(kind='solver_rejections',value=read(HERE/'raw/acceptance_03/snapshot.json',{}).get('fixed_failures',[])),
          dict(kind='MB1c_quarantine',value='TN128/TK64 method5; 12 points unavailable, not corrected'),
          dict(kind='optional_omitted',value=['E4e','E5e'])])
    ceiling=read(HERE/'raw/E1_TL2_r3/processes/dram_ceiling.json',{})
    maximum=ceiling.get('maximum_gbps')
    theory=read(HERE/'raw/E0_environment/device.json',{}).get('theoretical_bandwidth_gbps')
    mb=[]
    table('S2_processes.tsv',ceiling.get('processes',[]))
    for process in ceiling.get('processes',[]):
        data=read(process['path'],{})
        mb.extend(dict(item,source=process['path'],pid=process['pid'],contaminated=process['contaminated'])
                  for item in data.get('points',[]))
    for suite,attempt in (('b','r3'),('c','r4'),('d','r3'),('e','r3'),('f','r3')):
        path=HERE/f'raw/E2a_MB-1{suite}_{attempt}/loadbench.json'
        data=read(path,{})
        for item in data.get('points',[]):
            item=dict(item,source=str(path),contaminated=data.get('contaminated'))
            if item.get('gbps') is not None and maximum:item['ratio_mb1a_max']=item['gbps']/maximum
            mb.append(item)
    for item in mb:
        if item.get('gbps') is not None and maximum:item['ratio_mb1a_max']=item['gbps']/maximum
    table('S3_points.tsv',mb)
    sm89_mb=[]
    path=FRAME/'results/T3.tsv'
    if path.exists():
        with path.open() as stream:sm89_mb=list(csv.DictReader(stream,delimiter='\t'))
    old_ceiling=next((item for item in sm89_mb if item['suite']=='MB-1a'),{})
    table('S2_ceiling.tsv',[dict(architecture='sm120',**ceiling),dict(architecture='sm89',**old_ceiling)])
    summary=[]
    for suite in sorted({item['suite'] for item in mb}):
        points=[item for item in mb if item['suite']==suite]
        clean=[item['gbps'] for item in points if item.get('gbps') is not None and not item.get('contaminated')]
        older=[item for item in sm89_mb if item.get('suite')==suite]
        older=[dict(item,normalized_maximum=float(item['maximum_gbps'])/float(sm89_mb[0]['maximum_gbps'])
                    if item.get('maximum_gbps') and sm89_mb[0].get('maximum_gbps') else None) for item in older]
        summary.append(dict(suite=suite,points=len(points),unsupported=sum(item.get('status')=='unsupported' for item in points),
             maximum_gbps=max(clean) if clean else None,ratio_mb1a_max=max(clean)/maximum if clean and maximum else None,
             sm89_reference=older,sm89_qualification=provenance))
    table('S3.tsv',summary)
    anchor=[]
    for row in median:
        if row['matrix']!='E4a':continue
        row=dict(row)
        pair=paired(rows,'E4a',row['cell'],row['arm'],'vllm')
        row['TM_vllm_tpot']=pair.get('speedup_tpot')
        row['TM_vllm_e2e']=pair.get('speedup_e2e')
        floor=read(HERE/f"raw/E3_catalog_r5/floors/{row['cell']}/decode/floor.json",{})
        floor_point=next((point for point in floor.get('points',[]) if point['past']==575),None)
        if floor_point:
            row['tpot_over_floor_mid']=row['tpot_s_median']*1e9/floor_point['floor_ns']
            row['mid_bytes']=floor_point['dram_ns']*ceiling['calibration_median_gbps']
            row['effective_gbps_over_mb1a_max']=row['mid_bytes']/(row['tpot_s_median']*1e9)/maximum
        old_path=FRAME/'results/T10.tsv'
        old_anchor=[]
        if old_path.exists():
            with old_path.open() as stream:old_anchor=list(csv.DictReader(stream,delimiter='\t'))
        old_label={'B0':'B0-D'}.get(row['arm'],row['arm'])
        reference=next((item for item in old_anchor if item['cell']==row['cell'] and item['arm']==old_label),None)
        row['sm89_T10']=reference
        row['sm89_T10_status']=provenance
        if reference and reference.get('tm_vllm_median') and row['TM_vllm_e2e'] is not None:
            row['sm120_minus_sm89_TM_vllm_e2e']=row['TM_vllm_e2e']-float(reference['tm_vllm_median'])
        row['TM_vllm_definition']='median of same-round vLLM/TM E2E ratios follows R13 T10; TPOT ratio reported separately'
        anchor.append(row)
    table('S4.tsv',anchor)
    effects=[]
    old=old_tables()
    for cell in CELLS:
        for matrix,candidate,control,old_matrix,old_candidate,old_control in PAIRS:
            effect=paired(rows,matrix,cell,candidate,control)
            value,note=old_effect(old,old_matrix,cell,old_candidate,old_control)
            effect.update(sm89_relative=value,sm89_qualification=note,
                          same_sign=(effect['relative']*value>0) if effect['relative'] is not None and value is not None else None)
            effects.append(effect)
    table('S5.tsv',effects)
    sys.path.insert(0,str(FRAME))
    from ledger import stages,steps,stage_semantics
    from analyze import trace_v2
    from page_chain import pages,chain
    stage_rows=[]
    step_rows=[]
    page_rows=[]
    task_rows=[]
    errors=[]
    for cell in CELLS:
        floor=read(HERE/f'raw/E3_catalog_r5/floors/{cell}/decode/floor.json',{})
        mid=next((point for point in floor.get('points',[]) if point['past']==575),None)
        byte_count=mid['dram_ns']*ceiling['calibration_median_gbps'] if mid and ceiling else None
        for stage in ('E5a','E5b','E5c','E5d'):
            folder=HERE/f'raw/{stage}_{cell}_r5'
            for file in folder.rglob('stage_trace.tsv'):
                try:
                    semantics=stage_semantics(file.parent)
                    stage_rows.extend(dict(row,cell=cell,source=str(file.parent),**semantics[row['stage']]) for row in stages(file.parent,int(cell.split('_B')[1]),{'native':ceiling['calibration_median_gbps'],'mb1a_max':maximum}))
                except (ValueError,KeyError,OSError) as error:errors.append(dict(source=str(file),error=str(error)))
            for file in folder.rglob('step_trace.tsv'):
                try:step_rows.extend(dict(row,cell=cell,source=str(file.parent)) for row in steps(file.parent))
                except (ValueError,KeyError,OSError) as error:errors.append(dict(source=str(file),error=str(error)))
            for file in folder.rglob('slots.tsv'):
                try:
                    tasks,hol,reducers=trace_v2(file.parent)
                    task_rows.extend(dict(row,cell=cell,record_kind=kind) for kind,items in (('task',tasks),('hol',hol),('reducer',reducers)) for row in items)
                except (ValueError,KeyError,OSError) as error:errors.append(dict(source=str(file),error=str(error)))
            for file in folder.rglob('page_trace.tsv'):
                if byte_count is None:
                    errors.append(dict(source=str(file),error='native request floor missing; page ratios not computed'))
                    continue
                try:
                    page_rows.extend(dict(row,cell=cell) for row in pages(file,byte_count,{'native':ceiling['calibration_median_gbps'],'mb1a_max':maximum}))
                    if (file.parent/'slots.tsv').exists():
                        links,record=chain(file.parent)
                        page_rows.append(dict(record,cell=cell,source=str(file),record_kind='chain'))
                        table(f"chain_{cell}_{file.parent.name}.tsv",links)
                except (ValueError,KeyError,OSError) as error:errors.append(dict(source=str(file),error=str(error)))
    table('S6.tsv',stage_rows)
    grouped=defaultdict(list)
    for row in stage_rows:grouped[row['cell'],row['source'],row['past'],row['iteration'],row['semantic_kind']].append(row)
    table('S6_kinds.tsv',[dict(cell=key[0],source=key[1],past=key[2],iteration=key[3],kind=key[4],
          duration_ns=sum(row['duration_ns'] for row in values),excess_ns_native=sum(row['excess_ns_native'] for row in values),
          excess_ns_mb1a_max=sum(row['excess_ns_mb1a_max'] for row in values)) for key,values in grouped.items()])
    # Keep the other architecture's measurements separate; never reuse its binaries.
    for native,old_name in (('S6','T4'),('S7','T6_steps'),('S8','T9')):
        source=FRAME/f'results/{old_name}.tsv'
        if source.exists():
            with source.open() as stream:old_rows=list(csv.DictReader(stream,delimiter='\t'))
            table(native+'_sm89_reference.tsv',[dict(row,qualification=provenance) for row in old_rows])
        else:table(native+'_sm89_reference.tsv',[dict(status='not supplied',source=str(source))])
    table('S7.tsv',step_rows)
    table('S7_tasks.tsv',task_rows)
    table('S7_pages.tsv',page_rows)
    table('analysis_errors.tsv',errors)
    from fidelity import candidates
    choices=[]
    for model in ('llama','qwen3'):
        run=HERE/f'raw/E3_R13F_{model}_r4/run'
        if not (run/'plans.json').exists():
            failure=read(HERE/'raw/acceptance_03/snapshot.json',{}).get('joint_failures',[])
            choices.extend(dict(stage='joint',status='unavailable',**item) for item in failure if item['model']==model)
            for command_file in sorted((run/'commands').glob('build-*/command.json')):
                command_record=read(command_file,{})
                arguments=command_record.get('argv',[])
                if '--options' not in arguments:continue
                options=read(arguments[arguments.index('--options')+1],[])
                if len(options)<2:continue
                phase=options[options.index('--serving')+1]
                batch=options[options.index('--batch')+1]
                pg=options[options.index('--pg')+1]
                choices.extend(dict(row,model=model,batch=batch,phase=phase,pg=pg,
                                    qualification='partial first-level candidate results only; final pg/executor/loop joint selection failed')
                               for row in candidates(Path(options[1])))
        for batch,pair in read(run/'plans.json',{}).items():
            if not batch.isdigit():continue
            for phase in ('prefill','decode'):
                if phase in pair:
                    choices.extend(dict(row,model=model,batch=batch,phase=phase) for row in candidates(Path(pair[phase])))
            for item in pair.get('decode_pg_choice',{}).get('candidates',[]):
                choices.append(dict(model=model,batch=batch,stage='joint',candidate=item))
    table('S8.tsv',choices)
    checks=[dict(cell=cell,**record) for cell in CELLS for record in read(HERE/f'raw/E4f_{cell}_r5/checks.json',[])]
    checks.extend(dict(check='smoke',**row) for row in read(HERE/'raw/E2b_smoke_r4/results.json',[]))
    for group in GROUPS:checks.append({'check':'50 fresh processes','group':group,**read(HERE/f'raw/E2c_{group}_r5/status.json',{})})
    checks.append(dict(check='first-execution synchronization fixes',status='n/a: no synchronization code fix so far'))
    checks.append(dict(check='vLLM wrapper CPU replay',report=read(HERE/'raw/acceptance_03/anchor_replay.json')))
    checks.append(dict(check='PDL PTX CPU replay',report=read(HERE/'raw/acceptance_03/pdl_replay.json')))
    table('S9.tsv',checks)
    predictions=[]
    bulk=[item['gbps'] for item in mb if item.get('bulk') and item.get('loader_warps')==1 and not item.get('consume') and not item.get('contaminated')]
    initial={'dram_theory':ceiling['calibration_median_gbps']/theory if theory and ceiling.get('calibration_median_gbps') is not None else None,
             'ceiling_calibration':maximum/ceiling['calibration_median_gbps'] if maximum else None,
             'bulk_loader':max(bulk)/maximum if bulk and maximum else None,
             'llama_B1_TM_vllm':next((row.get('TM_vllm_e2e') for row in anchor if row['cell']=='llama_B1' and row['arm']=='R13F'),None)}
    t1=FRAME/'results/T1.tsv'
    if t1.exists() and maximum:
        with t1.open() as stream:old_anchor=list(csv.DictReader(stream,delimiter='\t'))
        b0=next((item for item in old_anchor if item.get('cell')=='llama_B1' and item.get('arm')=='B0'),None)
        old_floor=read(FRAME/'raw/inputs/llama_decode_B1_floor.json',{})
        point=next((item for item in old_floor.get('points',[]) if item['past']==575),None)
        old_peak=next((float(item['maximum_gbps']) for item in sm89_mb if item['suite']=='MB-1a' and item.get('maximum_gbps')),None)
        native=next((item for item in anchor if item['cell']=='llama_B1' and item['arm']=='B0'),{})
        if b0 and point and old_peak and native.get('effective_gbps_over_mb1a_max') is not None:
            # R13 analyze.py's inherited floor-byte reconstruction convention.
            old_ratio=point['dram_ns']*884.5010943/(float(b0['tpot_s_median'])*1e9)/old_peak
            initial['llama_B1_effective_bandwidth']=(native['effective_gbps_over_mb1a_max']-old_ratio)*100
    map_pairs={'PR_L1_B0h':('E4d','PR_L1','B0h'),'B0p_B0_noev':('E4c','B0p','B0-noev'),
               'B0p1_B0_noev':('E4c','B0p1','B0-noev'),'NL2gp1_NL2g_noev':('E4c','NL2gp1','NL2g-noev'),
               'PRp_PR_noev':('E4c','PRp_L2','PR_L2-noev'),'B0l_B0_noev':('E4c','B0l','B0-noev'),
               'NL2g_B0':('E4b','NL2g','B0'),'NL2e_B0':('E4b','NL2e','B0')}
    for prediction in read(HERE/'predictions_sm120.json')['predictions']:
        id_=prediction['id']
        observed=[paired(rows,map_pairs[id_][0],cell,map_pairs[id_][1],map_pairs[id_][2]) for cell in CELLS] if id_ in map_pairs else [dict(value=initial.get(id_))]
        if id_=='step_boundary':
            observed=[dict(mode=item['mode'],value=item['overhead_ns_per_step']/1000 if item['overhead_ns_per_step']>=0 else None,
                           raw_residual_us=item['overhead_ns_per_step']/1000,median_ms=item.get('median_ms'),
                           reason='negative cold-stream subtraction is not a physical boundary latency' if item['overhead_ns_per_step']<0 else 'protocol residual')
                      for item in mb if item['suite']=='MB-1e' and 'overhead_ns_per_step' in item]
            if not observed:observed=[dict(value=None,reason='MB-1e result not collected')]
        for observation in observed:
            value=observation.get('relative',observation.get('value'))
            matches=None
            if value is not None and 'range' in prediction:matches=prediction['range'][0]<=value<=prediction['range'][1]
            elif value is not None and 'minimum' in prediction:matches=value>=prediction['minimum']
            if id_ in ('NL2g_B0','NL2e_B0'):
                entry=next((item for item in effects if item['cell']==observation['cell'] and item['candidate']==map_pairs[id_][1] and item['matrix']=='E4b'),{})
                matches=entry.get('same_sign')
            predictions.append(dict(prediction=prediction,observation=observation,matches=matches,
                    saving_us_matches=(prediction['saving_us'][0]<=observation['saving_us']<=prediction['saving_us'][1])
                    if prediction.get('saving_us') and observation.get('saving_us') is not None else None,
                    cross_architecture_final_head_aligned=aligned,status='measured' if value is not None else 'unavailable/invalid residual'))
    table('S10.tsv',predictions)
    write(HERE/'raw/E6_collect_r5/completeness.json',dict(round_records=len(rows),expected_round_records=297,
          measurements_complete=all(len([row for row in rows if row['matrix']==matrix and row['cell']==cell and row['arm']==label])==3
          for matrix,labels in MATRICES.items() for cell in CELLS for label in labels if not(label=='B0h' and cell=='qwen3_B16')),
          errors=errors,sm89_final_head_aligned=aligned,model_hashes_verified=alignment.get('cross_architecture_model_hashes_verified',False),all_gpu_unit_tests_passed=False,
          correctness_failures_preserved=True,final_human_review_pending=True))
    draft=[ '# R13 sm120 自动汇总（待最终验收）','',
            '本轮在 sm89 R13 Phase D 尚未完成时从 `36f17e6ee` 启动。',
            f"最终 sm89 HEAD：`{alignment.get('sm89_final_head','未提供')}`；编译器/runtime 源码对齐：{aligned}；保留原二进制及允许的 native profile 移植。",
            '跨机器模型 config/权重 SHA 未齐，vLLM 版本分别为 sm120 0.29.0 / sm89 0.30.0；sm89 的两个金丝雀标记仍保留。',
            f"Prompt SHA256：`{read(HERE/'start.json')['prompt_sha256']}`。",
            'verified：保留安装的 vLLM 0.29.0；未修改求解器/代价模型、未调优、未 push。',
            '移植修正：native-only target 格式兼容 `17860b184`；无同步代码修正，无虚构 50 进程结论。',
            'MB-1c TN128/TK64 method5 十二点隔离为不可用；E4e/E5e 按可选项省略。','',
            '带宽口径：MB-1a 最大值是重复拷贝有效吞吐，不冒充物理 DRAM 带宽；MB-1e 负冷流扣减仅保留残差，不称负步边界延迟。',
            '## S1–S10','']
    draft += [f'- [S{number}](results/S{number}.tsv)：自动收集，缺失项不填零、不记通过。' for number in range(1,11)]
    draft += ['', '## 四个问题与结论边界','',
              '- 机制跨架构变化：S5 提供同组同轮次对照；结合源码、模型及 sm89 原始测量限制判读同号/异号。',
              '- bulk 分页是否胜过非分页 L1：按 S5 的 PR_L1/B0h；不是拿 bulk-only 微基准替代模型 TPOT。',
              '- B1 优势：S4 对照最终 T10；本机 R13F 求解失败时不能用 B0 冒充 R13F 或宣称预测成立。',
              '- PDL 回收量：S5 的 B0p/B0p1、NL2gp1、PRp 配对 saving_us；S9 同步证据不齐时不得宣称 verified。','',
              '## R14 方案（不实施）','',
              '- 根据 S5/S7 的配对收益与边界开销，为各架构分别规划分页/执行器/循环/PDL 默认值。',
              '- 定位不可用的 cache-hint 形状；对 trace 暴露的等待、固定开销与模型残差提出细粒度方案，不回写代价模型。',
              '- 定位 StageFlowModel MainStart 的分页预取扣减与 DRAM 计数不平衡；仅提出 R14 方案，不改求解器。',
              '- 补齐两机模型 SHA，并保留 vLLM 版本和原始金丝雀的不确定性。','',
              '详细失败、可用性、50 新 PID、原始轮次与金丝雀替代见 raw/；最终结论与本地提交仍需人工验收。']
    observed=next((item for item in anchor if item['cell']=='llama_B1' and item['arm']=='R13F'),{})
    if observed.get('TM_vllm_e2e') is not None:
        draft.append(f"verified（仅本机测量）：Llama B1 的 TM/vLLM E2E 加速比为 {observed['TM_vllm_e2e']:.6f}，TPOT 加速比为 {observed['TM_vllm_tpot']:.6f}；正确性限定见 S9。")
    for cell in CELLS:
        for candidate,control in (('PR_L1','B0h'),('B0p','B0-noev'),('B0p1','B0-noev'),('NL2gp1','NL2g-noev'),('PRp_L2','PR_L2-noev')):
            item=next((item for item in effects if item['cell']==cell and item['candidate']==candidate and item['control']==control),{})
            if item.get('relative') is not None:
                draft.append(f"verified（条件化实测）：{cell} {candidate}/{control} 的 TPOT 变化 {item['relative']*100:+.3f}%，每步回收 {item['saving_us']:.3f} µs；数值/同步是否通过须结合 S9。")
    (HERE/'summary.generated.md').write_text('\n'.join(draft)+'\n')
    return 0


if __name__=='__main__':
    raise SystemExit(collect())
