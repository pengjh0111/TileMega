#!/usr/bin/env python3
"""Collect diagnostic evidence separately from uninstrumented request timing."""
from collections import Counter, defaultdict
import csv
import os
from functools import lru_cache
from pathlib import Path
import statistics as st

from analyze_sm90d1_ledger import analyze, category, tsv
from sm90d1_common import ROOT, RUN, read, write


def records():
    seen = set()
    for state in sorted((RUN / 'scheduler').glob('gpu*/state.json')):
        for job, record in read(state).items():
            if record['status'] != 'done' or not record.get('out'):
                continue
            raw = Path(record['out'])
            if raw in seen or not (raw / 'guard_result.json').exists():
                continue
            seen.add(raw)
            if read(raw / 'guard_result.json')['code'] == 0:
                yield job, raw, state.parent.name[3:]


@lru_cache(None)
def artifacts():
    result = {}
    for path in (RUN / 'build').glob('*/*/*.so.identity.json'):
        value = read(path)
        binary = Path(str(path).removesuffix('.identity.json'))
        value['_binary'] = str(binary)
        diagnostic = Path(str(binary)+'.diagnostic.json')
        value['_diagnostic'] = read(diagnostic) if diagnostic.exists() else {}
        result[value['binary_sha256']] = value
    return result


def traces():
    """Use a nonwrapping ring and a deterministic latest-protocol selection."""
    candidates = {}
    for job, raw, gpu in records():
        selection = raw / 'job/selected-config.json'
        if not selection.exists():
            continue
        selected = read(selection)
        if selected['stage'] not in ('trace', 'trace-control'):
            continue
        artifact = artifacts().get(selected['binary_sha256'])
        if not artifact:
            continue
        config = read(selected['config'])
        tag=Path(artifact['_binary']).parent.parent.name
        if tag.startswith('A-Ec') and tag.endswith('-full') and not job.startswith('validated-'):
            continue
        for folder in (raw / 'job/trace').glob('past*'):
            meta_path = folder / 'trace.json'
            if not meta_path.exists():
                continue
            meta = read(meta_path)
            if meta.get('protocol_version', 0) < 3:
                continue
            key = selected['binary_sha256'], gpu, meta['past']
            candidate = dict(job=job, raw=raw, folder=folder, gpu=int(gpu), selected=selected,
                             artifact=artifact, config=config, meta=meta)
            if key not in candidates or job < candidates[key]['job']:
                candidates[key] = candidate
    return list(candidates.values())


def ledger_data(trace):
    folder = trace['folder']
    for name in ('stage_trace.tsv','task_profile.tsv','step_trace.tsv'):
        latest_tsv(trace,name)
    path = folder / 'ledger.json'
    data = read(path) if path.exists() else {}
    if data.get('floor_model_version') != 3 or data.get('trace') != trace['meta']:
        model = read(ROOT / trace['config']['model_dir'] / 'config.json')
        binary = trace['artifact']['_binary']
        details, closure = analyze(folder, model, trace['config']['batch'], read(binary+'.plan.json'))
        data = dict(floor_model_version=3, stages=details, steps=closure, trace=trace['meta'])
        write(path, data)
    return data


def latest_tsv(trace,name):
    original=trace['folder']/name
    if not original.exists():
        return original
    path=original.with_name(original.stem+'_latest.tsv')
    if path.exists():
        return path
    temporary=path.with_suffix(f'.{os.getpid()}.tmp')
    with original.open() as source, temporary.open('w') as target:
        target.write(next(source))
        for line in source:
            if line.strip() and int(line.split('\t',2)[1]) >= trace['meta']['warmup_steps']:
                target.write(line)
    temporary.replace(path)
    return path


def aggregates(trace):
    if not (trace['folder']/'runtime_stages.tsv').exists():
        return []
    data = ledger_data(trace)
    groups = defaultdict(list)
    for row in data['stages']:
        groups[row['iteration'], row['kind']].append(row)
    categories = defaultdict(list)
    numeric = ('envelope_ns','bytes','floor_ns','body_mean_ns','body_critical_ns','start_ns',
               'first_block_ns','gaps_ns','release_ns','skew_ns','inefficiency_ns',
               'imbalance_ns','residual_ns')
    for (iteration, kind), values in groups.items():
        row = {key:sum(v[key] for v in values) if all(v[key] is not None for v in values)
               else None for key in numeric}
        row.update(tasks=sum(v['tasks'] for v in values),
                   active_ctas=st.median(v['active_ctas'] for v in values),
                   max_tasks_per_cta=max(v['max_tasks_per_cta'] for v in values),
                   profile_complete=all(v['profile_complete'] for v in values))
        categories[kind].append(row)
    output = []
    for kind, values in categories.items():
        row = dict(job=trace['job'], tag=Path(trace['artifact']['_binary']).parent.parent.name,
                   model=trace['config']['model'], batch=trace['config']['batch'],
                   gpu=trace['gpu'], past=trace['meta']['past'], kind=kind, steps=len(values))
        row['role']='first_block_probe' if row['tag'].startswith('ready-') else 'stage_ledger'
        row['binary_sha256']=trace['artifact']['binary_sha256']
        row['source_head']=trace['artifact']['source']['head']
        row.update({key:st.median(v[key] for v in values) if all(v[key] is not None for v in values)
                    else None for key in values[0]})
        row['event_step_ms'] = trace['meta']['mean_step_ms']
        row['trace_step_ms'] = st.median(v['step_ns'] for v in data['steps'])/1e6 if data['steps'] else None
        row['boundary_ns'] = st.median(v['boundary_ns'] for v in data['steps']) if data['steps'] else None
        row['step_event_difference_pct'] = (row['trace_step_ms']/row['event_step_ms']-1)*100 if row['trace_step_ms'] else None
        row['closure_within_3pct'] = abs(row['step_event_difference_pct']) <= 3 if row['trace_step_ms'] else False
        row['evidence'] = 'verified timestamps; inferred compulsory-byte floor; zero-KV diagnostic'
        output.append(row)
    return output


def ledger_tables(trace_rows, references):
    rows = [r for trace in trace_rows for r in aggregates(trace)]
    controls = {}
    stage_only = {}
    for trace in trace_rows:
        identity = trace['artifact']
        key = identity['artifact_id'], trace['gpu'], trace['meta']['past']
        if not identity['trace']:
            controls[key] = trace['meta']['mean_step_ms']
        defines = identity['_diagnostic'].get('defines', [])
        if 'TILEMEGA_TRACE_STAGE=1' in defines and 'TILEMEGA_TRACE_TASK=1' not in defines:
            for row in aggregates(trace):
                stage_only[identity['_diagnostic'].get('parent'), trace['gpu'], trace['meta']['past'], row['kind']] = row['envelope_ns']
    by_reference = {(r['model'],r['batch'],r['gpu']):r['tpot_ms'] for r in references}
    for row in rows:
        trace = next(t for t in trace_rows if t['job']==row['job'] and t['meta']['past']==row['past'])
        identity = trace['artifact']
        parent = identity['_diagnostic'].get('parent')
        control = controls.get((parent, row['gpu'], row['past']))
        row['matched_control_ms'] = control
        row['instrumentation_overhead_ms'] = row['event_step_ms']-control if control is not None else None
        reference = by_reference.get((row['model'],row['batch'],row['gpu']))
        row['trace_minus_request_tpot_ms'] = row['event_step_ms']-reference if reference else None
        row['short_window_bias_ms'] = control-reference if control is not None and reference else None
        stage = stage_only.get((parent,row['gpu'],row['past'],row['kind']))
        row['full_vs_stage_envelope_pct'] = (row['envelope_ns']/stage-1)*100 if stage else None
        row['distorted_over_10pct'] = abs(row['full_vs_stage_envelope_pct'])>10 if stage else None
    return rows


def phase_rows(trace):
    path = latest_tsv(trace,'task_profile.tsv')
    if not path.exists():
        return
    minimum = trace['meta']['warmup_steps']
    for row in tsv(path):
        if int(row['iteration']) < minimum:
            continue
        yield {k:v if k=='kind' else int(v) for k,v in row.items()}


def frequency(trace):
    values = [(r['local_end']-r['local_begin'])/(r['run_end']-r['run_begin'])
              for r in phase_rows(trace) if r.get('local_begin') and r['run_end']-r['run_begin']>=10000]
    return st.median(values) if values else None


def attention(trace_rows, ledger):
    result = []
    for trace in trace_rows:
        if 'TILEMEGA_TRACE_TASK=1' not in trace['artifact']['_diagnostic'].get('defines', []):
            continue
        tag = Path(trace['artifact']['_binary']).parent.parent.name
        if tag not in ('full2','A-Ec64-full','A-Ec256-full'):
            continue
        stages = {int(r['stage']):r for r in tsv(trace['folder']/'runtime_stages.tsv')}
        freq = frequency(trace)
        if not freq:
            continue
        groups = defaultdict(list)
        for row in phase_rows(trace):
            stage = stages[row['stage']]
            if stage['kind_name'] in ('kFusedAttention','kAttentionMerge'):
                groups[stage['kind_name']].append(row)
        config = read(ROOT / trace['config']['model_dir'] / 'config.json')
        d = config.get('head_dim',config['hidden_size']//config['num_attention_heads'])
        for kind, values in groups.items():
            active = [r for r in values if r['query_ns'] or r['wave_compute_ns'] or r['first_ready']] if kind=='kFusedAttention' else values
            if not active:
                continue
            row = dict(model=trace['config']['model'], batch=trace['config']['batch'],
                       gpu=trace['gpu'], past=trace['meta']['past'], tag='full2' if tag=='ready-full' else tag,
                       kind='attention' if kind=='kFusedAttention' else 'merge', tasks=len(values),
                       active_tasks=len(active), assigned_ctas=len({r['worker'] for r in values}),
                       active_ctas=len({r['worker'] for r in active}), clock_cycles_per_ns=freq,
                       task_median_ns=st.median(r['run_end']-r['run_begin'] for r in active),
                       task_p99_ns=sorted(r['run_end']-r['run_begin'] for r in active)[min(len(active)-1,int(.99*len(active)))],
                       historical_bytes_per_task=st.median(r['bytes'] for r in active))
            row['sampled_steps'] = trace['meta']['measured_steps']
            row['layers'] = config['num_hidden_layers']
            row['tasks_per_layer_step'] = len(values)/(row['sampled_steps']*row['layers'])
            row['active_tasks_per_layer_step'] = len(active)/(row['sampled_steps']*row['layers'])
            for field in ('query_ns','first_page_wait_ns','later_page_wait_ns','wave_compute_ns',
                          'partial_store','new_row_warp0','new_row_warp1','new_row_warp2','new_row_warp3'):
                row[field+'_median_ns'] = st.median(r.get(field,0)/freq for r in active)
            row['new_row_owner_task_records'] = sum(any(r.get(f'new_row_warp{i}',0) for i in range(4)) for r in active)
            for warp in range(4):
                durations = [r.get(f'new_row_warp{warp}',0)/freq for r in active if r.get(f'new_row_warp{warp}',0)]
                row[f'new_row_warp{warp}_conditional_median_ns'] = st.median(durations) if durations else None
            row['compute_excluding_warp0_new_row_ns'] = st.median((r['wave_compute_ns']-r.get('new_row_warp0',0))/freq for r in active)
            row['task_effective_GBs'] = st.median(r['bytes']/(r['run_end']-r['run_begin']) for r in active if r['run_end']>r['run_begin'])
            stage = next((r for r in ledger if r['job']==trace['job'] and r['past']==row['past'] and r['kind']==row['kind']),None)
            stage_only = next((r for r in ledger if tag=='full2' and r['tag']=='stage2' and r['model']==row['model'] and r['batch']==row['batch'] and r['gpu']==row['gpu'] and r['past']==row['past'] and r['kind']==row['kind']),None)
            kv_bytes = 4*row['batch']*config['num_key_value_heads']*(row['past']+3.5)*d*config['num_hidden_layers']
            row['stage_instrumented_kv_GBs'] = kv_bytes/stage['envelope_ns'] if stage and row['kind']=='attention' else None
            row['stage_only_kv_GBs'] = kv_bytes/stage_only['envelope_ns'] if stage_only and row['kind']=='attention' else None
            row['evidence'] = 'verified zero-KV diagnostic; phase fields converted from local cycles; warp phases overlap'
            result.append(row)
    return result


def arm_details(trace_rows, ledger):
    output = []
    for trace in trace_rows:
        tag = Path(trace['artifact']['_binary']).parent.parent.name
        if tag in ('full2','stage2','reference2','ready-full','ready-stage') or not trace['artifact']['trace']:
            continue
        own = [r for r in ledger if r['job']==trace['job'] and r['past']==trace['meta']['past']]
        task_profile = 'TILEMEGA_TRACE_TASK=1' in trace['artifact']['_diagnostic'].get('defines', [])
        reference_tag = 'full2' if task_profile else 'stage2'
        reference = [r for r in ledger if r['tag']==reference_tag and r['model']==trace['config']['model'] and r['batch']==trace['config']['batch'] and r['gpu']==trace['gpu'] and r['past']==trace['meta']['past']]
        row = dict(record='diagnostic_delta', arm=tag, model=trace['config']['model'], batch=trace['config']['batch'],
                   gpu=trace['gpu'], past=trace['meta']['past'], reference_trace=reference_tag,
                   trace_ms=trace['meta']['mean_step_ms'])
        for field in ('envelope_ns','inefficiency_ns','imbalance_ns','start_ns','gaps_ns','release_ns','residual_ns'):
            a = [v[field] for v in own]; b = [v[field] for v in reference]
            row[field+'_delta'] = sum(a)-sum(b) if a and b and None not in a+b else None
        runtime = list(tsv(trace['folder']/'runtime_stages.tsv'))
        row['effective_stages'] = sum(not int(r['elided']) for r in runtime)
        row['elided_stages'] = sum(int(r['elided']) for r in runtime)
        if task_profile:
            freq = frequency(trace)
            phases = [r for r in phase_rows(trace) if r.get('la_total')]
            if freq and phases:
                row.update(la_participants=len(phases), la_total_median_ns=st.median(r['la_total']/freq for r in phases),
                    la_protocol_median_ns=st.median((r['la_total']-r['la_ns'])/freq for r in phases),
                    la_reduction_median_ns=st.median(r['la_ns']/freq for r in phases if r['la_ns']) if any(r['la_ns'] for r in phases) else None)
                critical={(r['iteration'],r['stage']):r['critical_cta'] for r in ledger_data(trace)['stages']}
                by_iteration=defaultdict(lambda:[0.,0.])
                for phase in phases:
                    if phase['worker']==critical.get((phase['iteration'],phase['stage'])):
                        by_iteration[phase['iteration']][0]+=(phase['la_total']-phase['la_ns'])/freq
                        by_iteration[phase['iteration']][1]+=phase['la_ns']/freq
                if by_iteration:
                    row['la_critical_protocol_per_step_ns']=st.median(v[0] for v in by_iteration.values())
                    row['la_critical_reduction_per_step_ns']=st.median(v[1] for v in by_iteration.values())
        row['evidence'] = 'verified instrumented zero-KV delta; not an additive decomposition of uninstrumented TPOT'
        output.append(row)
        for value in own:
            before = next((r for r in reference if r['kind']==value['kind']),None)
            if before is None:
                continue
            detail = {k:row[k] for k in ('arm','model','batch','gpu','past','reference_trace')}
            detail.update(record='diagnostic_category_delta',kind=value['kind'],
                          evidence='verified instrumented category delta')
            for field in ('envelope_ns','inefficiency_ns','imbalance_ns','start_ns','gaps_ns','release_ns','residual_ns'):
                detail[field+'_delta'] = value[field]-before[field] if value[field] is not None and before[field] is not None else None
            output.append(detail)
        slots = trace['folder']/'v2/slots.tsv'
        if slots.exists():
            groups = defaultdict(list)
            for slot in tsv(slots):
                slot = {k:int(v) for k,v in slot.items()}
                groups[slot['dependency_count']].append(slot)
            for fanin, values in groups.items():
                output.append(dict(record='event_fanin', arm=tag, model=row['model'], batch=row['batch'],gpu=row['gpu'],
                    past=row['past']+7, fanin=fanin, tasks=len(values),
                    wait_median_ns=st.median(r['ready']-r['wait_begin'] for r in values),
                    dispatch_median_ns=st.median(r['run_begin']-r['ready'] for r in values),
                    run_median_ns=st.median(r['run_end']-r['run_begin'] for r in values),
                    publish_median_ns=st.median(r['publish_end']-r['run_end'] for r in values),
                    event_wait_count_median=st.median(r['wait_count'] for r in values),
                    evidence='verified V2 last host-launch slot; fanin is dependency_count'))
        page = trace['folder']/'page_trace.tsv'
        if page.exists():
            values = [{k:int(v) for k,v in r.items()} for r in tsv(page)]
            write(trace['folder']/'page-summary.json',dict(normalization_visits=17, columns=list(values[0]) if values else [], totals={k:sum(r[k] for r in values) for k in values[0]} if values else {}))
            summary=dict(record='page_counters', arm=tag, model=row['model'],batch=row['batch'],gpu=row['gpu'],past=row['past'],
                page_summary=str((trace['folder']/'page-summary.json').relative_to(ROOT)),
                evidence='verified cumulative counters; 17 visits per logical step; loader and compute overlap')
            by_step=defaultdict(list)
            for value in values:
                if value.get('kernel_begin_ns'):
                    by_step[value['step']].append(value)
            for field in ('loader_issue_ns','dependency_wait_ns','page_full_ns','full_and_wait_ns'):
                summary[field+'_max_cta_per_visit']=st.median(max(v[field] for v in group)/17 for group in by_step.values()) if by_step else None
                summary[field+'_sum_cta_per_visit']=st.median(sum(v[field] for v in group)/17 for group in by_step.values()) if by_step else None
            output.append(summary)
    return output


def resources(trace_rows):
    queried = {}
    for job, raw, gpu in records():
        path = raw / 'job/resources.json'
        if path.exists():
            for row in read(path):
                queried[row['binary_sha256']] = dict(row, query_gpu=gpu)
    measured = {}
    for trace in trace_rows:
        if Path(trace['artifact']['_binary']).parent.parent.name not in ('full2', 'stage2'):
            continue
        path = latest_tsv(trace,'stage_trace.tsv')
        if not path.exists():
            continue
        mapping = {}
        for row in tsv(path):
            if int(row['iteration']) >= trace['meta']['warmup_steps']:
                mapping[int(row['worker'])] = int(row['smid'])
        if mapping:
            counts = Counter(mapping.values())
            key = trace['config']['model'],trace['config']['batch']
            measured[key] = dict(trace_gpu=trace['gpu'], trace_grid=len(mapping),
                trace_sms_used=len(counts), trace_max_ctas_per_sm=max(counts.values()))
    result = []
    heads = {read(RUN/f'{name}/receipt.json')['head'] for name in
             ('compiler-v2','compiler-p3','compiler-pg-profile','compiler-ready') if (RUN/f'{name}/receipt.json').exists()}
    for identity in artifacts().values():
        binary = Path(identity['_binary'])
        tag = binary.parent.parent.name
        if identity['trace'] or identity['source']['head'] not in heads or 'lineinfo' in tag or tag.startswith('P3-2'):
            continue
        model,batch = binary.parent.name.rsplit('-B',1)
        kernel_hint = ('l1_kernel' if binary.stem=='prefill' or tag.startswith('PG') else
                       'l2_kernel' if tag.startswith('EV') else 'l1_loop_kernel')
        kernel = next((k for k in identity['resources'] if kernel_hint in k),None)
        data = identity['resources'].get(kernel,{})
        sass_path = RUN/'sass'/tag/binary.parent.name/binary.stem/'resources.json'
        sass = read(sass_path) if sass_path.exists() else {}
        surrogate_tag = 'lineinfo2' if tag=='reference2' else tag+'-lineinfo'
        surrogate_path = RUN/'sass'/surrogate_tag/binary.parent.name/binary.stem/'resources.json'
        surrogate = read(surrogate_path) if surrogate_path.exists() else {}
        same = bool(sass and surrogate and sass['instruction_encoding_sha256'].get(kernel)==surrogate['instruction_encoding_sha256'].get(kernel))
        local = sass.get('functions',{}).get(kernel,{})
        native_mapping = sass.get('line_mapping',{}).get(kernel,{})
        mapping = native_mapping or surrogate.get('line_mapping',{}).get(kernel,{})
        mapped = bool(native_mapping) or same
        query = queried.get(identity['binary_sha256'],{})
        row = dict(tag=tag,model=model,batch=int(batch),phase=binary.stem,
                   binary_sha256=identity['binary_sha256'],source_head=identity['source']['head'],
                   kernel=kernel,registers=data.get('registers'),spill_store_bytes=data.get('spill_store_bytes'),
                   spill_load_bytes=data.get('spill_load_bytes'),stack_bytes=data.get('stack_bytes'),
                   dynamic_smem_bytes=identity['shared_memory_bytes'],
                   static_smem_bytes=sass.get('cubin_resource_usage',{}).get(kernel,{}).get('SHARED'),
                   grid=query.get('grid'),occupancy_limit_ctas_per_sm=query.get('runtime_queried_residency'),
                   occupancy_query_gpu=query.get('query_gpu'),query_error=query.get('query_error'),
                   LDL_instructions=local.get('LDL',0) if sass else None,
                   STL_instructions=local.get('STL',0) if sass else None,
                   lineinfo_identical_instructions=same if surrogate else None,
                   source_mapping='native_lineinfo' if native_mapping else 'instruction_identical_surrogate' if same else 'unavailable',
                   hot_local_instructions=mapping.get('hot',0) if mapped else None,
                   dispatch_or_other_local_instructions=mapping.get('dispatch_or_other',0) if mapped else None,
                   local_locations='|'.join(f'{k}={v}' for k,v in mapping.items() if ':' in k) if mapped else '',
                   evidence='verified compiler facts and occupancy query; trace mapping is a separate instrumented artifact')
        if binary.stem=='decode' and tag=='reference2':
            row.update(measured.get((model,int(batch)),{}))
        result.append(row)
    return result


def rejected_geometry():
    result = []
    seen = set()
    for state in (RUN/'scheduler').glob('gpu*/state.json'):
        for job, record in read(state).items():
            if not job.startswith(('G1','G2')) or '-corr-' not in job or record['status'] not in ('done','failed'):
                continue
            key = job,record.get('out')
            if key in seen:
                continue
            seen.add(key)
            log = state.parent/(job+'.log')
            tail = log.read_text(errors='replace').splitlines() if log.exists() else []
            error = next((r for r in reversed(tail) if any(s in r for s in
                         ('Error:','ValueError:','RuntimeError:','mismatch'))),'')
            result.append(dict(record='correctness_gate',arm=job,gpu=int(state.parent.name[3:]),
                status=record['status'],exit_code=record.get('exit_code'),error=error[:450],
                performance_eligible=record['status']=='done',
                code_location='python/tilemega/serving/r16_run.py:97; device launch failure, root cause unlocalized' if record['status']=='failed' else '',
                evidence='verified; rejected paths have no TPOT'))
    for path in (RUN/'build').glob('G2-*/*/decode.build.json'):
        receipt=read(path)
        if receipt['status']=='completed':
            continue
        log=(path.parent/'decode.log').read_text(errors='replace').splitlines()
        error=next((r for r in reversed(log) if 'tilemega-compile:' in r),'')
        result.append(dict(record='build_gate',arm=path.parent.parent.name,status=receipt['status'],
            exit_code=receipt.get('exit_code'),error=error,performance_eligible=False,
            code_location='lib/Frontend/Frontend.cpp:823',
            evidence='verified existing argmax packing constraint; no fallback implementation'))
    return result


def rejected_mechanisms():
    result=[]
    for path in (RUN/'build/PG-2').glob('*/decode.build.json'):
        receipt=read(path)
        if receipt['status']=='completed':
            continue
        log=(path.parent/'decode.log').read_text(errors='replace').splitlines()
        result.append(dict(record='build_gate',arm='PG',model=path.parent.name,status=receipt['status'],
            error=log[-1] if log else '',code_location='tools/commands/compile.cpp:686',
            evidence='verified existing paged/descriptor-prefetch incompatibility; arm stopped'))
    return result


def solver_validations(paired):
    output = []
    for state in sorted((RUN/'scheduler').glob('gpu*/state.json')):
        for job, receipt in read(state).items():
            if not job.startswith('recover-S-') or '-corr-' not in job:
                continue
            raw = Path(receipt['out']) if receipt.get('out') else None
            selected = raw/'job/selected-config.json' if raw else None
            if not selected or not selected.exists():
                continue
            selection = read(selected)
            config = read(selection['config'])
            arm = job.removeprefix('recover-').split('-corr-')[0]
            row = dict(record='solver_validation', arm=arm, model=config['model'],
                batch=config['batch'], gpu=state.parent.name[3:], job=job,
                binary_sha256=selection['binary_sha256'], status=receipt['status'],
                exit_code=receipt.get('exit_code'), correctness_output=str(raw/'job/measurements'),
                evidence='verified scheduler receipt; failed correctness paths never enter TPOT')
            for gate in ('c1','c2'):
                path = raw/f'job/measurements/{gate}.json'
                if path.exists():
                    value = read(path)
                    row[gate+'_pass'] = value['pass']
                    if gate == 'c1':
                        row['c1_positions'] = value['positions']
                        row['c1_max_gap'] = value['max_gap']
                    else:
                        row['c2_executor_checks'] = len(value['checks'])
            guard = raw/'guard_result.json'
            passed = (receipt['status']=='done' and guard.exists() and read(guard)['code']==0
                      and row.get('c1_pass') is True and row.get('c2_pass') is True)
            timing = next((r for r in paired if r['arm']==arm and str(r['gpu'])==row['gpu']),None)
            if timing:
                if not passed:
                    raise ValueError('solver TPOT has no accepted C-1/C-2: '+arm)
                row.update(timing)
                row['status'] = 'correctness_and_paired_timing_validated'
            elif passed:
                row['status'] = 'correctness_passed_timing_pending'
            elif receipt['status']=='failed':
                log = state.parent/(job+'.log')
                lines = log.read_text(errors='replace').splitlines() if log.exists() else []
                row['reason'] = next((s for s in reversed(lines) if any(w in s for w in
                    ('Error','error:','failed','failure'))), '\n'.join(lines[-3:]))[-500:]
            output.append(row)
    return output


def solver(ledger, paired=()):
    blockers = [
        ('native_seed','tools/commands/compile.cpp:586','native seed requires an unsolved plan; solve uses legacy seed'),
        ('wgmma_calibration','configs/targets/sm_90_h800.json:1738','scope excludes calibrated Hopper WGMMA bodies'),
        ('candidate_impl','lib/Solver/SkeletonSearch.cpp:807','WGMMA requires external serving_impl; not an endogenous implementation sweep'),
        ('gemv_bulk','include/tilemega/Solver/OperatorClasses.h','GEMV-bulk load policy is an external option'),
        ('execution_mode','lib/Solver/StageFlowModel.cpp:32','model does not distinguish L1 device-loop grid barrier from per-task event execution'),
        ('pricing_key','lib/Solver/TaskPriceParts.cpp:197','WGMMA shares the non-GEMV pricing family with mma16'),
    ]
    output = [dict(record='static_blocker',item=k,code_location=path,reason=reason,evidence='verified static inspection; performance implications inferred') for k,path,reason in blockers]
    for directory in (RUN/'solver').iterdir():
        if not directory.is_dir() or not (directory/'class-map-conversion.json').exists():
            continue
        config = read(ROOT/f'configs/serving/sm90d1/reference/{directory.name}.json')
        for label in ('solve','evaluate-classes'):
            path = directory/(label+'.log')
            lines = path.read_text(errors='replace').splitlines() if path.exists() else []
            error = next((l for l in reversed(lines) if any(s in l for s in ('error:','Error','requires','no admitted'))),'')
            output.append(dict(record='cpu_attempt',model=config['model'],batch=1,operation=label,
                               status='stopped' if error else 'inspect_raw',reason=error[:500],
                               evidence='verified; corrected seven-class evaluate input only'))
        classmap = {r['op']:int(r['gemm']) for r in tsv(directory/'classes.tsv')}
        plan = read(RUN/f"build/reference2/{config['model']}-B1/decode.so.plan.json")
        model = read(ROOT/config['model_dir']/'config.json')
        h=model['hidden_size']; d=model.get('head_dim',h//model['num_attention_heads'])
        qkv=(model['num_attention_heads']+2*model['num_key_value_heads'])*d
        # Existing evaluate outputs may contain alternatives: retain each key,
        # rather than silently labeling an arbitrary candidate as the final one.
        for path in directory.glob('evaluate-classes.cu.*.flow_spaces.tsv'):
            metrics_path = Path(str(path).replace('.flow_spaces.tsv','.metrics.tsv'))
            metrics = list(tsv(metrics_path)) if metrics_path.exists() else []
            for item in tsv(path):
                name=item['name']; kind=item['category']
                if name in classmap:
                    g=plan['gemms'][classmap[name]]
                    n,k=g.get('n'),g.get('k')
                    # Dimensions live in the exported class map, not geometry.
                    dims=next(r for r in tsv(directory/'classes.tsv') if r['op']==name)
                    n,k=int(dims['n']),int(dims['k'])
                    kind=('lm_head' if n==model['vocab_size'] else 'qkv' if n==qkv and k==h else
                          'gate_up' if n==2*model['intermediate_size'] else 'down' if k==model['intermediate_size'] else 'o')
                output.append(dict(record='existing_stage_prediction',model=config['model'],batch=1,
                    operation='evaluate-classes',candidate=path.name.split('.cu.')[1].split('.')[0],
                    candidate_key=metrics[0]['key'] if metrics else '',stage=item['stage'],kind=kind,
                    predicted_envelope_ns=float(item['last_end'])-float(item['first_start']),
                    fixed_sum_ns=item['fixed_sum'],mainloop_sum_ns=item['mainloop_sum'],
                    publication_sum_ns=item['publication_sum'],wait_sum_ns=item['wait_sum'],
                    evidence='verified existing export; execution differs from L1-loop; candidate identity must be checked'))
    for job, raw, gpu in records():
        if not (raw/'job/command.json').exists() or 'solver' not in job:
            continue
        row = dict(record='guarded_solver_attempt',job=job,gpu=gpu,status='compiled',
                   output=str((raw/'job').relative_to(ROOT)),evidence='verified guarded solver compilation; validation recorded separately')
        path = raw/'job/solver.so.plan.json'
        if path.exists():
            plan = read(path)
            row.update(model=plan['model'].rsplit('-B',1)[0], batch=1,
                grid=plan['grid'], residency=plan['residency'], pg=plan['pg'],
                attention_impl=plan['attention_impl'], Ec=plan['attention_kv_block'],
                Rq=plan['attention_query_rows'], handoff=plan['handoff'])
            selected = list(tsv(raw/'job/solver.so.resources.tsv'))
            if selected:
                row.update(predicted_step_ms=float(selected[0]['flow_ns'])/1e6,
                           simulated_step_ms=float(selected[0]['simulated_ns'])/1e6,
                           selected_key=selected[0]['key'])
        output.append(row)
    validations = solver_validations(paired)
    for row in validations:
        source = next((r for r in output if r.get('job')==row['arm'].replace('S-','solver-',1)+'-f0'),None)
        if source and 'predicted_step_ms' in source:
            row['predicted_step_ms'] = source['predicted_step_ms']
            if row.get('candidate_ms'):
                row['prediction_error_pct'] = (row['predicted_step_ms']/row['candidate_ms']-1)*100
    return output + validations
