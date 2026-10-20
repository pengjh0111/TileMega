#!/usr/bin/env python3
"""Collect SM90D1 tables from completed guards; never accept failed paths."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics as st

from sm90d1_common import ROOT, RUN, read, sha, write
import collect_sm90d1_diagnostics as diagnostic


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def completed():
    rows = []
    seen = set()
    for state in sorted((RUN / 'scheduler').glob('gpu*/state.json')):
        for job, record in read(state).items():
            if record['status'] != 'done' or not record.get('out'):
                continue
            raw = Path(record['out'])
            if str(raw) in seen or not (raw / 'guard_result.json').exists():
                continue
            seen.add(str(raw))
            guard = read(raw / 'guard_result.json')
            if guard['code'] != 0:
                continue
            selected = raw / 'job/selected-config.json'
            if not selected.exists():
                continue
            selection = read(selected)
            config = read(selection['config'])
            for path in (raw / 'job/measurements').glob('B*/measurements.json'):
                value = read(path)
                if value['max_tokens'] != 1024 or value['prompt_length'] != 64:
                    raise ValueError('wrong workload: '+str(path))
                if value['model_config_sha256'] != config['model_config_sha256']:
                    raise ValueError('model config changed: '+str(path))
                if value['configuration']['decode_sha256'] != selection['binary_sha256']:
                    raise ValueError('measured binary differs from selection')
                for generation in value['generation_runs']:
                    tokens = read(path.parent / generation['tokens_file'])
                    if len(tokens) != config['batch'] or any(len(row) != generation['N'] for row in tokens):
                        raise ValueError('wrong generated token count')
                if selection['stage'] == 'measure':
                    identity = value['configuration']['identities']['decode']
                    if identity['trace']:
                        raise ValueError('trace binary entered TPOT')
                    samples = [r for r in value['generation_runs'] if not r['warmup'] and r['N'] == 1024]
                    if len(samples) != 3:
                        raise ValueError('TPOT process did not have three repetitions')
                rows.append(dict(job=job, raw=raw, selection=selection, config=config,
                    measurement=value, tpot_ms=value['tpot_seconds']*1000,
                    gpu=int(value['gpu_index']), pid=value['pid'], round=selection['round']))
    return rows


def references(samples, select=True):
    historical = {(row['model'], int(row['batch'])): row for row in
                  csv.DictReader((ROOT / 'docs/results/SM90R1_final_llm.csv').open())}
    grouped = defaultdict(list)
    for row in samples:
        if row['job'].removeprefix('resume7-').startswith(('ref-', 'resolution-llama3_1_8b-B1-')) and row['selection']['stage'] == 'measure':
            c = row['config']
            grouped[c['model'], c['batch'], row['gpu']].append(row)
    paired_controls = defaultdict(list)
    for row in samples:
        if '-reference-f' in row['job'] and row['selection']['stage'] == 'measure':
            c = row['config']
            paired_controls[c['model'], c['batch'], row['gpu'],
                            row['job'].split('-reference-f')[0]].append(row)
    # Reuse the first complete control series only when a dedicated reference
    # is absent. The canonical binary hashes prevent variant timing leaking in.
    bindings = read(ROOT / 'configs/serving/sm90d1/reference/artifacts/index.json')['bindings']
    hashes = {(Path(r['config']).stem, r['phase']): r['binary_sha256'] for r in bindings}
    for (model, batch, gpu, prefix), group in sorted(paired_controls.items(),
            key=lambda item: min(r['pid'] for r in item[1])):
        if (model, batch, gpu) in grouped or {r['round'] for r in group} != {0, 1, 2} or len(group) != 3:
            continue
        stem = Path(group[0]['selection']['config']).stem
        if all(r['selection']['binary_sha256'] == hashes[stem, 'decode'] and
               r['selection']['prefill_sha256'] == hashes[stem, 'prefill'] and
               r['selection']['mode'] == 'L1' and r['selection']['loop'] == 1 for r in group):
            grouped[model, batch, gpu] = group
    rows = []
    for (model, batch, gpu), group in sorted(grouped.items()):
        initial = sorted(group, key=lambda r: r['round'])[:3]
        if {r['round'] for r in initial} != {0, 1, 2}:
            continue
        values = [r['tpot_ms'] for r in initial]
        median = st.median(values)
        old = historical[model, batch]
        old_range = float(old['tpot_ms_range_pct'])
        new_range = (max(values)-min(values))/median*100
        rows.append(dict(model=model, batch=batch, gpu=gpu, r1_gpus=old['gpus'], fresh_processes=3,
            status='measured', source_jobs='|'.join(r['job'] for r in initial),
            provenance='paired canonical controls' if '-reference-f' in initial[0]['job'] else 'dedicated reference',
            tpot_ms=median, process_range_pct=new_range, sm90r1_tpot_ms=old['tpot_ms'],
            difference_pct=(median/float(old['tpot_ms'])-1)*100,
            indistinguishable_pct=max(.5, old_range, new_range),
            decode_sha256=initial[0]['selection']['binary_sha256'],
            prefill_sha256=initial[0]['selection']['prefill_sha256'],
            evidence='verified; provisional_container_pid_namespace'))
    if not select:
        return rows
    selected = {}
    for row in rows:
        key = row['model'], row['batch']
        preference = str(row['gpu']) in row['r1_gpus'].split('|')
        if key not in selected or preference:
            selected[key] = row
    return list(selected.values())


def reference_statuses(measured):
    rows = list(measured)
    audit = RUN/'reference-drift-audit.json'
    inspections = {r['config']:r for r in read(audit)} if audit.exists() else {}
    for row in rows:
        inspection = inspections.get(f"{row['model']}_b{row['batch']}",{})
        for field in ('status','original_source_head','gemm_geometry_identical',
                      'compile_defines_identical','l1_loop_instruction_encoding_identical'):
            if field in inspection:
                row['drift_audit_'+field] = inspection[field]
    present = {(r['model'], r['batch']) for r in rows}
    historical = {(r['model'], int(r['batch'])): r for r in
                  csv.DictReader((ROOT / 'docs/results/SM90R1_final_llm.csv').open())}
    for path in sorted((ROOT / 'configs/serving/sm90d1/reference').glob('*.json')):
        config = read(path)
        key = config['model'], config['batch']
        if key not in present:
            old = historical[key]
            rows.append(dict(model=key[0], batch=key[1], status='not_measured',
                fresh_processes=0, sm90r1_tpot_ms=old['tpot_ms'],
                evidence='unmeasured; inspect scheduler receipts for pending or unsuccessful jobs'))
    return sorted(rows, key=lambda r: (r['model'], r['batch']))


def resolution(samples):
    rows = []
    for prefix, model in [('ref-qwen3_0_6b-B1-', 'qwen3_0_6b'),
                           ('resolution-llama3_1_8b-B1-', 'llama3_1_8b')]:
        group = [r for r in samples if r['job'].startswith(prefix) and r['selection']['stage'] == 'measure']
        if len(group) != 5:
            continue
        values = [r['tpot_ms'] for r in group]
        if len({r['gpu'] for r in group}) != 1 or len({r['pid'] for r in group}) != 5:
            raise ValueError('resolution requires five distinct processes on one GPU')
        rows.append(dict(kind='fresh_process', model=model, batch=1, gpu=group[0]['gpu'],
            processes=5, median_ms=st.median(values), minimum_ms=min(values), maximum_ms=max(values),
            range_pct=(max(values)-min(values))/st.median(values)*100))
    for model in ('qwen3_0_6b', 'llama3_1_8b'):
        for phase in ('prefill','decode'):
            original = RUN / f'build/reference2/{model}-B1/{phase}.so'
            repeated = RUN / f'build/repeat2/{model}-B1/{phase}.so'
            if repeated.exists():
                rows.append(dict(kind='build_hash', model=model, batch=1, phase=phase,
                    original_sha256=sha(original), repeated_sha256=sha(repeated),
                    identical=sha(original)==sha(repeated)))
    return rows


def pairs(samples):
    groups = defaultdict(list)
    for row in samples:
        if row['selection']['stage'] != 'measure' or '-reference-f' not in row['job']:
            continue
        prefix = row['job'].split('-reference-f')[0]
        groups[prefix, row['gpu']].append(row)
    rows = []
    for (prefix, gpu), controls in sorted(groups.items()):
        candidates = [r for r in samples if r['job'].startswith(prefix+'-') and
                      r['job'] not in {x['job'] for x in controls} and
                      r['selection']['stage'] == 'measure' and r['gpu'] == gpu]
        if len(controls) != 3 or len(candidates) != 3:
            continue
        if {r['round'] for r in controls} != {0,1,2} or {r['round'] for r in candidates} != {0,1,2}:
            raise ValueError('paired rounds are incomplete')
        if len({r['pid'] for r in controls+candidates}) != 6:
            raise ValueError('paired measurements must use six distinct processes')
        before, after = [r['tpot_ms'] for r in controls], [r['tpot_ms'] for r in candidates]
        c = controls[0]['config']
        rows.append(dict(arm=prefix, model=c['model'], batch=c['batch'], gpu=gpu,
            fresh_processes_per_arm=3, reference_ms=st.median(before), candidate_ms=st.median(after),
            difference_pct=(st.median(after)/st.median(before)-1)*100,
            reference_range_pct=(max(before)-min(before))/st.median(before)*100,
            candidate_range_pct=(max(after)-min(after))/st.median(after)*100,
            evidence='verified paired TPOT; provisional_container_pid_namespace'))
    return rows


def ledger_tables():
    rows = []
    for state in (RUN / 'scheduler').glob('gpu*/state.json'):
        for job, record in read(state).items():
            if record['status'] != 'done' or not job.endswith('-analyze'):
                continue
            trace_job = job.removesuffix('-analyze')
            trace_record = read(state).get(trace_job)
            if not trace_record or trace_record['status'] != 'done':
                continue
            raw = Path(trace_record['out'])
            if read(raw / 'guard_result.json')['code']:
                continue
            selected = read(raw / 'job/selected-config.json')
            config = read(selected['config'])
            for path in (raw / 'job/trace').glob('past*/ledger.json'):
                data = read(path)
                grouped = defaultdict(list)
                for r in data['stages']:
                    grouped[r['iteration'], r['kind']].append(r)
                categories = defaultdict(list)
                for (iteration, kind), values in grouped.items():
                    aggregate = dict(kind=kind, iteration=iteration)
                    for key in ('envelope_ns','bytes','floor_ns','body_mean_ns','body_critical_ns',
                                'start_ns','first_block_ns','gaps_ns','release_ns','skew_ns',
                                'inefficiency_ns','imbalance_ns','residual_ns'):
                        aggregate[key] = sum(r[key] for r in values) if all(r[key] is not None for r in values) else None
                    aggregate.update(tasks=sum(r['tasks'] for r in values),
                        active_ctas=st.median(r['active_ctas'] for r in values),
                        max_tasks_per_cta=max(r['max_tasks_per_cta'] for r in values),
                        profile_complete=all(r['profile_complete'] for r in values))
                    categories[kind].append(aggregate)
                for kind, values in categories.items():
                    row = dict(job=trace_job, model=config['model'], batch=config['batch'],
                               past=data['trace']['past'], kind=kind, steps=len(values))
                    for key in values[0]:
                        if key not in ('kind', 'iteration'):
                            row[key] = st.median(v[key] for v in values) if all(v[key] is not None for v in values) else None
                    row['event_step_ms'] = data['trace']['mean_step_ms']
                    row['trace_step_ms'] = st.median(r['step_ns'] for r in data['steps'])/1e6
                    rows.append(row)
    return rows


def primitives():
    groups = defaultdict(list)
    for state in (RUN / 'scheduler').glob('gpu*/state.json'):
        for job, record in read(state).items():
            if record['status'] != 'done' or not record.get('out'):
                continue
            raw = Path(record['out'])
            path = raw / 'job/primitives.csv'
            if not path.exists() or read(raw / 'guard_result.json')['code']:
                continue
            for row in csv.DictReader(path.open()):
                groups[row['name'], row['grid_ctas'], row['bytes'], row['unit'], job.startswith('memory-')].append(row)
    return [dict(name=k[0], grid_ctas=k[1], bytes=k[2], unit=k[3], extended=k[4],
                 fresh_processes=len(v), median=st.median(float(r['median']) for r in v),
                 median_process_p99=st.median(float(r['p99']) for r in v),
                 evidence='verified diagnostic; no target coefficients changed') for k,v in groups.items()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=RUN / 'collected')
    a = p.parse_args()
    samples = completed()
    reference = references(samples)
    csv_write(a.out / 'SM90D1_reference.csv', reference_statuses(reference))
    paired = pairs(samples)
    resolved = resolution(samples)+[dict(r,kind='paired_build_or_prefix') for r in paired if r['arm'].startswith(('build-repeat','prefix-'))]
    for row in samples:
        if row['selection']['stage']=='overhead':
            ref=next((r for r in reference if r['model']==row['config']['model'] and r['batch']==row['config']['batch']),None)
            resolved.append(dict(kind='trace_diagnostic_level',job=row['job'],model=row['config']['model'],
                batch=row['config']['batch'],gpu=row['gpu'],instrumented_request_step_ms=row['tpot_ms'],
                overhead_ms=row['tpot_ms']-ref['tpot_ms'] if ref else None,
                evidence='verified diagnostic only; one process; reference drift not isolated'))
    csv_write(a.out / 'SM90D1_resolution.csv', resolved)
    traces = diagnostic.traces()
    ledgers = diagnostic.ledger_tables(traces,references(samples,select=False))
    write(a.out / 'all-stage-categories.json',ledgers)
    arms = [r for r in paired if not r['arm'].startswith(('G1','G2','S-','build-repeat','prefix-'))]
    arms += [r for r in diagnostic.arm_details(traces,ledgers) if r['arm'].startswith(('P','LA','EV'))]
    arms += diagnostic.rejected_mechanisms()
    csv_write(a.out / 'SM90D1_arms.csv', arms)
    csv_write(a.out / 'SM90D1_geometry_probes.csv', [r for r in paired if r['arm'].startswith(('G1','G2'))]+diagnostic.rejected_geometry())
    csv_write(a.out / 'SM90D1_stage_ledger.csv', [r for r in ledgers if r['tag'] in ('full2','stage2','ready-full','ready-stage')])
    csv_write(a.out / 'SM90D1_primitives.csv', primitives())
    csv_write(a.out / 'SM90D1_attention.csv', diagnostic.attention(traces,ledgers))
    csv_write(a.out / 'SM90D1_resources.csv', diagnostic.resources(traces))
    solver = diagnostic.solver(ledgers, paired)
    write(a.out / 'solver-stage-predictions.json',solver)
    groups=defaultdict(list)
    for row in solver:
        if row['record']=='existing_stage_prediction':
            groups[row['model'],row['candidate'],row['kind']].append(row)
    compact=[r for r in solver if r['record']!='existing_stage_prediction']
    for (model,candidate,kind),values in groups.items():
        row=dict(values[0]); row['record']='existing_category_prediction';row['stages']=len(values);row.pop('stage',None)
        for key in ('predicted_envelope_ns','fixed_sum_ns','mainloop_sum_ns','publication_sum_ns','wait_sum_ns'):
            row[key]=sum(float(v[key]) for v in values)
        measured_kind = {'fused_attention':'attention','attention_merge':'merge',
                         'argmax_reduce':'argmax'}.get(kind,kind)
        measured = [r for r in ledgers if r['tag']=='full2' and r['model']==model and
                    r['batch']==1 and r['past']==575 and (r['kind']==measured_kind or
                    (kind=='combine' and r['kind'].endswith('-combine')))]
        row['measured_kind'] = measured_kind
        row['measured_envelope_ns'] = sum(r['envelope_ns'] for r in measured) if measured else None
        row['envelope_difference_pct'] = ((row['predicted_envelope_ns']/row['measured_envelope_ns']-1)*100
                                          if row['measured_envelope_ns'] else None)
        compact.append(row)
    csv_write(a.out / 'SM90D1_solver.csv',compact)
    write(a.out / 'collection.json', dict(accepted_processes=len(samples), reference_cells=len(references(samples)),
        completed_pairs=len(paired), gpu_budget=read(RUN / 'gpu-budget.json')))


if __name__ == '__main__':
    main()
