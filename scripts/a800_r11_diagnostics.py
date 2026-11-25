#!/usr/bin/env python3
"""Join guarded stage probes with same-allocation R11 timing decisions."""
import argparse
from collections import defaultdict
from pathlib import Path

from a100_baselines import sha
from a100_results import write_csv
from a800_r10_collect_round import guarded_stats
from a800_r11_gate import read
from a800_r11_stages import stage_rows
from tilemega.bench.scheduler import write_json


def trace_cases(scope):
    states = {}
    for line in (scope/'accounting.tsv').read_text().splitlines():
        child, allocation, state, _ = line.split('|')
        parent, _, index = child.partition('_')
        if index.isdecimal() and state.split()[0].rstrip('+') in ('COMPLETED', 'FAILED'):
            states[parent, int(index)] = allocation
    for wave in sorted(scope.glob('wave-*.rounds.json')):
        tag = wave.name.removesuffix('.rounds.json')
        receipt = wave.with_name(f'{tag}.id')
        if not receipt.exists():
            continue
        parent = receipt.read_text().strip()
        for index, case in enumerate(read(wave)['cells']):
            job = states.get((parent, index))
            if job is None:
                continue
            if sha(case['plan']) != case['sha256']:
                raise ValueError('stage trace allocation uses a changed arm set')
            entry = read(case['plan'])
            yield job, entry


def measurements(runs, job, spec):
    trace_root = runs/'optimization/decode-traces'/job
    roles_root = runs/'calibration/r2-supplement-service'/job
    arms, roles, rejected = {}, {}, {}
    for probe in spec['probes']:
        name = probe['arm']
        if probe.get('kind') == 'role' or not (trace_root/name/'ledger/run.json').is_file():
            continue
        parsed = argparse.Namespace(plan=Path(probe['trace_plan']), run=trace_root/name)
        try:
            rows = list(stage_rows(parsed))
        except ValueError as error:
            rejected[name] = str(error)
            continue
        arms[name] = {row['stage']: row for row in rows}
    devices = {row['gpu_uuid'] for arm in arms.values() for row in arm.values()}
    if len(devices)>1:
        raise ValueError('stage control and trial did not use the same A800')
    if not devices:
        return arms, roles, rejected
    gpu, = devices
    for probe in spec['probes']:
        if probe.get('kind') != 'role':
            continue
        identity = Path(probe['identity'])
        if sha(identity.with_name('probe')) != probe['binary_sha256']:
            raise ValueError('linked role probe changed')
        found = guarded_stats(roles_root/probe['arm'], {'identity_sha256':sha(identity)}, gpu)
        for row in found:
            if row['status'] != 'measured':
                rejected[row['stage']] = row['status']
        roles[probe['arm']] = {row['stage']: row for row in found if row['status']=='measured'}
    return arms, roles, rejected


def boundary_sass(scope, model, batch, label, probes):
    fields = dict(control_binary_global_add_sites='', trial_binary_global_add_sites='',
                  control_sass_sha256='', trial_sass_sha256='')
    if label != 'whole-output-groups':
        return fields
    root = scope.parent/'boundary-sass'
    path = root/f'{model}-B{batch}.json'
    if not path.exists() and batch == 1:
        path = root/f'{model}.json'
    if not path.exists():
        return fields
    proof = read(path)
    source = {row['arm']: read(row['trace_plan'])['original_sha256']
              for row in probes if row['arm'] in ('control', label)}
    if any(source.get(arm) != proof[arm]['binary_sha256'] for arm in ('control', label)):
        return fields
    return dict(control_binary_global_add_sites=proof['control']['add_sites'],
                trial_binary_global_add_sites=proof[label]['add_sites'],
                control_sass_sha256=proof['control']['sass_sha256'],
                trial_sass_sha256=proof[label]['sass_sha256'])


def boundary_sync(scope, model, batch, label, probes):
    fields = dict(sync_proof_job='', sync_process_pass_rate='', sync_proof_sha256='')
    if (model, batch, label) != ('olmoe_1b_7b', 4, 'whole-output-groups'):
        return fields
    root = scope.parent/'boundary-sync'
    ticket = root/'olmoe-b4.id'
    if not ticket.is_file():
        return fields
    job = ticket.read_text().strip()
    path = root/f'{job}.result.json'
    if not path.is_file():
        return fields
    proof = read(path)
    if len(proof['data']) != 5:
        return fields
    [probe] = [row for row in probes if row['arm'] == label]
    if (proof['job'] != int(job) or proof['required'] != 5
            or proof['plan_sha256'] != sha(root/'olmoe-b4.plan.json')
            or proof['binary_sha256'] != read(probe['trace_plan'])['original_sha256']
            or any(sha(row['receipt']) != row['receipt_sha256'] for row in proof['data'] if row['pass_'])):
        raise ValueError('OLMoE sync evidence no longer matches the measured whole-output body')
    return dict(sync_proof_job=job, sync_process_pass_rate=f'{proof["passes"]}/5', sync_proof_sha256=sha(path))


def collect(runs, scope, timings, out):
    tables, retry = defaultdict(list), []
    for job, spec in trace_cases(scope):
        if spec.get('scope')=='final-release':
            continue
        key = spec['model'], spec['batch']
        arms, roles, rejected = measurements(runs, job, spec)
        retry.extend(dict(job=job, model=key[0], batch=key[1], arm=arm, reason=reason)
                     for arm,reason in rejected.items())
        for label, stages in arms.items():
            if key[0] == 'qwen3_30b_a3b':
                [binding] = [probe for probe in spec['probes'] if probe['arm']==label]
                performance = binding.get('timing', timings.get(f'{key[0]}/B{key[1]}/{label}', {}))
                if 'timing' in binding and (performance['job'] != spec['timing_job'] or
                        performance['decode_sha256'] != read(binding['trace_plan'])['original_sha256']):
                    raise ValueError('Qwen stage measurements and TPOT refer to different arms')
                for stage, value in stages.items():
                    found = [row for case, row in roles.get(f'role-{label}', {}).items()
                             if f'-{stage}-' in case]
                    ratio = found[0]['consumer_wait_fraction'] if len(found)==1 else ''
                    tables['qwen3'].append(dict(value, arm=label, job=job, consumer_wait_fraction=ratio,
                        timing_job=performance.get('job', ''), tpot_ms=performance.get('tpot_ms', ''),
                        boundaries_per_step=value['sampled_boundaries']/value['sampled_steps'],
                        same_stage_overlap_sms=value['dual_active_sms'],
                        cross_stage_overlap_sms=max(value['overlaps_previous_sms'],value['overlaps_next_sms'])))
                continue
            if label == 'control' or 'control' not in arms:
                continue
            sass = boundary_sass(scope, *key, label, spec['probes'])
            sync = boundary_sync(scope, *key, label, spec['probes'])
            for stage in ('qkv', 'o', 'down'):
                if stage not in stages or stage not in arms['control']:
                    continue
                before, after = arms['control'][stage], stages[stage]
                names = ('grid_barrier_fraction','stage_span_ns','critical_barrier_ns','effective_gbps')
                values = {f'{prefix}_{name}': row[name] for prefix,row in (('control',before),('trial',after))
                          for name in names}
                performance = timings.get(f'{key[0]}/B{key[1]}/{label}', {})
                tables['boundary'].append(dict(model=key[0], batch=key[1], arm=label, stage=stage,
                    trace_job=job, timing_job=performance.get('job',''), gpu_uuid=after['gpu_uuid'],
                    boundaries_per_step=after['sampled_boundaries']/after['sampled_steps'],
                    saved_ns_per_boundary=before['critical_barrier_ns']-after['critical_barrier_ns'],
                    predicted_saving_ns=320, tpot_ms=performance.get('tpot_ms',''),
                    paired_improvement_pct=performance.get('improvement_pct',''),
                    fresh_pass_rate=performance.get('pass_rate',''),
                    outcome=performance.get('status',''), evidence=after['trace_sha256'], **values, **sass, **sync))
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(out/f'A800_R11_{name}.csv', rows)
    if retry:
        write_json(out/'diagnostic-retries.json', retry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('runs', 'scope', 'timings', 'out'):
        parser.add_argument(f'--{field}', type=Path, required=True)
    args = parser.parse_args()
    collect(args.runs, args.scope, read(args.timings), args.out)


if __name__ == '__main__':
    main()
