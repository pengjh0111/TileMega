#!/usr/bin/env python3
"""Watch required SM90X1 units, validate the delivery and publish tilemega."""
import argparse
import contextlib
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time

from sm90d1_common import ROOT, PYTHON, read, sha, write
from sm90x1 import cells
from sm90x1_campaign import live_start
from sm90x1_finalize import FILES, TERMINAL
from sm90x1_hardware import audit_trace_unit, audit_unit
from sm90x1_experiment import decomposition, families
import sm90x1_waiver as waiver

RUN = ROOT / 'runs/sm90x1'
STATE = RUN / 'completion-watch-state.json'
WORKERS = ('campaign-process.json', 'ready-phase2-process.json',
           'prepare-traces-phase2-process.json', 'post-phase2-process.json',
           'parallel-native-process.json', 'phase2-pipeline-process.json')
FROZEN = ('include', 'lib', 'tools', 'python', 'configs/targets/sm_90_h800.json',
          'configs/serving/sm90x1/cells.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


@contextlib.contextmanager
def monitor_lock(path):
    with Path(path).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def status(stage, **details):
    write(STATE, dict(status=stage, pid=os.getpid(), at=time.time(), **details))
    print(json.dumps(dict(status=stage, at=time.time(), **details)), flush=True)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def command(argv, log, timeout=1800):
    with Path(log).open('a') as output:
        subprocess.run(argv, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                       stdout=output, stderr=subprocess.STDOUT, check=True, timeout=timeout)


def directory(cell):
    return RUN / 'cells' / cell['cell'] / ('phase1' if cell['pilot'] else 'phase2')


def progress():
    result = dict(required_cells=14, timed_cells=0, trace_coordinates=0, total_traces=0, remaining=[])
    for cell in cells('all'):
        path = directory(cell)
        plan = read(path/'build-plan.json') if (path/'build-plan.json').exists() else []
        traces = sum((path/e['id']/'trace/unit.json').exists() and
                     read(path/e['id']/'trace/hardware-audit.json').get('passed', False)
                     for e in plan if (path/e['id']/'trace/hardware-audit.json').exists())
        timed = (path/'time/unit.json').exists() and (path/'time/hardware-audit.json').exists() and \
            read(path/'time/hardware-audit.json').get('passed', False)
        result['trace_coordinates'] += traces
        result['total_traces'] += len(plan)
        result['timed_cells'] += int(timed)
        if not timed:
            detail=dict(cell=cell['cell'], traces=traces, total_traces=len(plan))
            parallel=path/'time/parallel/progress.json'
            if parallel.exists():detail['timing_processes']=read(parallel)
            result['remaining'].append(detail)
    return result


def active_workers():
    names = list(WORKERS)
    if (RUN/'extra-worker-metadata.json').exists():
        names += read(RUN/'extra-worker-metadata.json')
    pids = {read(RUN/name)['pid'] for name in names if (RUN/name).exists()}
    handoff = RUN/'ready-queue-policy-handoff.json'
    if handoff.exists():
        pids.update(read(handoff)['continuing_children'])
    return sorted(pid for pid in pids if pid != os.getpid() and live_start(pid) is not None)


def accepted_counts(value):
    return value['timed_cells'] == value['required_cells'] == 14 and \
        value['trace_coordinates'] == value['total_traces'] == 165


def check_setup():
    manifest = read(RUN/'pending-build-tooling.json')
    require(git('branch', '--show-current') == 'tilemega', 'expected branch tilemega')
    require(not git('diff', manifest['measurement_head'], '--name-only', '--', *FROZEN),
            'frozen runtime, compiler, target or cell list changed')
    require(not git('diff', '--cached', '--name-only'), 'pre-existing staged edits')
    for name, digest in manifest['files'].items():
        require(sha(ROOT/name) == digest, 'pending tooling changed: '+name)
    paths = set(git('diff', '--name-only').splitlines()) | \
        set(git('ls-files', '--others', '--exclude-standard').splitlines())
    require(not paths-set(FILES)-set(manifest['files']), 'unexpected workspace edits')
    command(['git', 'diff', '--check'], RUN/'completion-checks.log')
    require(len(cells('all')) == 14, 'required cell list changed')
    return manifest


def check_binary(binding, expected_head, cache):
    binary = Path(binding['decode'])
    digest = cache.setdefault(str(binary), None)
    if digest is None:
        digest = cache[str(binary)] = sha(binary)
    identity = read(str(binary)+'.identity.json')
    require(binding['status'] == 0 and binding['host_abi_status'] == 0, 'native build or ABI failed')
    require(not binding['trace'] and not identity['trace'], 'trace binary in reportable arms')
    require(binding['head'] == identity['source']['head'] == expected_head, 'mixed measurement HEADs')
    require(binding['decode_sha256'] == identity['binary_sha256'] == digest, 'native binary hash changed')
    require(sha(binding['coordinates']) == binding['coordinates_sha256'], 'coordinate hash changed')


def check_correctness(result):
    require(result.get('pass') and result.get('c1_pass') and result.get('c2_pass'), 'P-CORR failed')
    spec = result['spec']
    require(spec['kind'] == 'corr' and spec['correctness'] ==
            dict(c1=256, c2=True, reference_processes=1), 'P-CORR protocol changed')
    require(result['fresh_processes'] == result['processes_token_equal'] == len(result['rows']) == 5,
            'P-CORR needs five complete fresh processes')
    require(result['rows'][0].get('c1', {}).get('pass'), 'first process C1 missing')
    require(all(row.get('c2', {}).get('pass') for row in result['rows']), 'per-process C2 missing')


def check_timing(result, cell):
    spec = result['spec']; labels = ['W', 'L', 'Wstar']; n = cell['processes']
    expected = [label for i in range(n) for label in (labels if i % 2 == 0 else labels[::-1])]
    require(spec['kind'] == 'sel' and spec['order'] == expected, 'interleaved selection order changed')
    require([r['arm'] for r in result['rows']] == expected, 'logical process labels differ from spec')
    require(set(result['arms']) == set(labels), 'required timing arm missing')
    for arm in result['arms'].values():
        require(arm['processes'] == n, 'wrong fresh timing process count')
        require(math.isfinite(arm['tpot_ms']) and arm['tpot_ms'] > 0 and
                math.isfinite(arm['range_pct']) and arm['range_pct'] >= 0, 'invalid arm timing')


def check_structure(coordinates, reference):
    for key in ('residency', 'cluster', 'group_wait', 'emb', 'expert_fuse', 'layer_loop', 'weight_stream', 'tp'):
        # Accepted schema-5 W files predate the explicit false fusion flag.
        default = False if key == 'expert_fuse' else None
        require(coordinates.get(key, default) == reference.get(key, default), 'arm changed W structure: '+key)


def validate_units(expected_head, selected_cells=None):
    all_cells = cells('all') if selected_cells is None else selected_cells
    cache = {}; traces = 0; selected = 0; corr_count = 0; waived_cells=set()
    phase0 = read(RUN/'phase0.json'); pilot = read(RUN/'pilot-gate.json')
    require(phase0['complete'] and not phase0['stop'], 'Phase 0 gate incomplete or failed')
    require(pilot['continue'] and pilot['max_builds'] <= 600 and pilot['gpu_hours'] <= 40,
            'pilot projection gate failed')
    from sm90r4_dispatch import allowed, POLICY
    allowed_gpus = set(allowed(read(POLICY)))
    for cell in all_cells:
        path = directory(cell); plan = read(path/'build-plan.json')
        require(sha(cell['prefill']) == cell['prefill_sha256'], 'recorded prefill changed')
        variants = read(path/'variants.json'); analysis = read(path/'analysis.json')
        require(len(analysis) == len(variants), 'candidate analysis dropped variants')
        for variant in analysis:
            if not variant['selected']:
                continue
            selected += 1
            require(variant['legal'] and variant['processes'] == 2, 'illegal or incomplete selected variant')
            require(math.isfinite(variant['own_us']) and variant['own_us'] > 0 and
                    math.isfinite(variant['step_us']) and variant['step_us'] > 0, 'invalid trace metric')
            require(math.isclose(variant['delta_coupled_us'],
                    variant['delta_step_us']-variant['delta_own_us'], abs_tol=1e-8), 'coupled delta mismatch')
        for entry in plan:
            artifact = path/entry['id']; binding = read(artifact/'binding.json')
            check_binary(binding, expected_head, cache)
            unit = read(artifact/'trace/unit.json'); identity = read(str(artifact/'trace.so')+'.identity.json')
            require(unit['head'] == expected_head and unit['processes'] == len(unit['runs']) == 2,
                    'trace fresh-process count or source HEAD differs')
            require(identity['trace'] and identity.get('performance_eligible') is False and
                    identity['binary_sha256'] == sha(artifact/'trace.so'), 'trace identity changed')
            require(unit['performance_eligible'] is False and unit['class'] == cell['class'] and
                    unit['gpu'] in allowed_gpus and audit_trace_unit(artifact/'trace')['passed'],
                    'trace guard, GPU or class rejected')
            for run in unit['runs']:
                require(read(Path(run)/'run.json')['decode_sha256'] == identity['binary_sha256'],
                        'trace child used another binary')
            traces += 1
        arms = read(path/'arm-plan.json')
        require({e['arm'] for e in arms} == {'W', 'L', 'Wstar'}, 'required arm plan missing')
        for entry in {e['id']:e for e in arms}.values():
            binding = read(path/entry['id']/'binding.json');check_binary(binding, expected_head, cache)
            coordinates = entry.get('normalized_coordinates', read(entry['coordinates']))
            check_structure(coordinates, cell['coordinates'])
            corr = path/entry['id']/'corr'
            if not waiver.passed(corr):
                require(bool(waiver.authorization(cell)), 'missing P-CORR without user authorization')
                marker=read(path/'correctness-waiver.json')
                require(marker['pcorr_passed'] is False and marker['performance_eligible'] is False,
                        'waiver must not claim correctness acceptance')
                require(marker['cell']==cell['cell'] and marker['authorization']==waiver.authorization(cell),
                        'waiver authorization differs from the recorded cell')
                waived_cells.add(cell['cell'])
                continue
            check_correctness(read(corr/'result.json'))
            protocol = read(corr/'unit.json')
            require(protocol['class'] == cell['class'] and protocol['gpu'] in allowed_gpus and
                    audit_unit(corr)['passed'], 'P-CORR GPU, class or guard rejected')
            corr_count += 1
        result = read(path/'time/result.json');check_timing(result, cell)
        if cell['cell'] in waived_cells:
            require(result['spec'].get('correctness_waiver') == waiver.authorization(cell) and
                    result['spec'].get('performance_eligible') is False, 'timing waiver provenance missing')
        protocol = read(path/'time/unit.json')
        if protocol.get('mode')=='parallel_processes':
            import sm90x1_parallel as parallel
            require(parallel.enabled(result['spec']) and protocol['class']=='trial' and
                    result['spec'].get('execution')==protocol['execution'], 'parallel timing authorization differs')
            actual_gpus={r['gpu'] for r in result['rows']}
            require(set(protocol['gpus'])==actual_gpus and actual_gpus<=allowed_gpus,
                    'parallel timing devices differ from actual rows')
            parallel.validate_processes(result['spec'],read(path/'time'/protocol['try']/'scheduler/state.json'))
        else:
            require(protocol['gpu'] in allowed_gpus,'timing GPU rejected')
        require(protocol['class'] == cell['class'] and audit_unit(path/'time')['passed'],
                'timing class or guard rejected')
        for entry in arms:
            binary = read(path/entry['id']/'binding.json')
            actual = result['spec']['arms'][entry['arm']]
            require(actual['decode_sha256'] == binary['decode_sha256'] and
                    actual['prefill_sha256'] == cell['prefill_sha256'] and
                    actual['root_commit'] == expected_head, 'timing provenance differs from accepted arm')
    if selected_cells is None:
        require(traces == 165 and selected == 233, 'frozen candidate coverage changed')
    return dict(cells=len(all_cells), trace_coordinates=traces, trace_processes=2*traces,
                selected_family_variants=selected, native_binaries=len(cache),
                correctness_units=corr_count, measurement_head=expected_head,
                certified_cells=len(all_cells)-len(waived_cells),correctness_waived_cells=sorted(waived_cells),
                scope='required' if selected_cells is None else 'preflight_subset')


def csv_rows(name):
    with (ROOT/'docs/results'/('SM90X1_'+name+'.csv')).open() as stream:
        return list(csv.DictReader(stream))


def check_figures(arms, candidates, normalized, scatter):
    measured = {(r['cell'], r['arm']):r for r in arms if r['status'] == 'measured'}
    require(len(normalized) == len(measured) and
            {(r['cell'], r['arm']) for r in normalized} == set(measured), 'normalized arm coverage differs')
    for row in normalized:
        arm = measured[row['cell'], row['arm']];reference = measured[row['cell'], 'Wstar']
        require(math.isclose(float(row['normalized_tpot']), float(arm['tpot_ms'])/float(reference['tpot_ms']),
                            rel_tol=1e-10), 'normalized figure value differs')
        require(row['class'] == arm['class'] and row['gpu'] == arm['gpu'], 'figure provenance differs')
    families = {r['family'] for r in scatter}
    require(len(families) == 2, 'scatter needs two illustrative families')
    key = lambda r: (r['cell'], r['family'], r['variant'])
    expected = {key(r):r for r in candidates if r['family'] in families and r['processes']}
    require(len(scatter) == len(expected) and {key(r) for r in scatter} == set(expected), 'scatter coverage differs')
    for row in scatter:
        candidate = expected[key(row)]
        require(all(row[field] == candidate[field] for field in
                    ('own_us', 'step_us', 'processes', 'gpu', 'class')), 'scatter value or provenance differs')


def check_aggregates(summary, all_cells, by_cell, waived_cells=None):
    for batch in ('1','16','overall'):
        selected=[c for c in all_cells if batch=='overall' or str(c['batch'])==batch]
        available=[c for c in selected if c['cell'] in by_cell];complete=len(available)==len(selected)
        metrics={r['metric']:r for r in summary if str(r['batch'])==batch and not r.get('family') and not r.get('cell')}
        for metric,reference in [('X','Wstar'),('Xprime','W')]:
            row=metrics[metric]
            require(int(row['count'])==len(available) and int(row['denominator'])==len(selected), 'ratio coverage differs')
            if not complete:
                require(row['value']==row['min']==row['max']=='' and row['status']=='pending', 'partial-cell ratio substituted')
                continue
            values=[float(by_cell[c['cell']]['L']['tpot_ms'])/float(by_cell[c['cell']][reference]['tpot_ms']) for c in available]
            for field,value in [('value',math.exp(sum(map(math.log,values))/len(values))),('min',min(values)),('max',max(values))]:
                require(math.isclose(float(row[field]),value,rel_tol=1e-10), 'paper ratio mismatch')
        for metric,reference in [('Y','Wstar'),('Yprime','W')]:
            row=metrics[metric];denominator=sum(len(families(c)) for c in selected)
            require(int(row['denominator'])==denominator, 'selection denominator differs')
            if not complete:
                require(row['value']==row['count']=='' and row['status']=='pending', 'partial-cell fraction substituted')
                continue
            count=0
            for cell in available:
                coords={e['arm']:read(e['coordinates']) for e in read(directory(cell)/'arm-plan.json')}
                count+=sum(decomposition(coords['L'],f)!=decomposition(coords[reference],f) for f in families(cell))
            require(int(row['count'])==count and math.isclose(float(row['value']),count/denominator,abs_tol=1e-12),
                    'paper selection fraction mismatch')
        if waived_cells is not None:
            expected=sorted({c['cell'] for c in selected} & set(waived_cells))
            for row in metrics.values():
                require(row['correctness_waived_cells']==';'.join(expected) and
                        row['performance_eligible']==str(not expected), 'supplemental waiver scope differs')


def validate_tables():
    all_cells = cells('all'); arms = csv_rows('arms'); summary = csv_rows('summary')
    uncertified=csv_rows('uncertified_arms');supplemental=csv_rows('summary_user_waived')
    require(len(arms) == 42 and {r['cell'] for r in arms} == {c['cell'] for c in all_cells},
            'arm table must retain every required cell')
    by_cell = {c['cell']: {r['arm']: r for r in arms if r['cell'] == c['cell']} for c in all_cells}
    strict={};waived_cells=set()
    for cell in all_cells:
        data = by_cell[cell['cell']]
        require(set(data) == {'W', 'L', 'Wstar'}, 'arm table labels missing')
        if all(r['status']=='measured' for r in data.values()):
            strict[cell['cell']]=data
            require(all(r['c1_pass']==r['c2_pass']=='True' for r in data.values()), 'uncertified reportable arm')
            require(all(r['correctness_status']=='passed' and r['performance_eligible']=='True'
                        for r in data.values()), 'strict eligibility labels differ')
        else:
            require(bool(waiver.authorization(cell)), 'uncertified cell lacks explicit waiver')
            require(all(r['status']=='correctness_waived' and r['tpot_ms']==r['c1_pass']==r['c2_pass']==''
                        for r in data.values()), 'uncertified timing entered strict arm table')
            data={r['arm']:r for r in uncertified if r['cell']==cell['cell']};by_cell[cell['cell']]=data
            require(set(data)=={'W','L','Wstar'} and all(r['status']=='measured_under_user_waiver' and
                    r['performance_eligible']=='False' and r['correctness_status']=='waived_by_user' and
                    r['c1_pass']==r['c2_pass']=='' for r in data.values()), 'uncertified timing labels differ')
            waived_cells.add(cell['cell'])
        raw=read(directory(cell)/'time/result.json');protocol=read(directory(cell)/'time/unit.json')
        for label,arm in data.items():
            require(arm['class'] == cell['class'] and int(arm['processes']) == cell['processes'],
                    'arm table protocol differs')
            measured=raw['arms'][label]
            require(all(math.isclose(float(arm[key]),measured[key],rel_tol=1e-10)
                        for key in ('tpot_ms','range_pct')), 'arm CSV differs from actual timing')
            gpu=';'.join(map(str,measured['gpus'])) if protocol.get('mode')=='parallel_processes' else str(protocol['gpu'])
            require(arm['gpu']==gpu, 'arm CSV devices differ from actual timing')
            if protocol.get('mode')=='parallel_processes':
                require(arm['execution_mode']=='parallel_processes','parallel CSV provenance missing')
            for ref, suffix in [('W', 'w'), ('Wstar', 'wstar')]:
                other = data[ref];change = 100*(float(arm['tpot_ms'])/float(other['tpot_ms'])-1)
                bound = max(0.5, 2*max(float(arm['range_pct']), float(other['range_pct'])))
                decision = 'indistinguishable' if abs(change) <= bound else 'faster' if change < 0 else 'slower'
                require(math.isclose(float(arm['change_vs_'+suffix+'_pct']), change, abs_tol=1e-8) and
                        arm['decision_vs_'+suffix] == decision, 'P-PAIR table decision differs')
            if cell['cell'] in waived_cells:
                require(math.isclose(float(arm['normalized_tpot']),
                        float(arm['tpot_ms'])/float(data['Wstar']['tpot_ms']), rel_tol=1e-10),
                        'supplemental normalization differs')
    require(len(uncertified)==3*len(waived_cells), 'extra or missing uncertified arms')
    check_aggregates(summary,all_cells,strict)
    check_aggregates(supplemental,all_cells,by_cell,waived_cells)
    candidates = csv_rows('candidates')
    require(len(candidates) == 272 and sum(r['selected'] == 'True' for r in candidates) == 233,
            'candidate table coverage differs')
    import sm90x1_analysis as stats
    numeric = [{**row, **{field:float(row[field]) for field in
                ('own_us', 'step_us', 'delta_own_us', 'delta_coupled_us')}}
               for row in candidates if row['processes']]
    for row in summary:
        if row['metric'] not in ('tau', 'tau_cell', 'coupled_share'):
            continue
        data = [r for r in numeric if r['family'] == row['family'] and
                (row['batch'] == 'overall' or next(c['batch'] for c in all_cells if c['cell'] == r['cell']) == int(row['batch'])) and
                (not row['cell'] or r['cell'] == row['cell'])]
        value = stats.coupled_share(data) if row['metric'] == 'coupled_share' else stats.kendall(data)
        require(int(row['count']) == len(data), 'diagnostic coverage differs')
        require(row['value'] == '' if value is None else
                math.isclose(float(row['value']), value, abs_tol=1e-10), 'tau or coupled-share value differs')
    check_figures(arms, candidates, csv_rows('figure_arms'), csv_rows('figure_scatter'))
    require(bool(csv_rows('stage_deltas')), 'stage deltas missing')
    require(all((ROOT/path).is_file() for path in FILES), 'required compact file missing')
    require('## Numbers for the paper' in (ROOT/'docs/results/SM90X1_h800.md').read_text(), 'paper section missing')
    return dict(paper=[r for r in summary if r['metric'] in ('X', 'Xprime', 'Y', 'Yprime')],
                supplemental_user_waived=supplemental,correctness_waived_cells=sorted(waived_cells),
                families=[r for r in summary if r['metric'] in ('tau', 'coupled_share')],
                improvements=[r for r in arms if r['arm'] in ('L', 'Wstar') and r['decision_vs_w'] == 'faster'],
                strongest_rank_reversals=read(RUN/'rank-reversals.json')[:2],
                open_items=(['P-CORR explicitly waived for '+c+'; excluded from strict paper aggregates.' for c in sorted(waived_cells)]+
                           ['Optional Phase 3 arms/cells were not run.',
                            'Host GPU PID ownership remains provisional in this container.',
                            'Earlier excluded exit-code-247 attempts have an unresolved cause.']))


def finalize_if_needed(manifest):
    finalizer = read(RUN/'finalizer-state.json') if (RUN/'finalizer-state.json').exists() else {}
    if finalizer.get('status') == 'finalized':
        return
    # A dead legacy finalizer can be resumed without duplicating completed commits.
    check_setup()
    for group in manifest['commit_groups']:
        paths = set(git('diff', '--name-only', '--', *group['files']).splitlines()) | \
            set(git('ls-files', '--others', '--exclude-standard', '--', *group['files']).splitlines())
        if paths:
            subprocess.check_call(['git', 'add', '--', *group['files']], cwd=ROOT)
            subprocess.check_call(['git', 'commit', '-m', group['message']], cwd=ROOT)
    command([PYTHON, str(ROOT/'scripts/sm90x1_finalize.py'), '--once'], RUN/'completion-checks.log')


def push(remote, branch):
    require(git('branch', '--show-current') == branch, 'branch changed before push')
    require(not git('status', '--porcelain'), 'uncommitted edits before push')
    subprocess.run(['git', 'fetch', remote, branch], cwd=ROOT, check=True, timeout=120)
    subprocess.run(['git', 'merge-base', '--is-ancestor', 'FETCH_HEAD', 'HEAD'], cwd=ROOT, check=True)
    commit = git('rev-parse', 'HEAD')
    subprocess.run(['git', 'push', remote, 'HEAD:refs/heads/'+branch], cwd=ROOT, check=True, timeout=180)
    confirmed = git('ls-remote', '--heads', remote, branch).split()[0]
    require(confirmed == commit, 'remote branch tip differs after push; retain receipt for review')
    return dict(commit=commit, remote=remote, branch=branch, confirmed_remote_head=confirmed)


def finish(args, manifest):
    status('validating_units')
    proof = validate_units(manifest['measurement_head'])
    status('checking_cpu_tests', proof=proof)
    command([PYTHON, '-m', 'unittest', 'discover', '-s', 'test/unit', '-p', 'test_sm90x1*.py'],
            RUN/'completion-cpu-tests.log')
    command(['ctest', '--test-dir', str(RUN/'compiler'), '-R', '^sm90_native_sweep$', '--output-on-failure'],
            RUN/'completion-sweep-test.log')
    finalize_if_needed(manifest)
    status('validating_tables', proof=proof)
    command([PYTHON, str(ROOT/'scripts/sm90x1.py'), 'tables', '--phase', '1', '--cell', 'all'],
            RUN/'completion-tables.log')
    summary = validate_tables()
    write(RUN/'completion-acceptance.json', dict(passed=True, strict_pcorr_complete=not proof['correctness_waived_cells'],
                                               proof=proof, summary=summary, at=time.time()))
    report = ROOT/'docs/results/SM90X1_h800.md'
    report.write_text(report.read_text()+
        '\n## Collection acceptance\n\n[verified] All 14 required cells completed non-trace final timing. '+
        str(proof['certified_cells'])+' cells passed complete P-CORR; user-waived cells: '+
        (', '.join(proof['correctness_waived_cells']) or 'none')+'. Uncertified timings and aggregates are supplemental. '
        'All 165 sweep coordinates have two accepted diagnostic trace processes (330 total). '
        'Final CPU tooling tests and the native-sweep C++ unit test passed. '
        'Arm binary hashes, source HEAD, frozen prefills, guard receipts, process counts, '
        'logical interleaved round labels, authorized parallel execution, paper aggregates and P-PAIR decisions were rechecked. '
        'Validation receipts and failed attempts remain under ignored runs/sm90x1/.\n\n'
        '[verified] Optional Phase 3 was not run. Host GPU PID ownership remains provisional; '
        'earlier excluded exit-code-247 attempts have an unresolved cause.\n')
    require(not git('diff', '--cached', '--name-only'), 'staged edits appeared during report generation')
    changed = set(git('diff', '--name-only').splitlines())
    require(changed <= set(FILES) and not git('ls-files', '--others', '--exclude-standard'),
            'unexpected edits after finalization')
    command(['git', 'diff', '--check'], RUN/'completion-checks.log')
    if changed:
        subprocess.check_call(['git', 'add', '--', *FILES], cwd=ROOT)
        message='docs: verify SM90X1 delivery with explicit waivers' if proof['correctness_waived_cells'] else \
            'docs: verify complete SM90X1 paper delivery'
        subprocess.check_call(['git', 'commit', '-m', message], cwd=ROOT)
    delivery = dict(status='organized', proof=proof, summary=summary, commit=git('rev-parse', 'HEAD'))
    write(RUN/'completion-summary.json', delivery)
    if args.push:
        for attempt in range(3):
            status('pushing', attempt=attempt+1, commit=delivery['commit'])
            try:
                delivery.update(push(args.remote, args.branch), status='pushed')
                break
            except (subprocess.SubprocessError, ValueError) as error:
                if attempt == 2:
                    raise
                status('push_retry', error=str(error), attempt=attempt+1)
                time.sleep(30)
    write(RUN/'completion-summary.json', delivery)
    status(delivery['status'], commit=delivery['commit'], summary_file=str(RUN/'completion-summary.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--push', action='store_true', help='publish after complete required-scope validation')
    parser.add_argument('--remote', default='origin');parser.add_argument('--branch', default='tilemega')
    parser.add_argument('--poll', type=float, default=30)
    parser.add_argument('--check-setup', action='store_true')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    require(args.branch == 'tilemega' and 1 <= args.poll <= 60, 'invalid branch or polling interval')
    RUN.mkdir(parents=True, exist_ok=True)
    with monitor_lock(RUN/'completion-watch.lock'):
        manifest = check_setup()
        if args.check_setup:
            print(json.dumps(dict(setup_passed=True, progress=progress())))
            return 0
        write(RUN/'completion-watch-process.json', dict(pid=os.getpid(), start=live_start(os.getpid()),
              source_sha256=sha(__file__), argv=list(os.sys.argv), at=time.time()))
        while True:
            campaign = read(RUN/'campaign-state.json');value = progress()
            if campaign['status'] in TERMINAL:
                require(campaign['status'] == 'complete' and accepted_counts(value),
                        'campaign ended before all required measurements passed: '+str(campaign))
                active = active_workers()
                finalizer_meta = RUN/'finalize-with-build-tooling-process.json'
                finalizer_pid = read(finalizer_meta)['pid'] if finalizer_meta.exists() else None
                if finalizer_pid and live_start(finalizer_pid) is not None:
                    active.append(finalizer_pid)
                if not active:
                    break
                status('waiting_workers', active=active, progress=value)
            else:
                require(live_start(read(RUN/'campaign-process.json')['pid']) is not None,
                        'campaign parent exited without terminal state')
                status('watching', campaign=campaign['status'], progress=value)
            if args.once:
                return 2
            time.sleep(args.poll)
        finish(args, manifest)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except BlockingIOError:
        print('another SM90X1 completion monitor holds the lock', flush=True)
        raise SystemExit(3)
    except Exception as error:
        status('needs_review', error=str(error))
        raise
