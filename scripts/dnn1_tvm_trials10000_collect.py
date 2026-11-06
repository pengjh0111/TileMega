#!/usr/bin/env python3
"""Prepare, release, run and collect an isolated 25-cell TVM fresh10000 epoch.

Prepare never schedules work. Release requires a committed/pushed terminal AIT25
CSV and explicit completion declaration. Wait only reads CPU files until that
release exists, then invokes run. All outputs stay in the new runs epoch.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import io
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / 'python')]
import dnn1_baselines as base
import dnn1_device as device
import dnn1_tvm_resume_finalize as finalizer
from dnn1_aitemplate_cell import AFFINITY
from dnn1_baseline_boundaries import idle_boundary, final_boundary
from dnn1_background_cards import main_reservations
from dnn1_dispatch import affinities
from dnn1_owned_controller import OwnedController, NONCE_KEY
import dnn1_tvm_trials10000_sample as experiment

need, read, save, sha = experiment.need, experiment.read, experiment.save, experiment.sha
CELLS = {(m, b) for m in base.MODELS for b in base.BATCHES}
NEW_MODELS = ('resnet34', 'vgg16', 'alexnet')
PRIORITY_MODELS = NEW_MODELS + ('resnet18', 'bert')
PROCESS_TIMEOUT = 18000
CELL_TIMEOUT = 108000
EXTRA_FIELDS = ['workload_id', 'experiment_profile', 'requested_trials', 'tuning_timeout_s',
    'committed_trials', 'successful_trials', 'failed_trials', 'attempted_builds',
    'build_errors', 'tune_wall_seconds', 'deadline_hit', 'tuning_receipt',
    'tuning_receipt_sha256', 'experiment_manifest_sha256']


def csv_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(base.FIELDS) + EXTRA_FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(stream.getvalue())
    temporary.replace(path)


def gpu_pool(value):
    gpus = [int(x) for x in value.split(',')]
    need(gpus and len(set(gpus)) == len(gpus) and set(gpus) <= set(range(7)),
         'distinct assigned indices0–6 only; index7 does not exist')
    return gpus


def source_freeze(tvm_root):
    from tilemega.dnn import tvm_baseline as adapter
    old = os.environ.get('TILEMEGA_DNN1_TVM_ROOT')
    os.environ['TILEMEGA_DNN1_TVM_ROOT'] = str(tvm_root)
    try:
        identity = adapter._source_identity()
        libraries = adapter._libraries()
    finally:
        if old is None:
            os.environ.pop('TILEMEGA_DNN1_TVM_ROOT', None)
        else:
            os.environ['TILEMEGA_DNN1_TVM_ROOT'] = old
    files = dict(identity['modified_sources'])
    files.update(identity['patches'])
    files.update({row['path']: row['sha256'] for row in libraries})
    names = ('dnn1_tvm_trials10000_sample.py', 'dnn1_tvm_trials10000_collect.py',
        'dnn1_baselines.py', 'dnn1_device.py', 'dnn1_baseline_boundaries.py',
        'dnn1_aitemplate_cell.py', 'dnn1_receipt_integrity.py',
        'dnn1_tvm_resume_collect.py', 'dnn1_tvm_resume_finalize.py', 'dnn1_tvm_publish.py',
        'dnn1_tvm_audit.py', 'dnn1_owned_controller.py', 'dnn1_background_cards.py',
        'dnn1_dispatch.py', 'dnn1_fifo_background_python.py', 'dnn1_background_python.py',
        'dnn1_idle_floor_python.py', 'dnn1_resource_entry.py', 'dnn1_guard_entry.py',
        'dnn1_settling_background_python.py', 'dnn1_preflight_settle.py',
        'dnn1_preflight_settle_guard.patch', 'run_r16_with_device_identity.py')
    for name in names:
        path = ROOT / 'scripts' / name
        files[str(path)] = sha(path)
    for name in ('tvm_baseline.py', 'baselines.py'):
        path = ROOT / 'python/tilemega/dnn' / name
        files[str(path)] = sha(path)
    for name in ('gpu_guard.py', 'scheduler.py'):
        path = ROOT / 'python/tilemega/bench' / name
        files[str(path)] = sha(path)
    for name in ('tune.py', 'relax_integration.py', 'search_strategy/evolutionary_search.py'):
        path = tvm_root / 'source/python/tvm/s_tir/meta_schedule' / name
        files[str(path)] = sha(path)
    files[str(base.POLICY)] = sha(base.POLICY)
    files[str(base.CLASSIC_DEVICES)] = sha(base.CLASSIC_DEVICES)
    allowlist = Path(read(base.POLICY)['gpu_allowlist_file'])
    files[str(allowlist)] = sha(allowlist)
    return identity, files


def asset_cells(exports, preparation, historical_rows):
    need(preparation.get('schema') == 1 and preparation.get('status') == 'ready' and
         preparation.get('seed') == 20261007 and set(preparation.get('models', [])) == set(NEW_MODELS),
         'ready CPU asset proof for the three new models and fixed seed is required')
    fresh = preparation.get('cells', [])
    need(len(fresh) == 15 and {(c['model'], c['batch']) for c in fresh} ==
         {(m, b) for m in NEW_MODELS for b in base.BATCHES}, 'new model CPU proof must cover exactly15 cells')
    fresh = {(c['model'], c['batch']): c for c in fresh}
    retained = [r for r in historical_rows if r['system'] == 'tvm' and
                r['model'] in ('resnet18', 'bert') and r['precision'] == 'bf16' and r['status'] == 'pass']
    need(len(retained) == 10 and {(r['model'], int(r['batch'])) for r in retained} ==
         {(m, b) for m in ('resnet18', 'bert') for b in base.BATCHES},
         'the ten historical RN18/BERT TVM references must remain available')
    retained = {(r['model'], int(r['batch'])): r for r in retained}
    cells = []
    for model in PRIORITY_MODELS:
        export = (exports / model).resolve()
        for batch in base.BATCHES:
            assets = {str(p): sha(p) for p in (export / 'manifest.json',
                export / 'exported_program.pt2', export / 'checkpoint/model.safetensors',
                export / f'inputs_b{batch}.safetensors')}
            archive_sha = assets[str(export / 'exported_program.pt2')]
            checkpoint_sha = assets[str(export / 'checkpoint/model.safetensors')]
            if model in NEW_MODELS:
                proof = fresh[model, batch]
                need(proof.get('status') == 'ready' and Path(proof['export']).resolve() == export and
                     proof['exported_program_sha256'] == archive_sha and
                     proof['source_checkpoint_sha256'] == checkpoint_sha and
                     proof['manifest_sha256'] == assets[str(export / 'manifest.json')] and
                     Path(proof['inputs_file']).resolve() == export / f'inputs_b{batch}.safetensors' and
                     proof['inputs_file_sha256'] == assets[str(export / f'inputs_b{batch}.safetensors')],
                     'new model CPU asset proof differs: ' + model + f'_b{batch}')
                expected_inputs = proof['expected_inputs_sha256']
                identity_source = 'preparation_classic_v1 CPU runtime-input proof'
            else:
                old = retained[model, batch]
                need(old['exported_program_sha256'] == archive_sha and
                     old['source_checkpoint_sha256'] == checkpoint_sha, 'historical RN18/BERT assets changed')
                expected_inputs = old['inputs_sha256']
                identity_source = 'unchanged historical formal TVM2000 reference; new TVM10000 remains fresh'
            need(bool(expected_inputs), 'fixed runtime input digest missing')
            cells.append(dict(model=model, batch=batch, export=str(export), assets=assets,
                expected_inputs_sha256=expected_inputs, identity_source=identity_source,
                exported_program_sha256=archive_sha, source_checkpoint_sha256=checkpoint_sha))
    return cells


def prepare(options):
    epoch = options.epoch.resolve()
    need(epoch.is_relative_to(ROOT / 'runs/dnn1'), 'new epochs belong below runs/dnn1')
    need(not epoch.exists(), 'use a new absent epoch; prior caches/evidence are immutable')
    tvm_root = options.tvm_root.resolve()
    identity, files = source_freeze(tvm_root)
    ait_proof = options.ait_source_proof.resolve()
    proof = read(ait_proof)
    new = {str(ROOT / 'scripts' / name) for name in (
        'dnn1_tvm_trials10000_sample.py', 'dnn1_tvm_trials10000_collect.py')}
    need(not new & set(proof['identity']['files']), 'new scripts unexpectedly enter AIT source7 identity')
    installed = (ROOT / 'runs/dnn1/resume/aitemplate/python').resolve()
    need(all(not Path(p).is_relative_to(installed) for p in new), 'new file enters installed AIT tree')
    protected = {str((ROOT / name).resolve()): sha(ROOT / name) for name in (
        'docs/results/DNN1_baselines.csv', 'docs/results/DNN1_tvm_diag.csv')}
    preparation = options.asset_preparation.resolve()
    cells = asset_cells(options.exports, read(preparation), csv_rows(ROOT / 'docs/results/DNN1_baselines.csv'))
    manifest = dict(schema=1, profile=experiment.PROFILE, budget=dict(experiment.BUDGET),
        workload_id=experiment.WORKLOAD_ID, workload=experiment.WORKLOAD,
        asset_preparation=dict(path=str(preparation), sha256=sha(preparation)),
        priority_models=list(PRIORITY_MODELS),
        compatibility_entry=dict(experiment.COMPATIBILITY), fresh_database=True,
        official_cuda_graph=False, epoch=str(epoch), tvm_root=str(tvm_root), cells=cells,
        process_timeout_s=PROCESS_TIMEOUT, cell_timeout_s=CELL_TIMEOUT,
        gpus=gpu_pool(options.gpus), max_parallel=options.max_parallel,
        other_baseline_max_parallel=options.other_baseline_max_parallel,
        concurrency_scope='parent partition across TRT/TC/AIT/TVM; allocated pool only, global slots<=4',
        source_identity=identity, protocol_sources=files, protected_files=protected,
        ait_source7_exclusion=dict(proof=str(ait_proof), proof_sha256=sha(ait_proof),
            new_files_excluded=sorted(new), new_files_outside_installed_tree=str(installed)),
        status='prepared only; release and explicit run required', created=time.time())
    need(1 <= manifest['max_parallel'] <= 4, 'at most four active physical cards')
    dispatch_limit(epoch, manifest)
    affinities(manifest['gpus'], 8, explicit={str(g): AFFINITY[g] for g in manifest['gpus']})
    epoch.mkdir(parents=True, exist_ok=False)
    save(epoch / 'manifest.json', manifest)
    save(epoch / 'progress.json', dict(status='prepared_not_running', accepted=0, pending=25,
        active={}, GPU_jobs_started=0, manifest_sha256=sha(epoch / 'manifest.json')))
    print(json.dumps(dict(status='prepared_only', epoch=str(epoch), cells=25,
        manifest_sha256=sha(epoch / 'manifest.json'))))


def validate_manifest(epoch):
    epoch = Path(epoch).resolve()
    manifest = read(epoch / 'manifest.json')
    need(manifest.get('profile') == experiment.PROFILE and manifest['budget'] == experiment.BUDGET and
         manifest['fresh_database'] is True and manifest['official_cuda_graph'] is False,
         'fixed fresh10000 epoch required')
    need(manifest.get('workload_id') == experiment.WORKLOAD_ID and manifest.get('workload') == experiment.WORKLOAD,
         'manifest belongs to a different workload; retain it and prepare a new epoch')
    need(sha(manifest['asset_preparation']['path']) == manifest['asset_preparation']['sha256'],
         'classic CPU asset preparation proof changed')
    need(manifest['epoch'] == str(epoch) and epoch.is_relative_to(ROOT / 'runs/dnn1'), 'epoch binding differs')
    need({(c['model'], c['batch']) for c in manifest['cells']} == CELLS and len(manifest['cells']) == 25,
         'full25 distinct cells required')
    need(1 <= manifest['max_parallel'] <= 4 and manifest['process_timeout_s'] >= 18000 and
         manifest['cell_timeout_s'] > 5 * manifest['process_timeout_s'], 'dispatch limit/outer timeouts changed')
    need(gpu_pool(','.join(map(str, manifest['gpus']))) == manifest['gpus'], 'GPU pool changed')
    dispatch_limit(epoch, manifest)
    for field in ('protocol_sources', 'protected_files'):
        for path, digest in manifest[field].items():
            need(sha(path) == digest, 'frozen experiment or original25 file changed: ' + path)
    if (epoch / 'withdrawal.json').exists():
        raise ValueError('epoch withdrawn; preserve raw data')
    current_diff = subprocess.check_output(['git', '-C', str(Path(manifest['tvm_root']) / 'source'),
        'diff', '--binary', 'HEAD'])
    need(hashlib.sha256(current_diff).hexdigest() == manifest['source_identity']['source_diff_sha256'],
         'the pinned TVM source overlay changed')
    return manifest


def terminal_ait(csv_path):
    selected = [r for r in csv_rows(csv_path) if r.get('system') == 'aitemplate']
    need(len(selected) == 25 and {(r['model'], int(r['batch'])) for r in selected} == CELLS,
         'AIT must have exactly25 distinct terminal cells')
    evidence = {}
    for row in selected:
        need(row['precision'] == 'bf16' and row['status'] in ('pass', 'N/A'), 'AIT outcome not terminal')
        need(bool(row.get('versions')) and bool(row.get('source')), 'terminal AIT row lacks source/version')
        source = json.loads(row['source'])
        if isinstance(source, str):
            source = [source]
        need(isinstance(source, list) and source, 'terminal AIT source list empty')
        payloads = []
        for item in source:
            path = Path(item['path'] if isinstance(item, dict) else item)
            evidence[str(path.resolve())] = sha(path)
            payloads.append(read(path))
        if row['status'] == 'pass':
            need(row.get('correctness_passes') == '5/5' and len(source) == 5, 'AIT pass lacks whole P5')
            need(len({p['pid'] for p in payloads}) == 5 and
                 all(p.get('system') == 'aitemplate' and p.get('precision') == 'bf16' and
                     p.get('model') == row['model'] and p.get('batch') == int(row['batch']) and
                     p.get('correctness_passed') is True for p in payloads), 'AIT pass receipt identity/P5 differs')
            need(sum(p.get('process_role') == 'measurement' for p in payloads) == 3 and
                 sum(p.get('process_role') == 'correctness' for p in payloads) == 2,
                 'AIT terminal pass has wrong fresh process roles')
            need(all(p['identity']['exported_program_sha256'] == row['exported_program_sha256'] and
                     p['identity']['source_checkpoint_sha256'] == row['source_checkpoint_sha256'] and
                     p['inputs_sha256'] == row['inputs_sha256'] for p in payloads),
                 'AIT terminal receipts use different exports/weights/inputs')
            artifacts = json.loads(row['engine_or_library_sha256'])
            need(all(p['library_sha256'] == artifacts['library_sha256'] and
                     p['compiled_receipt_sha256'] == artifacts['compiled_receipt_sha256'] for p in payloads),
                 'AIT terminal receipt/library identity differs from its published row')
            session = Path(row['cell_session'])
            value = read(session)
            outer = session.parents[2] / 'guard_result.json'
            need(read(outer)['code'] == 0 and value.get('status') == 'complete_pending_outer_audit',
                 'AIT pass lacks its original accepted outer guard and complete cell')
            evidence[str(session.resolve())] = sha(session)
            evidence[str(outer.resolve())] = sha(outer)
            for p in payloads:
                need(base.strict_correctness(row['model'], p['correctness'])[0], 'AIT original numerical gate failed')
        else:
            need(bool(row.get('reason')), 'AIT N/A lacks its original failure reason')
            need(not any(word in row['reason'].lower() for word in
                ('pending', 'waiting', 'interference', 'guard75', 'guard rejection', 'cancelled')),
                'AIT infrastructure/pending outcome cannot release retuning')
            need(any(p.get('error') or p.get('failure_kind') for p in payloads),
                 'AIT terminal failure lacks actual original failure evidence')
    return selected, evidence


def seal_release(options):
    epoch = options.epoch.resolve()
    validate_manifest(epoch)
    need(not (epoch / 'release.json').exists(), 'retain existing release; do not silently refresh it')
    csv_path = options.ait_csv.resolve()
    need(csv_path.is_relative_to(ROOT), 'AIT terminal CSV must be in the published repository')
    rows, evidence = terminal_ait(csv_path)
    completion = options.ait_completion_ledger.resolve()
    validate_ait_completion(completion, csv_path, rows)
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', options.pushed_commit + '^{commit}'], text=True).strip()
    branch = options.branch
    remote = subprocess.check_output(['git', '-C', str(ROOT), 'ls-remote', 'origin', 'refs/heads/' + branch], text=True).strip().split()
    need(remote and remote[0] == commit, 'the selected commit is not the actual pushed branch tip')
    blob = subprocess.check_output(['git', '-C', str(ROOT), 'show', commit + ':' + str(csv_path.relative_to(ROOT))])
    need(hashlib.sha256(blob).hexdigest() == sha(csv_path), 'terminal AIT CSV is not in the pushed commit')
    blob = subprocess.check_output(['git', '-C', str(ROOT), 'show',
        commit + ':' + str(completion.relative_to(ROOT))])
    need(hashlib.sha256(blob).hexdigest() == sha(completion), 'AIT final completion declaration is not pushed')
    result = dict(schema=1, profile=experiment.PROFILE, epoch_manifest_sha256=sha(epoch / 'manifest.json'),
        workload_id=experiment.WORKLOAD_ID, workload=experiment.WORKLOAD,
        ait_terminal_csv=str(csv_path), ait_terminal_csv_sha256=sha(csv_path), ait_terminal_cells=25,
        ait_status_counts=dict(__import__('collections').Counter(r['status'] for r in rows)),
        ait_source_files=evidence, pushed_commit=commit, pushed_branch=branch,
        ait_completion_ledger=str(completion), ait_completion_ledger_sha256=sha(completion),
        remote_ref='origin/refs/heads/' + branch, remote_tip_verified=True, sealed=time.time())
    save(epoch / 'release.json', result)
    print(json.dumps(dict(status='released_only_not_started', release_sha256=sha(epoch / 'release.json'))))


def validate_ait_completion(path, csv_path, rows):
    """Repairable failed cohorts are not a declaration that AIT is finished."""
    path = Path(path).resolve()
    need(path.is_relative_to(ROOT), 'AIT completion declaration must be published in the repository')
    value = read(path)
    need(value.get('schema') == 1 and value.get('system') == 'aitemplate' and
         value.get('workload_id') == experiment.WORKLOAD_ID and value.get('workload') == experiment.WORKLOAD and
         value.get('status') == 'final_25_terminal' and value.get('repair_pending') is False and
         value.get('csv_sha256') == sha(csv_path), 'explicit final AIT25 completion declaration required')
    want = {(r['model'], int(r['batch']), r['status']) for r in rows}
    cells = value.get('cells', [])
    need(len(cells) == 25 and {(c['model'], c['batch'], c['status']) for c in cells} == want and
         all(c.get('final') is True for c in cells), 'AIT completion declaration differs from its terminal CSV')
    return value


def validate_release(epoch, expected=None):
    manifest = validate_manifest(epoch)
    path = Path(epoch) / 'release.json'
    digest = sha(path)
    need(expected is None or expected == digest, 'AIT/push release changed during collection')
    value = read(path)
    need(value['profile'] == experiment.PROFILE and value['epoch_manifest_sha256'] == sha(Path(epoch) / 'manifest.json') and
         value.get('workload_id') == experiment.WORKLOAD_ID and value.get('workload') == experiment.WORKLOAD and
         value['ait_terminal_cells'] == 25 and value['remote_tip_verified'] is True, 'wrong release binding')
    need(sha(value['ait_terminal_csv']) == value['ait_terminal_csv_sha256'], 'published AIT ledger changed')
    rows, _ = terminal_ait(value['ait_terminal_csv'])
    need(sha(value['ait_completion_ledger']) == value['ait_completion_ledger_sha256'],
         'AIT completion declaration changed')
    validate_ait_completion(value['ait_completion_ledger'], value['ait_terminal_csv'], rows)
    for name, want in value['ait_source_files'].items():
        need(sha(name) == want, 'AIT terminal evidence changed: ' + name)
    blob = subprocess.check_output(['git', '-C', str(ROOT), 'show',
        value['pushed_commit'] + ':' + str(Path(value['ait_terminal_csv']).relative_to(ROOT))])
    need(hashlib.sha256(blob).hexdigest() == value['ait_terminal_csv_sha256'],
         'release AIT terminal ledger differs from its committed bytes')
    blob = subprocess.check_output(['git', '-C', str(ROOT), 'show', value['pushed_commit'] + ':' +
        str(Path(value['ait_completion_ledger']).relative_to(ROOT))])
    need(hashlib.sha256(blob).hexdigest() == value['ait_completion_ledger_sha256'],
         'release AIT completion declaration differs from its committed bytes')
    return manifest, digest


def run_cell(options):
    prepared = read(options.cell)
    base.require_session(prepared)
    validate_release(prepared['epoch'], prepared['release_sha256'])
    options.out.mkdir(parents=True, exist_ok=True)
    need(not any((options.out / name).exists() for name in
         ('runtime-cell.json', 'cell-result.json', 'measurements')), 'retain the original cell attempt')
    cache = Path(prepared['cache_root']) / options.out.parent.name
    need(not cache.exists(), 'guard retry must use an absent cohort cache')
    runtime = dict(prepared, cache=str(cache), fresh_database=True)
    runtime_path = options.out / 'runtime-cell.json'
    save(runtime_path, runtime)
    session = dict(schema=1, status='running', model=runtime['model'], batch=runtime['batch'],
        gpu=runtime['gpu'], logical_jobs=[], started=time.time(), p_cell_complete=False,
        cell_spec=str(runtime_path), cell_spec_sha256=sha(runtime_path),
        prepared_cell=str(options.cell), prepared_cell_sha256=sha(options.cell),
        profile=experiment.PROFILE, cache_initially_absent=True)
    try:
        for index in range(5):
            experiment.validate_cell(runtime)
            before_drain = idle_boundary(runtime)
            session['pending_boundary'] = before_drain
            before = final_boundary(before_drain)
            role = 'measurement' if index < 3 else 'correctness'
            position = index if index < 3 else index - 3
            folder = options.out / 'measurements' / f'{role}-{position}' / 'tvm-bf16/attempt1'
            payload = folder / 'payload'
            command = [runtime['python'], str(ROOT / 'scripts/dnn1_tvm_trials10000_sample.py'),
                '--cell', str(runtime_path), '--out', str(payload), '--round-index', str(index)]
            if role == 'correctness':
                command += ['--correctness-only']
            result = base.execute(command, folder, os.environ.copy(), runtime['process_timeout_s'])
            drain = idle_boundary(runtime)
            after = drain['samples'][-1]['telemetry'] if drain.get('samples') else {}
            receipt = payload / 'measurements.json'
            result.update(folder=str(payload), process_before=before, process_after=after,
                before_drain=before_drain, drain=drain, measurement=str(receipt) if receipt.exists() else None,
                receipt_sha256=sha(receipt) if receipt.exists() else None,
                status='pass' if result['returncode'] == 0 else 'fail')
            job = dict(arm=dict(arm='tvm-bf16', system='tvm', precision='bf16'), role=role,
                index=position, status=result['status'], attempts=[result], final=result)
            session['logical_jobs'].append(job)
            session.pop('pending_boundary', None)
            save(options.out / 'cell-result.json', session)
            final_boundary(drain)
            if receipt.exists():
                value = read(receipt)
                snapshots = [json.loads(s) for s in (payload / 'guard.jsonl').read_text().splitlines()]
                for label, field in [('before-measurement', 'clock_start'), ('after-measurement', 'clock_end')]:
                    sample = next(s for s in snapshots if s['label'] == label)
                    result[field] = dict(sm_clock_mhz=float(sample['telemetry']['clocks.sm']), sample=sample)
                inputs = value['configuration']['inputs_sha256']
                need(session.get('inputs_sha256', inputs) == inputs, 'fixed inputs changed across fresh processes')
                session['inputs_sha256'] = inputs
            elif (payload / 'failure.json').exists():
                result['failure'] = read(payload / 'failure.json')
            save(options.out / 'cell-result.json', session)
            need(result['returncode'] == 0 and receipt.exists(), 'original process failed; whole cohort retained')
        session.update(status='complete', ended=time.time())
        return 0
    except BaseException as error:
        session.update(status='failed', error=repr(error), ended=time.time())
        raise
    finally:
        save(options.out / 'cell-result.json', session)


def collect_cell(folder, manifest, release_sha):
    state = read(folder / 'scheduler/state.json')
    prepared = read(folder / 'cell.json')
    key = f"{prepared['model']}_b{prepared['batch']}"
    need(state[key]['status'] == 'done', 'scheduler did not accept the whole cell')
    attempt = Path(state[key]['out']).resolve()
    need(attempt.is_relative_to((folder / 'raw').resolve()), 'raw attempt lies outside its scheduler')
    session = read(attempt / 'job/cell-result.json')
    cell = read(session['cell_spec'])
    need(sha(session['cell_spec']) == session['cell_spec_sha256'] and
         session['prepared_cell_sha256'] == sha(folder / 'cell.json'), 'runtime/prepared cell changed')
    need(cell['epoch_manifest_sha256'] == sha(Path(manifest['epoch']) / 'manifest.json') and
         cell['release_sha256'] == release_sha and session['cache_initially_absent'] is True,
         'runtime epoch/release/fresh-cache identity differs')
    experiment.validate_cell(cell)
    receipts = [read(j['final']['measurement']) for j in session['logical_jobs']]
    need(len(receipts) == 5 and [r['round_index'] for r in receipts] == list(range(5)), 'full indivisible P5 required')
    for index, receipt in enumerate(receipts):
        diag = receipt['configuration']['tvm_diag']
        expected_recipe = diag['cache_identity']['recipe']
        need(expected_recipe['schema'] == experiment.PROFILE and expected_recipe['tuning'] == experiment.BUDGET and
             expected_recipe['workload'] == experiment.WORKLOAD and
             expected_recipe['fresh_database'] is True and expected_recipe['prior_database'] is None and
             expected_recipe['official_cuda_graph'] is False, 'wrong compiled fresh-budget identity')
        need(expected_recipe['compatibility_entry'] == experiment.COMPATIBILITY and
             expected_recipe['epoch_manifest_sha256'] == cell['epoch_manifest_sha256'] and
             diag['cache_identity']['tvm_source_identity'] == manifest['source_identity'],
             'compiled compatibility/source/epoch metadata differs')
        tuning = diag['tuning']
        need(tuning['maximum_trials'] == 10000 and tuning['timeout_s'] == 9000 and
             tuning['initial_database_records'] == 0, 'actual worker budget differs from manifest')
        need(sha(diag['model_library']) == diag['model_sha256'], 'compiled VM changed')
        need(receipt['driver_sha256'] == sha(ROOT / 'scripts/dnn1_tvm_trials10000_sample.py'), 'measurement runner changed')
        need(receipt['exported_program_sha256'] == cell['exported_program_sha256'] and
             receipt['source_checkpoint_sha256'] == cell['source_checkpoint_sha256'], 'measurement weights/export differ')
        need(receipt['configuration']['inputs_sha256'] == cell['expected_inputs_sha256'],
             'measurement inputs differ from original25')
        if index < 3:
            values = sorted(receipt['samples_ms'])
            need(len(values) == 200 and all(math.isfinite(x) and x > 0 for x in values) and
                 math.isclose(receipt['median_ms'], statistics.median(values), rel_tol=1e-12) and
                 math.isclose(receipt['p90_ms'], values[179], rel_tol=1e-12), 'timing samples/summary differ')
        else:
            need(receipt['samples_ms'] == [], 'extra correctness process contains timing samples')
    tuning = receipts[0]['configuration']['tvm_diag']['tuning']
    need(all(r['configuration']['tvm_diag']['tuning'] == tuning for r in receipts), 'processes used different tuning epochs')
    work = Path(tuning['work_dir']).resolve()
    need(work.is_relative_to(Path(cell['cache']).resolve()), 'actual DB lies outside fresh cohort cache')
    need(sha(tuning['original_tuning_receipt']) == tuning['original_tuning_receipt_sha256'] and
         sha(tuning['actual_worker_request']) == tuning['actual_worker_request_sha256'], 'tune request/raw receipt changed')
    need(read(tuning['actual_worker_request'])['max_trials_global'] == 10000, 'real worker request cap differs')
    records = [json.loads(line) for line in (work / 'database_tuning_record.json').read_text().splitlines() if line.strip()]
    successes = sum(bool(r[1][1]) and all(float(v) < 1e9 for v in r[1][1]) for r in records)
    need(len(records) == tuning['committed_trials'] and successes == tuning['successful_trials'] and
         len(records) - successes == tuning['failed_trials'], 'actual DB trial counts differ')
    builders = [json.loads(line) for line in (work / 'builder_results.jsonl').read_text().splitlines() if line.strip()]
    need(len(builders) == tuning.get('attempted_builds') and
         sum(bool(r.get('error')) for r in builders) == tuning.get('build_errors'),
         'actual builder attempt/error counts differ')
    destination = folder / 'audits' / uuid.uuid4().hex
    row = finalizer.collect(cell, attempt, destination)
    row.update(workload_id=experiment.WORKLOAD_ID,
        experiment_profile=experiment.PROFILE, requested_trials=10000, tuning_timeout_s=9000,
        committed_trials=tuning['committed_trials'], successful_trials=tuning['successful_trials'],
        failed_trials=tuning['failed_trials'], attempted_builds=tuning.get('attempted_builds', ''),
        build_errors=tuning.get('build_errors', ''), tune_wall_seconds=tuning['elapsed_s'],
        deadline_hit=tuning['deadline_hit'], tuning_receipt=tuning['original_tuning_receipt'],
        tuning_receipt_sha256=tuning['original_tuning_receipt_sha256'],
        experiment_manifest_sha256=cell['epoch_manifest_sha256'])
    evidence = {str(attempt / 'job/cell-result.json'): sha(attempt / 'job/cell-result.json'),
        str(attempt / 'guard_result.json'): sha(attempt / 'guard_result.json'),
        str(attempt / 'guard.jsonl'): sha(attempt / 'guard.jsonl')}
    for pattern in ('database_*.json', '*_results.jsonl', 'receipt.json', 'request.json', 'tuning.log', 'fresh_budget_contract.json'):
        evidence.update({str(p): sha(p) for p in work.glob(pattern)})
    evidence.update({p: sha(p) for p in json.loads(row['source'])})
    return dict(schema=1, profile=experiment.PROFILE, row=row, attempt=str(attempt),
        raw_files=evidence, annotation_audit=str(destination / 'accepted.json'),
        annotation_audit_sha256=sha(destination / 'accepted.json'))


def enqueue(spec, manifest, gpu, release_sha, lifetime):
    epoch = Path(manifest['epoch'])
    key = f"{spec['model']}_b{spec['batch']}"
    folder = epoch / 'jobs' / key
    folder.mkdir(parents=True, exist_ok=False)
    cell = dict(spec, epoch=str(epoch), profile=experiment.PROFILE, budget=dict(experiment.BUDGET),
        workload_id=experiment.WORKLOAD_ID, workload=experiment.WORKLOAD,
        fresh_database=True, official_cuda_graph=False, prior_database=None,
        tvm_root=manifest['tvm_root'], gpu=gpu, cpu_affinity=AFFINITY[gpu], parallel=True,
        python=str(ROOT / 'scripts/dnn1_fifo_background_python.py'),
        process_timeout_s=manifest['process_timeout_s'], cell_timeout_s=manifest['cell_timeout_s'],
        protocol_sources=manifest['protocol_sources'], epoch_manifest=str(epoch / 'manifest.json'),
        asset_preparation=manifest['asset_preparation'],
        epoch_manifest_sha256=sha(epoch / 'manifest.json'), release=str(epoch / 'release.json'),
        release_sha256=release_sha, cache_root=str(epoch / 'cache' / key),
        correctness_policy='five-independent-gates-v2', prefix=dict(available=False),
        boundary_protocol='paired-process-memory-v2')
    need(not Path(cell['cache_root']).exists(), 'fresh per-cell cache root already exists')
    save(folder / 'cell.json', cell)
    env = base.gpu_env(gpu, dict(TILEMEGA_NUMA_CONCURRENCY_TRIAL='1',
        TILEMEGA_DNN1_TVM_ROOT=manifest['tvm_root'], PYTHONDONTWRITEBYTECODE='1',
        OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', OMP_NUM_THREADS='4'))
    values = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid,pci.bus_id',
        '--format=csv,noheader,nounits'], text=True).strip().split(',')
    device_uuid, pci = [s.strip() for s in values]
    domain, bus, slot = pci.split(':')
    pci = f'{int(domain,16):04x}:{bus}:{slot}'
    expected = base.classic_device_identity(gpu)
    need(device_uuid == expected['uuid'] and pci.lower() == expected['pci'].lower(),
         'current NVML identity differs from the frozen classic device mapping')
    python = cell['python']
    command = ['numactl', '--all', '--physcpubind=' + ','.join(map(str, AFFINITY[gpu])), python,
        str(ROOT / 'scripts/run_r16_with_device_identity.py'), '--guard-policy', str(base.POLICY),
        '--out', '{attempt_dir}/job', '--expected-uuid', device_uuid, '--expected-pci', pci,
        '--numa-node', str(0 if gpu < 4 else 1), '--allow-no-measurements', '--', python,
        str(Path(__file__).resolve()), 'cell', '--cell', str(folder / 'cell.json'), '--out', '{attempt_dir}/job']
    save(folder / 'queue/queue_tvm_trials10000.json', [dict(name=key, gpu=True, timing=True,
        priority=200, timeout_s=manifest['cell_timeout_s'], needs_free_mib=8192,
        cwd=str(ROOT), after=[], env=env, command=command)])
    launcher = ['flock', f'/tmp/tilemega_dnn1_baseline_scheduler_gpu{gpu}.lock', 'env',
        '-u', 'TILEMEGA_DEVICE_GROUP', '-u', 'TILEMEGA_GPU_LOCK_HELD', '-u', 'TILEMEGA_RUN_NONCE',
        *[f'{k}={v}' for k, v in env.items()], python, '-m', 'tilemega.bench.scheduler',
        '--queue-dir', str(folder / 'queue'), '--out', str(folder / 'scheduler'),
        '--policy', str(base.POLICY), '--deadline-hours', '168']
    log = (folder / 'scheduler.log').open('w')
    process = None
    try:
        process = subprocess.Popen(launcher, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
        lifetime.register(process)
        save(folder / 'launch.json', dict(pid=process.pid, controller_nonce=lifetime.nonce,
            release_sha256=release_sha, command=launcher, started=time.time()))
        print(json.dumps(dict(event='queued', cell=key, gpu=gpu, pid=process.pid,
            profile=experiment.PROFILE)), flush=True)
        return process, log, folder
    except BaseException:
        log.close()
        raise


def prior_jobs_live(epoch):
    nonces = {read(p)['controller_nonce'] for p in Path(epoch).glob('jobs/*/launch.json')}
    if not nonces:
        return False
    needles = {(NONCE_KEY + '=' + nonce).encode() for nonce in nonces}
    for path in Path('/proc').glob('[0-9]*/environ'):
        try:
            if needles & set(path.read_bytes().split(b'\0')):
                fields = path.with_name('stat').read_text().rsplit(')', 1)[1].split()
                if fields[0] not in ('Z', 'X'):
                    return True
        except (OSError, IndexError):
            pass
    return False


def collect_epoch(epoch, manifest, release_sha):
    accepted, failures = {}, {}
    for spec in manifest['cells']:
        key = f"{spec['model']}_b{spec['batch']}"
        folder = Path(epoch) / 'jobs' / key
        if not (folder / 'scheduler/state.json').exists():
            continue
        state = read(folder / 'scheduler/state.json').get(key, {})
        if state.get('status') == 'done':
            try:
                result = collect_cell(folder, manifest, release_sha)
                cache = folder / 'accepted.json'
                if cache.exists():
                    old = read(cache)
                    need(old['row'] == result['row'] and old['raw_files'] == result['raw_files'],
                         'accepted cache differs from complete original raw re-audit')
                save(cache, result)
                accepted[key] = result['row']
            except Exception as error:
                failures[key] = dict(error=repr(error), folder=str(folder), status='quarantined')
        elif state.get('status') in ('failed', 'cancelled', 'canceled', 'skipped'):
            failures[key] = dict(status=state['status'], original_scheduler=state,
                folder=str(folder), disposition='original raw retained; no fallback or mixed attempt')
    for key, failure in failures.items():
        trials = []
        cache = Path(epoch) / 'cache' / key
        for path in sorted(cache.glob('attempt*/**/receipt.json')):
            try:
                value = read(path)
                if value.get('maximum_trials') == 10000:
                    trials.append(dict(path=str(path), sha256=sha(path), counts=value))
            except (OSError, ValueError):
                continue
        failure['retained_tuning_attempts'] = trials
    ordered = [accepted[f"{c['model']}_b{c['batch']}"] for c in manifest['cells'] if f"{c['model']}_b{c['batch']}" in accepted]
    csv_write(Path(epoch) / 'DNN_classic_tvm_trials10000.csv', ordered)
    outcomes = []
    for spec in manifest['cells']:
        key = f"{spec['model']}_b{spec['batch']}"
        if key in accepted:
            outcomes.append(accepted[key])
        elif key in failures:
            outcome = {name: '' for name in list(base.FIELDS) + EXTRA_FIELDS}
            outcome.update(model=spec['model'], batch=spec['batch'], system='tvm',
                arm='tvm-bf16', precision='bf16', status='quarantined',
                reason=base.compact(failures[key]), source=failures[key]['folder'],
                workload_id=experiment.WORKLOAD_ID,
                experiment_profile=experiment.PROFILE, requested_trials=10000, tuning_timeout_s=9000,
                experiment_manifest_sha256=sha(Path(epoch) / 'manifest.json'))
            outcomes.append(outcome)
    csv_write(Path(epoch) / 'DNN_classic_tvm_trials10000_outcomes.csv', outcomes)
    save(Path(epoch) / 'collection.json', dict(status='complete' if len(accepted) == 25 else 'incomplete',
        accepted=len(accepted), terminal=len(outcomes), failures=failures, profile=experiment.PROFILE,
        workload_id=experiment.WORKLOAD_ID,
        manifest_sha256=sha(Path(epoch) / 'manifest.json'), release_sha256=release_sha,
        original_results_unchanged=manifest['protected_files'], time=time.time()))
    return accepted, failures


def dispatch_limit(epoch, manifest):
    path = Path(epoch) / 'dispatch_limit.json'
    if path.exists():
        value = read(path)
        need(value.get('workload_id') == experiment.WORKLOAD_ID and
             value.get('manifest_sha256') == sha(Path(epoch) / 'manifest.json') and value.get('reason'),
             'dynamic limit must bind the prepared workload and explicit parent allocation')
        limit, others = value['max_parallel'], value['other_baseline_max_parallel']
    else:
        limit, others = manifest['max_parallel'], manifest['other_baseline_max_parallel']
    need(type(limit) is int and type(others) is int and 1 <= limit <= 4 and
         0 <= others <= 3 and limit + others <= 4 and limit <= len(manifest['gpus']),
         'parent-assigned baseline slots must total at most four')
    return limit


def dispatch(options):
    epoch = options.epoch.resolve()
    manifest, release_sha = validate_release(epoch)
    released = read(epoch / 'release.json')
    remote = subprocess.check_output(['git', '-C', str(ROOT), 'ls-remote', 'origin',
        'refs/heads/' + released['pushed_branch']], text=True).strip().split()
    need(remote, 'cannot verify pushed AIT checkpoint')
    if remote[0] != released['pushed_commit']:
        need(subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor',
            released['pushed_commit'], remote[0]], check=False).returncode == 0,
            'the released AIT checkpoint is not on the current remote branch')
    need(not prior_jobs_live(epoch), 'earlier controller jobs still live; refuse duplicate dispatch')
    lock = (epoch / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run_id = uuid.uuid4().hex
    active = {}
    try:
        with OwnedController(epoch / 'controllers' / run_id) as lifetime:
            save(epoch / 'controller.json', dict(pid=os.getpid(), nonce=lifetime.nonce,
                profile=experiment.PROFILE, run_id=run_id, release_sha256=release_sha, started=time.time()))
            accepted, failures = collect_epoch(epoch, manifest, release_sha)
            pending = [s for s in manifest['cells'] if not (epoch / 'jobs' / f"{s['model']}_b{s['batch']}").exists()]
            while pending or active:
                validate_release(epoch, release_sha)
                limit = dispatch_limit(epoch, manifest)
                for gpu, (process, log, folder) in list(active.items()):
                    if process.poll() is None:
                        continue
                    log.close()
                    print(json.dumps(dict(event='scheduler_finished', gpu=gpu,
                        pid=process.pid, exit_code=process.returncode)), flush=True)
                    del active[gpu]
                    accepted, failures = collect_epoch(epoch, manifest, release_sha)
                for gpu in manifest['gpus']:
                    if not pending or len(active) >= limit:
                        break
                    if gpu in active or main_reservations(gpu) or base.held(f'/tmp/tilemega_gpu{gpu}.lock'):
                        continue
                    try:
                        sample = device.gpu(gpu)
                    except Exception as error:
                        with (epoch / 'resource_errors.jsonl').open('a') as stream:
                            stream.write(json.dumps(dict(gpu=gpu, error=repr(error), time=time.time())) + '\n')
                        continue
                    if sample['owners'] or not 0 <= sample['used_mib'] <= 1 or sample['free_mib'] < 8192:
                        continue
                    active[gpu] = enqueue(pending.pop(0), manifest, gpu, release_sha, lifetime)
                save(epoch / 'progress.json', dict(status='collecting', accepted=len(accepted), failures=failures,
                    pending=[f"{c['model']}_b{c['batch']}" for c in pending],
                    active={str(g): dict(pid=p.pid, folder=str(f)) for g, (p, _, f) in active.items()},
                    max_parallel=limit, gpus=manifest['gpus'],
                    dispatch_limit_sha256=sha(epoch / 'dispatch_limit.json') if (epoch / 'dispatch_limit.json').exists() else None,
                    profile=experiment.PROFILE, power_monitoring=False, time=time.time()))
                if pending or active:
                    time.sleep(15)
            accepted, failures = collect_epoch(epoch, manifest, release_sha)
            save(epoch / 'progress.json', dict(status='complete' if len(accepted) == 25 else 'needs_diagnosis',
                accepted=len(accepted), pending=[], active={}, failures=failures,
                profile=experiment.PROFILE, power_monitoring=False, ended=time.time()))
            return int(len(accepted) != 25)
    except BaseException as error:
        save(epoch / 'controller_error.json', dict(error=repr(error), run_id=run_id,
            unfinished_cohorts_invalidated=True, release_sha256=release_sha, time=time.time()))
        save(epoch / 'progress.json', dict(status='stopped', error=repr(error),
            active={}, pending='see retained controller/session records',
            cleanup=str(epoch / 'controllers' / run_id / 'controller_cleanup.json'), time=time.time()))
        raise
    finally:
        for process, log, folder in active.values():
            if not log.closed:
                log.close()
        lock.close()


def wait_for_release(options):
    epoch = options.epoch.resolve()
    validate_manifest(epoch)
    need(options.poll_seconds >= 1, 'positive CPU file polling interval required')
    started = time.time()
    try:
        while not (epoch / 'release.json').exists():
            save(epoch / 'waiting.json', dict(status='waiting_for_explicit_AIT25_release',
                pid=os.getpid(), started=started, updated=time.time(),
                release=str(epoch / 'release.json'), GPU_jobs_started=0,
                GPU_queries=0, physical_locks_acquired=0))
            time.sleep(options.poll_seconds)
        save(epoch / 'waiting.json', dict(status='release_seen_validation_pending',
            pid=os.getpid(), started=started, updated=time.time(),
            release=str(epoch / 'release.json'), release_sha256=sha(epoch / 'release.json')))
        return dispatch(options)
    except BaseException as error:
        save(epoch / 'wait_error.json', dict(error=repr(error), pid=os.getpid(),
            started=started, ended=time.time(), disposition='raw retained; no automatic replacement epoch'))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--epoch', type=Path, required=True)
    p.add_argument('--exports', type=Path, default=ROOT / 'runs/dnn1/exports')
    p.add_argument('--asset-preparation', type=Path, default=ROOT / 'runs/dnn1/exports/preparation_classic_v1.json')
    p.add_argument('--tvm-root', type=Path, default=ROOT / 'runs/dnn1/resume_tvm/typed_runner_repair_v1/tvm')
    p.add_argument('--ait-source-proof', type=Path, default=ROOT / 'runs/dnn1/resume/aitemplate/runtime_ABI_graph_v7/prepare/prepare.json')
    p.add_argument('--gpus', required=True, help='explicit parent-assigned pool, current indices0–6 only')
    p.add_argument('--max-parallel', type=int, choices=range(1, 5), required=True)
    p.add_argument('--other-baseline-max-parallel', type=int, choices=range(4), required=True,
        help='parent-reserved concurrent TRT/TC/AIT slots; total including TVM must be <=4')
    p = sub.add_parser('release')
    p.add_argument('--epoch', type=Path, required=True)
    p.add_argument('--ait-csv', type=Path, required=True)
    p.add_argument('--ait-completion-ledger', type=Path, required=True,
        help='pushed schema1 final_25_terminal AIT declaration; repair_pending=false and25 final cell identities')
    p.add_argument('--pushed-commit', required=True)
    p.add_argument('--branch', default='dnn1')
    for action in ('run', 'collect'):
        p = sub.add_parser(action)
        p.add_argument('--epoch', type=Path, required=True)
    p = sub.add_parser('wait', help='CPU file polling only until release.json exists; then validate and run')
    p.add_argument('--epoch', type=Path, required=True)
    p.add_argument('--poll-seconds', type=float, default=30)
    p = sub.add_parser('cell')
    p.add_argument('--cell', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    options = parser.parse_args()
    if options.action == 'prepare':
        return prepare(options)
    if options.action == 'release':
        return seal_release(options)
    if options.action == 'cell':
        return run_cell(options)
    if options.action == 'run':
        return dispatch(options)
    if options.action == 'wait':
        return wait_for_release(options)
    manifest, digest = validate_release(options.epoch.resolve())
    collect_epoch(options.epoch.resolve(), manifest, digest)


if __name__ == '__main__':
    raise SystemExit(main())
