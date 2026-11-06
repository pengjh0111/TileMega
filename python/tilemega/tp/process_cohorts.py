"""Preserve independent TP starts when a later guarded process is interrupted."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
from types import MethodType


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def checked(value):
    path = Path(value['path'])
    if sha(path) != value['sha256']:
        raise ValueError('guarded-process evidence changed: ' + str(path))
    return path


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def validate_run(value):
    data = json.loads(checked(value).read_text())
    if data['kind'] != 'mx7_guarded_process' or not data['attempts']:
        raise ValueError('missing individually guarded process evidence')
    checked(data['producer'])
    for number, item in enumerate(data['attempts']):
        process = json.loads(checked(item['process']).read_text())
        guard = json.loads(checked(item['guard']).read_text())
        last = number + 1 == len(data['attempts'])
        if last:
            if process['returncode'] != 0 or guard['code'] != 0 or item['child_pid'] is None:
                raise ValueError('individual process did not pass its guard')
            if int(checked(item['pgid']).read_text()) != item['child_pid']:
                raise ValueError('guarded child PID differs from its captured PGID')
        elif process['returncode'] != 75 or guard['code'] != 75 or item['child_pid'] is not None or 'preflight' not in guard['reason']:
            raise ValueError('only unlaunched preflight refusals may precede the accepted process')
    return data


def wait_idle(cell, folder, root):
    from tilemega.bench.gpu_guard import gpu, idle
    policy = json.loads((root / 'configs/bench/guard_policy_c550.json').read_text())
    startup_limit = getattr(cell.a, 'startup_hidden_mib', None)
    if startup_limit is not None and not 0 <= startup_limit <= policy.get('max_hidden_mib', 1024):
        raise ValueError('startup scheduling may only narrow the existing idle condition')
    start = time.monotonic()
    log = folder.parent / (folder.name + '_readiness.jsonl')
    log.parent.mkdir(parents=True, exist_ok=True)
    while True:
        rows, errors = {}, {}
        for gpu_index in cell.a.group.split(','):
            try:
                rows[gpu_index] = gpu(policy, gpu_index)
            except Exception as error:
                errors[gpu_index] = type(error).__name__ + ': ' + str(error)
        ready = not errors and all(idle(row, policy, 12288, g) for g, row in rows.items())
        if ready and startup_limit is not None:
            ready = all(row['hidden_mib'] <= startup_limit for row in rows.values())
        with log.open('a') as stream:
            stream.write(json.dumps(dict(time=time.time(), ready=ready, rows=rows, telemetry_errors=errors,
                                         startup_hidden_mib=startup_limit)) + '\n')
        if ready:
            return
        if (os.getenv('MX_GPU_UNLIMITED_RETRIES') != '1' and
                time.monotonic() - start > policy['max_idle_wait_s']):
            raise RuntimeError('no idle TP group; no new GPU process was launched')
        time.sleep(policy['interval_s'])


def guarded_command(cell, folder, argv, guarded=False, cap=900):
    if not guarded:
        return cell._unsegmented_command(folder, argv, guarded=False, cap=cap)
    folder = Path(folder)
    attempts = []
    while True:
        execution = folder if not attempts else folder / f'preflight_retry{len(attempts)}'
        wait_idle(cell, execution, cell._cohort_root)
        process = cell._unsegmented_command(execution, argv, guarded=True, cap=cap)
        guard_path = execution / 'guard/guard_result.json'
        guard = json.loads(guard_path.read_text())
        pgid = execution / 'guard/pgid'
        child = int(pgid.read_text()) if pgid.exists() else None
        item = dict(process=ref(execution / 'PROCESS.json'), guard=ref(guard_path),
                    child_pid=child, pgid=ref(pgid) if child is not None else None)
        attempts.append(item)
        if process['returncode'] == 75 and guard['code'] == 75 and child is None and 'preflight' in guard['reason']:
            continue
        manifest = folder / 'GUARDED_ATTEMPTS.json'
        save(manifest, dict(kind='mx7_guarded_process', producer=ref(__file__), attempts=attempts,
                            command=list(map(str, argv))))
        return dict(process, guard_manifest=ref(manifest), child_pid=child)


def output_job(cell, folder, role, output, command):
    passed = folder / (role + '_GUARD_PASSED.json')
    if passed.exists():
        run = json.loads(passed.read_text())
        data = validate_run(run['manifest'])
        target = Path(run['output'])
        if data['command'] != list(map(str, command(target))):
            raise ValueError('retained guarded process command differs')
        return target, run
    failures = []
    unlimited = os.getenv('MX_GPU_UNLIMITED_RETRIES') == '1'
    for number in itertools.count() if unlimited else (0, 1):
        suffix = f'_retry{number}' if number else ''
        target = output.with_name(output.stem + suffix + output.suffix)
        execution = folder / (role + '_execution' + suffix)
        argv = list(map(str, command(target)))
        process = cell.run_command(execution, argv, guarded=True)
        run = dict(role=role, output=str(target), manifest=process['guard_manifest'])
        if process['returncode'] == 0:
            data = validate_run(run['manifest'])
            run['PID'] = data['attempts'][-1]['child_pid']
            run['failed_runs'] = failures
            save(passed, run)
            return target, run
        guard = json.loads(checked(json.loads(checked(run['manifest']).read_text())['attempts'][-1]['guard']).read_text())
        for diagnostic in (target, target / 'check.json', target / 'measure/measurements.json'):
            if diagnostic.is_file() and diagnostic.suffix == '.json':
                value = json.loads(diagnostic.read_text())
                if any(value.get(key) is False for key in ('pass', 'rank_tokens_identical', 'timed_tokens_identical')):
                    raise RuntimeError('a numerical/token failure cannot be hidden by guard recovery: ' + str(diagnostic))
        if process['returncode'] == 75 and guard['code'] == 75:
            failures.append(run)
            continue
        raise RuntimeError('check/timing process failed; numerical failures require diagnosis: ' + str(execution))
    save(folder / (role + '_FAILED.json'), dict(runs=failures))
    raise RuntimeError('guard interruption repeated after one fresh-process redo: ' + str(folder / role))


def correctness(cell, folder, fresh=2):
    if fresh not in (2, 5):
        raise ValueError('candidate/final correctness needs exactly2/5 fresh C-2 processes')
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    if (folder / 'QUALIFICATION.json').exists():
        previous = validate_checks(ref(folder / 'QUALIFICATION.json'))
        if previous['fresh_processes'] != fresh or previous['batch'] != cell.a.batch or previous['tp'] != cell.a.tp:
            raise ValueError('retained correctness scope differs')
        return dict(C1=previous['C1'], C2=previous['C2'], qualification=ref(folder / 'QUALIFICATION.json'))
    so = folder.parent / 'plan.so'
    devices = ','.join(map(str, range(cell.a.tp)))
    check = [sys.executable, '-m', 'tilemega.serving.check_tp', '--model', cell.a.model,
             '--decode-so', so, '--prompt-ids', cell.a.prompt_ids, '--batch', cell.a.batch,
             '--devices', devices]
    native, run = output_job(cell, folder, 'c1_native', folder / 'c1_tokens',
                            lambda out: check + ['--steps', '512', '--out', out])
    native_check = json.loads((native / 'check.json').read_text())
    if not native_check.get('pass', True) or not native_check['rank_tokens_identical'] or native_check['PID'] != run['PID']:
        raise RuntimeError('native C-1 rank/PID consistency failed')
    runs = [run]
    hf, run = output_job(cell, folder, 'c1_hf', folder / 'hf_check.json', lambda out: [
        sys.executable, '-m', 'tilemega.serving.hf_check', '--model', cell.a.model,
        '--prompt-ids', cell.a.prompt_ids, '--generated', native / 'tokens.json',
        '--out', out, '--skip-free-greedy'])
    if not json.loads(hf.read_text())['pass']:
        raise RuntimeError('C-1 failure; diagnose BF16 AR under MX7 B6: ' + str(hf))
    runs.append(run)
    checks, reference, pids = [], None, []
    for number in range(fresh):
        def command(out):
            argv = check + ['--steps', '1024', '--out', out]
            return argv + ['--reference', reference] if reference is not None else argv
        out, run = output_job(cell, folder, f'c2_{number}', folder / f'c2_process{number}', command)
        path = out / 'check.json'
        value = json.loads(path.read_text())
        if not value['pass'] or not value['rank_tokens_identical'] or value['PID'] != run['PID']:
            raise RuntimeError('C-2 / rank / PID consistency failed')
        if reference is None:
            reference = out / 'tokens.json'
        if sha(out / 'tokens.json') != sha(reference):
            raise RuntimeError('C-2 token sequences differ across fresh processes')
        pids.append(value['PID'])
        checks.append(ref(path))
        runs.append(run)
    if len(set(pids)) != fresh:
        raise RuntimeError('C-2 requires distinct fresh processes')
    result = dict(C1=ref(hf), C2=checks)
    qualification = folder / 'QUALIFICATION.json'
    save(qualification, dict(kind='mx7_individual_check_cohort', producer=ref(__file__), binary=ref(so),
                            native_C1=ref(native / 'check.json'), fresh_processes=fresh,
                            tp=cell.a.tp, batch=cell.a.batch, group=cell.a.group,
                            artifact_id=json.loads(Path(str(so) + '.identity.json').read_text())['artifact_id'],
                            runs=runs, **result))
    return dict(result, qualification=ref(qualification))


def validate_checks(value):
    data = json.loads(checked(value).read_text())
    if data['kind'] != 'mx7_individual_check_cohort' or data['fresh_processes'] not in (2, 5):
        raise ValueError('invalid independent-check qualification')
    checked(data['producer'])
    checked(data['binary'])
    runs = data['runs']
    if len(runs) != data['fresh_processes'] + 2:
        raise ValueError('incomplete C-1/C-2 process cohort')
    validated = [validate_run(run['manifest']) for run in runs]
    expected_group = list(map(int, data['group'].split(',')))
    if any(list(map(int, json.loads(checked(run['attempts'][-1]['guard']).read_text())['gpu_indices'])) != expected_group for run in validated):
        raise ValueError('guarded checks used a different physical GPU group')
    pids = [run['attempts'][-1]['child_pid'] for run in validated]
    if not json.loads(checked(data['C1']).read_text())['pass']:
        raise ValueError('HF C-1 did not pass')
    native = json.loads(checked(data['native_C1']).read_text())
    if (not native['rank_tokens_identical'] or native['PID'] != pids[0] or native['steps'] != 512 or
            native['execution_identity']['artifact_id'] != data['artifact_id']):
        raise ValueError('native C-1 qualification differs from its guarded artifact')
    values = [json.loads(checked(item).read_text()) for item in data['C2']]
    for item, value in zip(data['C2'], values):
        if sha(checked(item).parent / 'tokens.json') != value['token_sha256']:
            raise ValueError('C-2 token payload changed')
    if (len(values) != data['fresh_processes'] or len(set(pids[2:])) != len(values) or
            [item['PID'] for item in values] != pids[2:] or
            len({item['token_sha256'] for item in values}) != 1 or
            not all(item['pass'] and item['rank_tokens_identical'] and item['steps'] == 1024 and
                    item['execution_identity']['artifact_id'] == data['artifact_id'] and
                    item['tp'] == data['tp'] and item['batch'] == data['batch'] for item in values)):
        raise ValueError('independent C-2 qualification failed')
    return data


def measure(cell, folder):
    folder = Path(folder)
    records, processes, runs, values = [], [], [], []
    for number in (1, 2):
        out, run = output_job(cell, folder, f'timing{number}', folder / f'timing_process{number}', lambda target: [
            sys.executable, '-m', 'tilemega.serving.measure_tp', '--model', cell.a.model,
            '--decode-so', folder / 'plan.so', '--prompt-ids', cell.a.prompt_ids,
            '--batch', cell.a.batch, '--devices', ','.join(map(str, range(cell.a.tp))),
            '--max-new-tokens', '1024', '--warmup', '1', '--repeats', '2',
            '--policy', cell._cohort_root / 'configs/bench/measurement_policy_c550.json', '--out', target / 'measure'])
        path = out / 'measure/measurements.json'
        value = json.loads(path.read_text())
        if not value['timed_tokens_identical'] or not value['rank_tokens_identical']:
            raise RuntimeError('timed C-2 / rank consistency failed')
        manifest = validate_run(run['manifest'])
        process_path = folder / (f'timing{number}_qualified_process.json')
        evidence = dict(PID=run['PID'], returncode=0, argv=manifest['command'], guard_manifest=run['manifest'])
        if process_path.exists():
            if json.loads(process_path.read_text()) != evidence:
                raise ValueError('retained timing process evidence differs')
        else:
            save(process_path, evidence)
        records.append(ref(path))
        processes.append(ref(process_path))
        runs.append(run)
        values.append(value['tpot_seconds'] * 1e9)
    if len({run['PID'] for run in runs}) != 2:
        raise RuntimeError('timing requires two distinct fresh processes')
    return dict(fresh_process_ns=values, measurements=records, processes=processes,
                independent_guards=[run['manifest'] for run in runs], producer=ref(__file__))


def attach(cell, root):
    cell._cohort_root = Path(root)
    cell._unsegmented_command = cell.run_command
    cell.run_command = MethodType(guarded_command, cell)
    cell.correctness = MethodType(correctness, cell)
    cell.measure = MethodType(measure, cell)
    cell.output_job = MethodType(output_job, cell)


def load_and_attach(path, cell, root):
    import importlib.util
    path = Path(path)
    spec = importlib.util.spec_from_file_location('mx7_frozen_process_cohorts', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.attach(cell, root)
    return ref(path)


def seal_and_attach(path, cell, root):
    import shutil
    path = Path(path)
    if path.exists():
        raise ValueError('preserve the existing process-cohort source archive')
    shutil.copy2(__file__, path)
    if sha(path) != sha(__file__):
        raise ValueError('process-cohort source archive differs')
    return load_and_attach(path, cell, root)
