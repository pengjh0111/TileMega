"""Resume a qualified TP cell after one guard-invalidated timing cohort."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

from tilemega.tp.solve_tp import ROOT, check_qualification, ref, save, sha
from tilemega.tp.selection_tp import pin, select_measured


def checked(value):
    path = Path(value['path'])
    if sha(path) != value['sha256']:
        raise ValueError('changed archived evidence: ' + str(path))
    return path


def successful_guard(folder):
    process = json.loads((folder / 'PROCESS.json').read_text())
    guard = json.loads((folder / 'guard/guard_result.json').read_text())
    if process['returncode'] or guard['code']:
        raise ValueError('reused cohort did not pass its process and guard')


def completed_correctness(folder):
    successful_guard(folder / 'execution')
    hf = folder / 'hf_check.json'
    if not json.loads(hf.read_text())['pass']:
        raise ValueError('reused C-1 did not pass')
    paths = [folder / f'c2_process{i}/check.json' for i in range(2)]
    values = [json.loads(path.read_text()) for path in paths]
    if (len({v['PID'] for v in values}) != 2 or
            len({v['token_sha256'] for v in values}) != 1 or
            not all(v['pass'] and v['rank_tokens_identical'] and v['steps'] == 1024 for v in values)):
        raise ValueError('reused C-2 requires two exact fresh processes')
    return dict(C1=ref(hf), C2=list(map(ref, paths)))


def completed_timing(value):
    if json.loads(checked(value['guard']).read_text())['code']:
        raise ValueError('invalid timing guard cannot participate in selection')
    checked(value['driver'])
    processes = [json.loads(checked(p).read_text()) for p in value['processes']]
    if len(processes) != 2 or len({p['PID'] for p in processes}) != 2 or any(p['returncode'] for p in processes):
        raise ValueError('two successful fresh timing processes are required')
    for p in value['measurements']:
        measured = json.loads(checked(p).read_text())
        if not measured['timed_tokens_identical'] or not measured['rank_tokens_identical']:
            raise ValueError('timing token consistency failed')



def check_unlaunched_retry(retry):
    execution = retry / 'timing_execution/execution'
    if (execution / 'guard/pgid').exists() or list(retry.glob('timing_process*')):
        raise ValueError('a launched timing retry cannot be repeated after preflight')
    process_path = execution / 'PROCESS.json'
    if process_path.exists():
        process = json.loads(process_path.read_text())
        guard = json.loads((execution / 'guard/guard_result.json').read_text())
        if process['returncode'] != 75 or guard['code'] != 75 or 'preflight' not in guard['reason']:
            raise ValueError('only a guard preflight rejection leaves the single timing retry unused')


def resume(args):
    out = args.out.resolve()
    if (out / 'results.json').exists() or ((out / 'inputs/RESUME1.json').exists() and not (args.prelaunch_remedy or args.preflight_remedy)):
        raise ValueError('this cell is complete or its single retry has already started')
    sealed = out / 'inputs'
    if args.prelaunch_remedy:
        if not (sealed / 'RESUME1.json').exists():
            raise ValueError('a prelaunch failure receipt is required')
        for retry in out.glob('rank*/build/timing_retry1'):
            if (retry / 'timing_execution/execution/PROCESS.json').exists() or list(retry.glob('timing_process*')):
                raise ValueError('a launched retry cannot be repeated as a prelaunch remedy')
    if args.preflight_remedy:
        if not (sealed / 'RESUME1.json').exists():
            raise ValueError('an original single-retry receipt is required')
        for retry in out.glob('rank*/build/timing_retry1*'):
            check_unlaunched_retry(retry)
    original = json.loads((sealed / 'IDENTITY.json').read_text())
    for value in original['inputs']:
        checked(value)
    if sha(sealed / 'solve_tp.py') != original['runner']['sha256']:
        raise ValueError('archived solver source differs from the original runner')
    spec = importlib.util.spec_from_file_location('mx7_archived_solver', sealed / 'solve_tp.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT  # The archived file lives below runs/, not the package root.
    cell = module.Cell.__new__(module.Cell)
    cell.a, cell.out = args, out
    cell.env = dict(os.environ, PYTHONPATH=str(ROOT / 'python'), NEEDS_MIB='12288',
                    TILEMEGA_GPU_JOB_LOG=str(ROOT / 'runs/mx7/gpu_jobs.log'))
    cell.cpu_env = dict(cell.env, CUDA_VISIBLE_DEVICES='', MACA_VISIBLE_DEVICES='')
    cell.compiler = sealed / 'tilemega'
    cell.bridge = sealed / 'bridge.json'
    cell.catalog, _ = check_qualification(sealed / 'xcore1000_mx7_tp.json')
    cell.cost_sha = sha(sealed / 'xcore1000_mx7_tp.json')
    manifest = json.loads((sealed / 'manifest.json').read_text())
    if manifest['tp'] != args.tp or len(set(args.group.split(','))) != args.tp:
        raise ValueError('resumed TP scope differs from the sealed inputs')
    qualified = json.loads((out / 'QUALIFIED.json').read_text())
    candidates, rejected = qualified['qualified'], qualified['rejected']
    if len(candidates) != 4:
        raise ValueError('four already qualified candidates are required')
    failures = []
    for candidate in candidates:
        folder = Path(candidate['folder'])
        checked(candidate['build']['binary'])
        occupancy = json.loads(checked(candidate['build']['occupancy']).read_text())
        if (not occupancy['qualified'] or
                any(r['guard']['gpu_indices'] != list(map(int, args.group.split(','))) for r in occupancy['ranks'])):
            raise ValueError('candidate residency did not qualify')
        successful_guard(folder / 'occupancy')
        plan = json.loads(Path(str(checked(candidate['build']['binary'])) + '.plan.json').read_text())
        if plan['tp'] != args.tp or not (plan['batch_lo'] <= args.batch <= plan['batch_hi']):
            raise ValueError('resumed batch differs from the qualified artifact')
        candidate['correctness'] = completed_correctness(folder / 'correctness')
        process_path = folder / 'timing_execution/execution/PROCESS.json'
        if (folder / 'RESULT.json').exists():
            saved = json.loads((folder / 'RESULT.json').read_text())
            completed_timing(saved['timing'])
            candidate['timing'] = saved['timing']
        elif process_path.exists():
            process = json.loads(process_path.read_text())
            guard_path = process_path.parent / 'guard/guard_result.json'
            guard = json.loads(guard_path.read_text())
            if process['returncode'] != 75 or guard['code'] != 75 or 'interference' not in guard['reason']:
                raise ValueError('only one guard-interference timing anomaly may be retried here')
            failures.append(dict(candidate_rank=candidate['choice']['candidate_rank'],
                                 process=ref(process_path), guard=ref(guard_path)))
    if len(failures) != 1:
        raise ValueError('exactly one excluded timing cohort is required')
    suffix = f'_preflight{args.preflight_remedy}' if args.preflight_remedy else ('_prelaunch_remedy' if args.prelaunch_remedy else '')
    archived = sealed / f'resume_tp{suffix}.py'
    shutil.copy2(__file__, archived)
    receipt = sealed / (f'RESUME1_PREFLIGHT{args.preflight_remedy}.json' if args.preflight_remedy else ('RESUME1_PREFLIGHT_REMEDY.json' if args.prelaunch_remedy else 'RESUME1.json'))
    save(receipt, dict(classification='MX7 B5 single timing anomaly', runner=ref(archived),
                       original_runner=original['runner'], group=args.group, tp=args.tp, batch=args.batch,
                       excluded_timing=failures, retry_limit=1,
                       policy='retain successful build/correctness/timing evidence; repeat invalid timing only'))
    for candidate in candidates:
        if 'timing' in candidate:
            continue
        folder = Path(candidate['folder'])
        if (folder / 'timing_execution').exists():
            # Sidecars remain beside the retry symlink, preserving the original artifact identity.
            retry = folder / f'timing_retry1{suffix}'
            retry.mkdir()
            for source in folder.glob('plan.so*'):
                (retry / source.name).symlink_to(source.resolve())
            candidate['timing'] = cell.measure(retry)
        else:
            candidate['timing'] = cell.measure(folder)
        completed_timing(candidate['timing'])
        save(folder / 'RESULT.json', candidate)
    selection = select_measured([c['choice']['predicted_ns'] for c in candidates],
                                [c['timing']['fresh_process_ns'] for c in candidates])
    selected = candidates[selection['selected']]
    final_pin = pin(selected['choice'], cell.cost_sha, args.tp, 'manual')
    final_pin['evidence'].update(selection=selection, resume=ref(receipt),
        candidates=[dict(candidate_rank=c['choice']['candidate_rank'],
                         predicted_ns=c['choice']['predicted_ns'], **c['timing']) for c in candidates])
    selected_path = out / 'selected_pin.json'
    save(selected_path, final_pin)
    final = out / 'final'
    resources = cell.compile(final, selected_path)
    if resources['status'] != 'qualified':
        raise RuntimeError('selected rebuild did not qualify')
    correctness = cell.correctness(final / 'correctness', fresh=5)
    timing = cell.measure(final)
    result = dict(tp=args.tp, batch=args.batch, group=args.group, selection=selection,
                  selected_pin=ref(selected_path), cost_table_sha256=cell.cost_sha,
                  candidates=candidates, rejected=rejected, resources=resources,
                  final_correctness=correctness, final_timing=timing, resume=ref(receipt))
    save(out / 'results.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prelaunch-remedy', action='store_true')
    parser.add_argument('--preflight-remedy', type=int, default=0)
    parser.add_argument('--tp', type=int, choices=(2, 4), required=True)
    parser.add_argument('--batch', type=int, choices=(1, 2, 4, 8, 16), required=True)
    parser.add_argument('--group', required=True)
    parser.add_argument('--target', type=Path, default=ROOT / 'configs/targets/xcore1000.json')
    parser.add_argument('--model', type=Path, default=Path('/root/models/qwen3_1_7b_rand'))
    parser.add_argument('--prompt-ids', type=Path, default=ROOT / 'configs/prompts/qwen_isl64.json')
    print(json.dumps(resume(parser.parse_args()), indent=2), flush=True)


if __name__ == '__main__':
    main()
