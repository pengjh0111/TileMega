"""Refine measured TP8 depths; preserve the frozen three-system baseline matrix."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import itertools
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from types import SimpleNamespace

from tilemega.build.identity import verify
from tilemega.mx5_resources import inspect
from tilemega.mx5_plan_occupancy import validate_inputs
from tilemega.tp.selection_tp import STAGES, k_best, pin, stage_lists
from tilemega.tp.solve_tp import Cell, ROOT, ref, save
from tilemega.tp.process_cohorts import seal_and_attach, validate_run, validate_checks

REGULAR = ('qkv', 'o', 'gate_up', 'down', 'head')


def read(path):
    return json.loads(Path(path).read_text())


def family_matches(choice, config):
    for stage in REGULAR:
        key = choice['stages'][stage]['key'];old = config[stage]
        if (key['tn'], key['wn'], bool(key['part'])) != (old['tn'], old['wn'], old['part'] == 'kTileRange'):
            return False
    attention = choice['stages']['attention']['key']
    return attention['b_min'] == config['b_min']


def refinements(catalog, batch, config, limit=25000):
    lists = stage_lists(catalog['rows'], 'qwen3_1_7b_tp8', batch, catalog['tp_constants'])
    choices = []
    seen = set()
    for choice in itertools.islice(k_best(lists, 28), limit):
        if not family_matches(choice, config):
            continue
        keys = choice['stages']
        signature = tuple(keys[stage]['key']['d'] for stage in REGULAR) + (keys['attention']['key']['prefetch'],)
        if signature not in seen:
            choices.append(choice);seen.add(signature)
    selected = []
    signatures = set()

    def add(label, predicate):
        found = next((choice for choice in choices if predicate(choice)), None)
        if found is None:
            return
        signature = tuple(found['ranks'].values())
        if signature not in signatures:
            selected.append(dict(name=label, choice=found));signatures.add(signature)

    depths = lambda c: [c['stages'][stage]['key']['d'] for stage in REGULAR]
    for depth in (2, 3, 4):
        add(f'uniform_d{depth}', lambda c, d=depth: all(v == d for v in depths(c)))
    for depth in (2, 3):
        add(f'minimum_d{depth}', lambda c, d=depth: min(depths(c)) >= d)
    for stage in ('qkv', 'o', 'gate_up', 'down'):
        add(stage + '_different_depth', lambda c, s=stage: c['stages'][s]['key']['d'] != config[s]['d'])
    add('attention_prefetch1', lambda c: c['stages']['attention']['key']['prefetch'] == 1)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/mx7/tp8_opt/depth')
    parser.add_argument('--original', type=Path, default=ROOT / 'runs/mx7/tp8_queue')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    out = args.out.resolve();out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / 'controller.py')
    original = args.original.resolve()
    accepted = read(original / 'ACCEPTED.json')
    catalog = read(accepted['costs']['path'])
    frozen_baselines = [ref(p) for p in (original / 'results').glob('*.json')
                        if not p.name.startswith('tilemega_')]
    save(out / 'FROZEN_BASELINES.json', dict(aggregate=accepted['table'], records=frozen_baselines,
        policy='reuse original TP8 baselines; no external baseline process is launched'))
    export = Path(read(original / 'jobs/export/ACCEPTED.json')['folder']) / 'export'
    bridge = export / 'qwen3_1_7b_tp8_decode/bridge.json'
    cell = Cell.__new__(Cell)
    cell.a = SimpleNamespace(group='0,1,2,3,4,5,6,7', startup_hidden_mib=None, tp=8,
        model=Path('/root/models/qwen3_1_7b_rand'), prompt_ids=ROOT / 'configs/prompts/qwen_isl64.json')
    cell.env = dict(os.environ, PYTHONPATH=str(ROOT / 'python'), NEEDS_MIB='12288',
        MX_GPU_WAIT_IDLE='1', MX_GPU_IDLE_WAIT_S='-1', MX_GPU_UNLIMITED_RETRIES='1',
        MX_GPU_JOB_LOG=str(out / 'gpu_jobs.log'), TILEMEGA_GPU_JOB_LOG=str(out / 'gpu_jobs.log'))
    cell.cpu_env = dict(cell.env, CUDA_VISIBLE_DEVICES='', MACA_VISIBLE_DEVICES='')
    seal_and_attach(out / 'process_cohorts.py', cell, ROOT)
    previous, prepared = {}, {}
    work = []
    for batch in (16, 1, 2, 4, 8):
        receipt = read(original / f'jobs/select_B{batch}/ACCEPTED.json')
        result = read(Path(receipt['folder']) / 'selection/results.json');previous[batch] = result
        original_pin = read(result['selected_pin']['path'])
        scope = out / f'B{batch}';scope.mkdir()
        candidates = [dict(name='control', binary=result['resources']['binary']['path'],
            binary_sha256=result['resources']['binary']['sha256'], pin=result['selected_pin'],
            build_folder=str(Path(result['resources']['binary']['path']).parent))]
        for item in refinements(catalog, batch, original_pin['mx_decode_v2']):
            folder = scope / item['name'];folder.mkdir()
            coordinate = pin(item['choice'], accepted['costs']['sha256'], 8, 'manual')
            coordinate['evidence'].update(algorithm='global measured k-best depth refinement; ordinary paired E2E selection',
                same_weight_layout=True, original_selected_pin=result['selected_pin'])
            save(folder / 'pin.json', coordinate)
            work.append((batch, item, folder))
        prepared[batch] = candidates

    def compile_one(job):
        batch, item, folder = job
        binary = folder / 'plan.so'
        argv = [ROOT / 'build/tools/tilemega', 'compile', bridge, binary, '--serving', 'decode',
            '--emit', 'serving', '--frontend', 'decoder', '--batch', batch, '--prompt-len', '64',
            '--capacity', '1088', '--past-range', '0:1086', '--serving-tp', '8', '--runtime-target',
            ROOT / 'configs/targets/xcore1000.json', '--evaluate-configs', folder / 'pin.json',
            '--selection', 'predicted', '--watchdog', '0']
        with (folder / 'compile.log').open('x') as log:
            code = subprocess.run(list(map(str, argv)), cwd=ROOT, env=cell.cpu_env,
                                  stdout=log, stderr=subprocess.STDOUT).returncode
        record = dict(batch=batch, candidate=item, returncode=code, command=list(map(str, argv)))
        if not code:
            verify(binary)
            resources = inspect(binary, str(binary) + '.ptxas.log', 512)
            save(folder / 'actual_resources.json', resources);record['resources'] = ref(folder / 'actual_resources.json')
            if not resources['violations']:
                spec = dict(kind='mx5_plan_occupancy_inputs', schema_version=1, threads=512,
                    context_kind='native_identity', binary=ref(binary), generated_source=ref(str(binary) + '.cu'),
                    resources=ref(folder / 'actual_resources.json'), log=ref(str(binary) + '.ptxas.log'),
                    context=ref(str(binary) + '.identity.json'), source_snapshot=ref(str(binary) + '.source.json'),
                    manifest=ref(str(binary) + '.plan.json'))
                validate_inputs(spec);save(folder / 'query_inputs.json', spec)
                record['qualified_cpu'] = True;record['binary'] = ref(binary)
        save(folder / 'BUILD.json', record)
        print('depth build', batch, item['name'], record.get('qualified_cpu', False), flush=True)
        return job, record

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for (batch, item, folder), build in pool.map(compile_one, work):
            if build.get('qualified_cpu'):
                prepared[batch].append(dict(name=item['name'], binary=build['binary']['path'],
                    binary_sha256=build['binary']['sha256'], pin=ref(folder / 'pin.json'), build_folder=str(folder),
                    global_candidate_rank=item['choice']['candidate_rank']))
    save(out / 'PREPARED.json', dict(candidates=prepared, costs=accepted['costs'], original=ref(original / 'ACCEPTED.json')))
    finals = []
    for batch in (16, 1, 2, 4, 8):
        scope = out / f'B{batch}';cell.a.batch = batch
        reference = Path(previous[batch]['final_correctness']['C2'][0]['path']).parent / 'tokens.json'
        inputs = scope / 'SCREEN_INPUTS.json'
        save(inputs, dict(batch=batch, model=str(cell.a.model), prompt_ids=str(cell.a.prompt_ids),
                          reference=str(reference), candidates=prepared[batch], timing_eligible=False))
        target, run = cell.output_job(scope, 'screen', scope / 'screen', lambda dst: [sys.executable,
            ROOT / 'experiments/mx7/tp8_depth_screen.py', '--inputs', inputs, '--out', dst])
        screened = read(target / 'results.json')['results']
        control = next(row for row in screened if row['candidate']['name'] == 'control')
        best = min(screened, key=lambda row: row['median_tpot_ms'])
        chosen = best if best['median_tpot_ms'] <= control['median_tpot_ms'] * .99 else control
        print('depth screen winner', batch, chosen['candidate']['name'], chosen['median_tpot_ms'], flush=True)
        finalists = [control['candidate']]
        if chosen['candidate']['name'] != 'control':
            finalists.append(chosen['candidate'])
        paired = {candidate['name']: [] for candidate in finalists}
        for number in (1, 2):
            for candidate in finalists:
                name = candidate['name']
                measured, trial = cell.output_job(scope, f'paired_{name}_{number}',
                    scope / f'paired_{name}_{number}', lambda dst, candidate=candidate: [sys.executable,
                        '-m', 'tilemega.serving.measure_tp', '--model', cell.a.model,
                        '--decode-so', candidate['binary'], '--prompt-ids', cell.a.prompt_ids,
                        '--batch', batch, '--devices', cell.a.group, '--max-new-tokens', '1024',
                        '--warmup', '1', '--repeats', '2', '--policy', ROOT / 'configs/bench/measurement_policy_c550.json',
                        '--out', dst / 'measure'])
                m = read(measured / 'measure/measurements.json')
                paired[name].append(dict(measurement=ref(measured / 'measure/measurements.json'), run=trial,
                                         tpot_ms=m['tpot_seconds'] * 1000))
        winner = finalists[0]
        if len(finalists) == 2:
            old_values = [r['tpot_ms'] for r in paired['control']]
            new_values = [r['tpot_ms'] for r in paired[finalists[1]['name']]]
            old_m, new_m = map(statistics.median, (old_values, new_values))
            threshold = max(.01, (max(old_values) - min(old_values)) / old_m,
                            (max(new_values) - min(new_values)) / new_m)
            if new_m < old_m * (1 - threshold):
                winner = finalists[1]
        selected_folder = scope / 'selected';selected_folder.mkdir()
        for source in Path(winner['build_folder']).glob('plan.so*'):
            (selected_folder / source.name).symlink_to(source.resolve())
        checks = cell.correctness(selected_folder / 'correctness', fresh=5)
        validate_checks(checks['qualification'])
        final = dict(batch=batch, selected=winner, paired=paired, checks=checks,
                     screen=ref(target / 'results.json'), timing=paired[winner['name']])
        save(scope / 'RESULT.json', final);finals.append(ref(scope / 'RESULT.json'))
        print('depth final accepted', batch, winner['name'], flush=True)
    for baseline in frozen_baselines:
        if ref(baseline['path']) != baseline:
            raise RuntimeError('frozen baseline reference changed')
    save(out / 'ACCEPTED.json', dict(results=finals, baseline_records=frozen_baselines,
                                    baseline_policy='no external remeasurement'))
    print('all five TP8 refinement scopes complete', flush=True)


if __name__ == '__main__':
    main()
