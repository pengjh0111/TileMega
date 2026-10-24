#!/usr/bin/env python3
"""SM90R3 LLM process jobs: guarded queues, fresh-process runs and decisions.

Protocols (prompt §2): pair = A/B alternation; sel = A B C D D C B A;
final = repeated fresh processes of one arm; corr = untimed C-1/C-2/HF64.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

from sm90d1_common import ROOT, PYTHON, read, sha, write

POLICY = ROOT / 'configs/bench/sm90r2_guard_policy.json'
PCI = {0: '0f', 1: '10', 2: '41', 3: '44', 4: '86', 5: '87', 7: 'b7'}
TIMING_GPUS = (0, 1, 2, 3, 4, 7)
MODELS = {
    'qwen3_0_6b': ('configs/serving/sm90d1/inputs/qwen3_0_6b', 'configs/prompts/qwen_isl64.json', 32),
    'qwen3_1_7b': ('configs/serving/sm90d1/inputs/qwen3_1_7b', 'configs/prompts/qwen_isl64.json', 32),
    'llama3_2_1b': ('configs/serving/sm90d1/inputs/llama3_2_1b', 'configs/prompts/llama3_isl64.json', 32),
    'llama3_2_3b': ('configs/serving/sm90d1/inputs/llama3_2_3b', 'configs/prompts/llama3_isl64.json', 32),
    'llama3_1_8b': ('configs/serving/sm90d1/inputs/llama3_1_8b', 'configs/prompts/llama3_isl64.json', 40),
}


def processes(model):
    return 5 if model in ('llama3_1_8b', 'qwen3_30b_a3b') else 3


def order(kind, labels, model):
    if kind == 'pair':
        a, b = labels
        return [a, b] * processes(model)
    if kind == 'sel':
        return list(labels) + list(reversed(labels))
    if kind == 'final':
        return [labels[0]] * processes(model)
    if kind == 'corr':
        return [labels[0]] * 5
    raise ValueError(kind)


def environment(gpu, concurrency):
    node = 0 if gpu < 4 else 1
    env = dict(CUDA_VISIBLE_DEVICES=str(gpu), TILEMEGA_DEVICE_INDEX=str(gpu),
               TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{gpu}.lock', TILEMEGA_NUMA_NODE=str(node),
               TILEMEGA_GPU_PID_MODE='namespace_limited', PYTHONPATH=str(ROOT / 'python'),
               OMP_NUM_THREADS='8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if concurrency == 'trial':
        env.update(TILEMEGA_NUMA_CONCURRENCY_TRIAL='1', TILEMEGA_NUMA_CONCURRENCY_VERIFIED='0')
    elif concurrency == 'verified':
        env.update(TILEMEGA_NUMA_CONCURRENCY_VERIFIED='1')
    return env


def queue(a):
    spec_path = a.spec.resolve()
    spec = read(spec_path)
    gpu, kind, model = spec['gpu'], spec['kind'], spec['model']
    timing = kind != 'corr'
    if gpu == 6 or (timing and gpu not in TIMING_GPUS):
        raise ValueError('GPU6 excluded; GPU5 functional-only')
    labels = list(spec['arms'])
    if kind == 'pair' and len(labels) != 2 or kind == 'sel' and len(labels) < 2 or \
            kind in ('final', 'corr') and len(labels) != 1:
        raise ValueError('arm count does not match the protocol')
    for arm in spec['arms'].values():
        for phase in ('decode', 'prefill'):
            if not Path(arm[phase]).is_file():
                raise FileNotFoundError(arm[phase])
        arm['decode_sha256'], arm['prefill_sha256'] = sha(arm['decode']), sha(arm['prefill'])
    write(spec_path, spec)
    sequence = spec.get('order') or order(kind, labels, model)
    policy = read(POLICY)
    node = 0 if gpu < 4 else 1
    uuid = policy['idle_power_mapping_provenance']['uuid_by_gpu'][str(gpu)]
    jobs, previous = [], list(a.after if a.after is not None else spec.get('after', []))
    if a.cohort_gpu is not None:
        gpu, node = a.cohort_gpu, 0 if a.cohort_gpu < 4 else 1
        uuid = policy['idle_power_mapping_provenance']['uuid_by_gpu'][str(gpu)]
        concurrent = [i for i, label in enumerate(sequence) if label == spec['dt0']['concurrent_arm']]
        sequence = [('cohort', i) for i in concurrent]
    for index, label in enumerate(sequence):
        action = 'measure'
        if isinstance(label, tuple):
            action, index = label
            label = 'load'
        name = f"{spec['name']}-{index:02d}-{label}" + (f'-g{gpu}' if action == 'cohort' else '')
        command = ['numactl', f'--cpunodebind={node}', PYTHON,
                   str(ROOT / 'scripts/run_r16_with_device_identity.py'), '--guard-policy', str(POLICY),
                   '--out', '{attempt_dir}/job', '--expected-uuid', uuid,
                   '--expected-pci', f'0000:{PCI[gpu]}:00.0', '--numa-node', str(node), '--',
                   PYTHON, str(Path(__file__).resolve()), action, '--spec', str(spec_path),
                   '--arm', label, '--fresh', str(index), '--out', '{attempt_dir}/job']
        jobs.append(dict(name=name, gpu=True, timing=timing, priority=spec.get('priority', 10),
                         timeout_s=spec.get('timeout_s', 5400 if not timing else 3600),
                         needs_free_mib=32768, cwd=str(ROOT), after=previous,
                         env=environment(gpu, spec.get('concurrency', 'serial')), command=command))
        previous = [name]
    suffix = '' if a.cohort_gpu is None else f'_g{a.cohort_gpu}'
    write(a.queue_dir / f"queue_{spec['name']}{suffix}.json", jobs)
    print(json.dumps(dict(queue=str(a.queue_dir / f"queue_{spec['name']}{suffix}.json"), jobs=len(jobs))))


def measure(a):
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise ValueError('GPU work requires the scheduler guard')
    spec = read(a.spec)
    arm = spec['arms'][a.arm]
    for phase in ('decode', 'prefill'):
        if sha(arm[phase]) != arm[phase + '_sha256']:
            raise ValueError('artifact changed after queueing: ' + phase)
    model_dir, prompt, gib = MODELS[spec['model']]
    a.out.mkdir(parents=True, exist_ok=True)
    child = [PYTHON, str(Path(__file__).resolve()), 'child', '--spec', str(a.spec.resolve()),
             '--arm', a.arm, '--fresh', str(a.fresh), '--out', str(a.out.resolve())]
    command = [PYTHON, '-m', 'tilemega.serving.random_checkpoint', '--config', str(ROOT / model_dir),
               '--out', str(a.out / 'checkpoint'), '--seed', '0', '--max-gib', str(gib), '--', *child]
    record = dict(spec=str(a.spec.resolve()), spec_name=spec['name'], kind=spec['kind'], arm=a.arm,
                  fresh=a.fresh, model=spec['model'], batch=spec['batch'],
                  decode_sha256=arm['decode_sha256'], prefill_sha256=arm['prefill_sha256'],
                  runner_sha256=sha(__file__), command=command)
    write(a.out / 'sm90r3-process.json', record)
    dt0 = spec.get('dt0')
    concurrent = dt0 is not None and a.arm == dt0['concurrent_arm']
    if concurrent:
        markers = Path(dt0['markers'])
        (markers / f'go-{a.fresh}').touch()
        wait_for([markers / f'running-{a.fresh}-{g}' for g in dt0['cohort_gpus']], 1800)
    elif dt0 is not None:
        markers = Path(dt0['markers'])
        slots = [int(p.name.split('-')[1]) for p in markers.glob('go-*')]
        wait_for([markers / f'done-{k}-{g}' for k in slots for g in dt0['cohort_gpus']], 1800)
    try:
        status = subprocess.call(command, cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT / 'python')))
    finally:
        if concurrent:
            (markers / f'stop-{a.fresh}').touch()
    record['exit_code'] = status
    write(a.out / 'sm90r3-process.json', record)
    return status


def wait_for(paths, timeout):
    import time
    begin = time.time()
    while not all(p.exists() for p in paths):
        if time.time() - begin > timeout:
            raise TimeoutError('DT-0 cohort markers did not appear')
        time.sleep(1)


def cohort(a):
    """Keep this card busy with fresh timing processes during one DT-0 slot."""
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise ValueError('GPU work requires the scheduler guard')
    spec = read(a.spec)
    markers, gpu = Path(spec['dt0']['markers']), int(os.environ['TILEMEGA_DEVICE_INDEX'])
    wait_for([markers / f'go-{a.fresh}'], 7200)
    (markers / f'running-{a.fresh}-{gpu}').touch()
    load = Path(spec['dt0']['cohort_specs'][str(gpu)])
    count = 0
    while not (markers / f'stop-{a.fresh}').exists():
        out = a.out / f'load{count}'
        status = measure(argparse.Namespace(spec=load, arm='load', fresh=count, out=out))
        if status:
            return status
        count += 1
    write(a.out / 'cohort.json', dict(slot=a.fresh, gpu=gpu, processes=count))
    (markers / f'done-{a.fresh}-{gpu}').touch()
    return 0


def child(a):
    spec = read(a.spec)
    arm = spec['arms'][a.arm]
    model_dir, prompt, _ = MODELS[spec['model']]
    checkpoint = os.environ['TILEMEGA_RANDOM_CHECKPOINT']
    out = a.out / 'measurements'
    command = [PYTHON, '-m', 'tilemega.serving.r16_run', '--prefill', arm['prefill'], '--decode', arm['decode'],
               '--prompt-ids', str(ROOT / prompt), '--batch', str(spec['batch']), '--max-tokens', '1024',
               '--warmup', '1', '--repeats', '3', '--mode', 'L1', '--decode-loop', '1',
               '--prefill-mode', 'L1', '--round-index', str(a.fresh), '--out', str(out)]
    correctness = spec.get('correctness', {})
    if correctness.get('c2'):
        command.append('--check-step-loop')
    if correctness.get('c1'):
        command += ['--hf-check', '--hf-positions', str(correctness['c1'])]
    status = subprocess.call(command, cwd=ROOT)
    if status or not correctness.get('hf64'):
        return status
    reference = read(out / 'reference_tokens.json')
    prefix = out / 'hf64_tokens.json'
    prefix.write_text(json.dumps([row[:64] for row in reference]) + '\n')
    return subprocess.call([PYTHON, '-m', 'tilemega.serving.hf_check', '--model', checkpoint,
                            '--prompt-ids', str(ROOT / prompt), '--generated', str(prefix),
                            '--out', str(out / 'hf64.json')], cwd=ROOT)


def span(values):
    return (max(values) - min(values)) / statistics.median(values) * 100


def job_result(state, name):
    row = state[name]
    if row['status'] != 'done':
        raise ValueError(f'{name} is {row["status"]}')
    folder = Path(row['out'])
    guard = read(folder / 'guard_result.json')
    if guard['code']:
        raise ValueError(f'{name} guard rejected the attempt')
    paths = list((folder / 'job/measurements').rglob('measurements.json'))
    if len(paths) != 1:
        raise ValueError(f'{name} lacks exactly one measurement')
    value = read(paths[0])
    process = read(folder / 'job/sm90r3-process.json')
    if value['configuration']['decode_sha256'] != process['decode_sha256']:
        raise ValueError(f'{name} timed another decode artifact')
    result = dict(job=name, tpot_ms=value['tpot_seconds'] * 1e3, ttft_ms=value['ttft_seconds'] * 1e3,
                  e2e_ms=value['e2e_seconds'] * 1e3, tokens_per_s=value['output_tokens_per_second'],
                  gpu=value['gpu_index'], tokens=str(folder / 'job/measurements/reference_tokens.json'))
    for check in ('c1', 'c2', 'hf64'):
        path = folder / 'job/measurements' / f'{check}.json'
        if path.exists():
            result[check] = read(path)
    return result


def collect(a):
    spec = read(a.spec)
    state = read(a.state)
    sequence = spec.get('order') or order(spec['kind'], list(spec['arms']), spec['model'])
    rows = [dict(arm=label, **job_result(state, f"{spec['name']}-{i:02d}-{label}"))
            for i, label in enumerate(sequence)]
    arms = {}
    for row in rows:
        arms.setdefault(row['arm'], []).append(row)
    summary = {label: dict(tpot_ms=statistics.median(r['tpot_ms'] for r in v),
                           ttft_ms=statistics.median(r['ttft_ms'] for r in v),
                           e2e_ms=statistics.median(r['e2e_ms'] for r in v),
                           tokens_per_s=statistics.median(r['tokens_per_s'] for r in v),
                           range_pct=span([r['tpot_ms'] for r in v]) if len(v) > 1 else 0.0,
                           processes=len(v)) for label, v in arms.items()}
    result = dict(schema=1, spec=spec, rows=rows, arms=summary, collector_sha256=sha(__file__))
    kind = spec['kind']
    if kind == 'pair':
        a_label, b_label = list(spec['arms'])
        ref, cand = summary[a_label], summary[b_label]
        threshold = max(0.5, 2 * max(ref['range_pct'], cand['range_pct']))
        change = (cand['tpot_ms'] / ref['tpot_ms'] - 1) * 100
        result.update(change_pct=change, threshold_pct=threshold,
                      decision='indistinguishable' if abs(change) <= threshold else
                      'faster' if change < 0 else 'slower')
    elif kind == 'sel':
        ranked = sorted(summary, key=lambda k: summary[k]['tpot_ms'])
        best, second = ranked[0], ranked[1]
        threshold = max(0.5, 2 * max(summary[best]['range_pct'], summary[second]['range_pct']))
        gap = (summary[second]['tpot_ms'] / summary[best]['tpot_ms'] - 1) * 100
        winner = best
        predicted = spec.get('predicted_ns', {})
        if gap <= threshold and predicted:
            winner = min((best, second), key=lambda k: predicted[k])
        result.update(measured_order=ranked, top_gap_pct=gap, threshold_pct=threshold, winner=winner,
                      tie_broken_by_prediction=winner != best)
    elif kind == 'corr':
        tokens = [read(r['tokens']) for r in rows]
        result.update(c1_pass=all(r.get('c1', {}).get('pass', False) for r in rows) if
                      spec.get('correctness', {}).get('c1') else None,
                      c2_pass=all(r.get('c2', {}).get('pass', False) for r in rows) if
                      spec.get('correctness', {}).get('c2') else None,
                      hf64_pass=all(r.get('hf64', {}).get('first_hf_greedy_divergence') is not None and
                                    all(d is None or d >= 64 for d in r['hf64']['first_hf_greedy_divergence'])
                                    for r in rows) if spec.get('correctness', {}).get('hf64') else None,
                      fresh_processes=len(rows),
                      processes_token_equal=sum(t == tokens[0] for t in tokens))
        result['pass'] = all(v in (None, True) for v in (result['c1_pass'], result['c2_pass'],
                                                       result['hf64_pass'])) and \
            result['processes_token_equal'] == len(rows)
    write(a.out, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('spec', 'rows')}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('queue', 'measure', 'child', 'collect', 'cohort'))
    p.add_argument('--cohort-gpu', type=int)
    p.add_argument('--after', nargs='*')
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--queue-dir', type=Path)
    p.add_argument('--arm')
    p.add_argument('--fresh', type=int, default=0)
    p.add_argument('--out', type=Path)
    p.add_argument('--state', type=Path)
    a = p.parse_args()
    if a.action == 'queue':
        return queue(a)
    if a.action == 'measure':
        return measure(a)
    if a.action == 'child':
        return child(a)
    if a.action == 'cohort':
        return cohort(a)
    return collect(a)


if __name__ == '__main__':
    sys.exit(main() or 0)
