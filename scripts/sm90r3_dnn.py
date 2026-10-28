#!/usr/bin/env python3
"""SM90R3 DNN forward process jobs: guarded P-DNN selection and closeout.

Protocols (prompt §2.3 and §10 N-1 item 8): sel = the single-process P-DNN
over the compiled top-K in A B C D D C B A order, one fresh process per visit;
final = repeated fresh processes of the selected artifact. Correctness is the
DNN gate of `tilemega.dnn.check`, which runs untimed on the functional card.
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
TIMING_GPUS = (0, 1, 2, 3, 4, 5, 7)  # GPU5 timing authorized by the user (V)
WARMUP, ITERATIONS = 50, 200


def span(values):
    return (max(values) / min(values) - 1) * 100 if values and min(values) > 0 else 0.0


def order(kind, labels):
    if kind == 'sel':
        return list(labels) + list(reversed(labels))
    if kind == 'final':
        return [labels[0]] * 3
    raise ValueError(kind)


def environment(gpu):
    node = 0 if gpu < 4 else 1
    return dict(CUDA_VISIBLE_DEVICES=str(gpu), TILEMEGA_DEVICE_INDEX=str(gpu),
                TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{gpu}.lock', TILEMEGA_NUMA_NODE=str(node),
                TILEMEGA_GPU_PID_MODE='namespace_limited', PYTHONPATH=str(ROOT / 'python'),
                OMP_NUM_THREADS='8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')


def queue(a):
    """One guarded job per visit, chained in the protocol's order."""
    spec_path = a.spec.resolve()
    spec = read(spec_path)
    gpu, kind = spec['gpu'], spec['kind']
    if gpu == 6 or gpu not in TIMING_GPUS:
        raise ValueError('GPU6 excluded; timing runs on an authorized card')
    labels = list(spec['arms'])
    if kind == 'sel' and len(labels) < 2 or kind == 'final' and len(labels) != 1:
        raise ValueError('arm count does not match the protocol')
    for arm in spec['arms'].values():
        for key in ('library', 'inputs'):
            if not Path(arm[key]).is_file():
                raise FileNotFoundError(arm[key])
        arm['library_sha256'], arm['inputs_sha256'] = sha(arm['library']), sha(arm['inputs'])
        arm['root_commit'] = subprocess.check_output(
            ['git', '-C', str(Path(arm.get('root', ROOT))), 'rev-parse', 'HEAD'], text=True).strip()
    write(spec_path, spec)
    sequence = spec.get('order') or order(kind, labels)
    policy = read(POLICY)
    node = 0 if gpu < 4 else 1
    uuid = policy['idle_power_mapping_provenance']['uuid_by_gpu'][str(gpu)]
    jobs, previous = [], list(a.after if a.after is not None else spec.get('after', []))
    for index, label in enumerate(sequence):
        name = f"{spec['name']}-{index:02d}-{label}"
        command = ['numactl', f'--cpunodebind={node}', PYTHON,
                   str(ROOT / 'scripts/run_r16_with_device_identity.py'), '--guard-policy', str(POLICY),
                   '--out', '{attempt_dir}/job', '--expected-uuid', uuid,
                   '--expected-pci', f'0000:{PCI[gpu]}:00.0', '--numa-node', str(node), '--',
                   PYTHON, str(Path(__file__).resolve()), 'measure', '--spec', str(spec_path),
                   '--arm', label, '--fresh', str(index), '--out', '{attempt_dir}/job']
        jobs.append(dict(name=name, gpu=True, timing=True, priority=spec.get('priority', 10),
                         timeout_s=spec.get('timeout_s', 1800), needs_free_mib=8192,
                         cwd=str(ROOT), after=previous, env=environment(gpu), command=command))
        previous = [name]
    path = a.queue_dir / f"queue_{spec['name']}.json"
    write(path, jobs)
    print(json.dumps(dict(queue=str(path), jobs=len(jobs))))


def measure(a):
    """One fresh process: time one arm's forward with CUDA events."""
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise ValueError('GPU work requires the scheduler guard')
    spec = read(a.spec)
    arm = spec['arms'][a.arm]
    if sha(arm['library']) != arm['library_sha256'] or sha(arm['inputs']) != arm['inputs_sha256']:
        raise ValueError('artifact or input changed after queueing')
    a.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(arm.get('root', ROOT)) / 'python'))
    import torch
    from safetensors.torch import load_file
    from tilemega.dnn.check import NativeForward
    from tilemega.dnn.baselines import timed
    inputs = load_file(arm['inputs'], device='cuda')
    with torch.inference_mode(), NativeForward(arm['library'], arm['export'], arm['bridge'],
                                              spec['batch']) as native:
        if set(native.inputs) != set(inputs):
            raise ValueError('timed input names differ from the native forward signature')
        # NativeForward's own call checks finite outputs and L1/L2 equality; its
        # copies and comparisons stay outside the timing region.
        reference = native([inputs[name] for name in native.inputs])
        stream = torch.cuda.current_stream().cuda_stream
        values = timed(lambda: native.plan.launch(0, 1, stream), WARMUP, ITERATIONS)
        torch.cuda.synchronize()
        if any(not torch.equal(value, native.tensors[name])
               for name, value in zip(native.outputs, reference)):
            raise AssertionError('timed forward output differs from the checked execution')
    ordered = sorted(values)
    result = dict(schema=1, spec=str(a.spec.resolve()), spec_name=spec['name'], kind=spec['kind'],
                  arm=a.arm, fresh=a.fresh, model=spec['model'], batch=spec['batch'],
                  gpu=spec['gpu'], pid=os.getpid(), library=arm['library'],
                  library_sha256=arm['library_sha256'], inputs_sha256=arm['inputs_sha256'],
                  root_commit=arm['root_commit'], warmup=WARMUP, iterations=ITERATIONS,
                  clock='CUDA events', forward_ms=statistics.median(values),
                  p90_ms=ordered[math.ceil(0.9 * len(ordered)) - 1], samples_ms=values)
    write(a.out / 'result.json', result)
    # The guarded wrapper annotates every measurement with the device identity
    # it verified, so the timing is retained under its own schema as well.
    write(a.out / 'measurements' / 'measurements.json',
          dict(schema_version=1, evidence='verified', system='tilemega', protocol='P-DNN',
               model=spec['model'], batch=spec['batch'], precision='bf16', executor='L1',
               arm=a.arm, round_index=a.fresh, median_ms=result['forward_ms'],
               p90_ms=result['p90_ms'], samples_ms=values,
               measurement=dict(clock='CUDA events', warmup=WARMUP, iterations=ITERATIONS,
                                cuda_graph=False,
                                scope='full model forward; input and weight transfers excluded'),
               library_path=arm['library'], library_sha256=arm['library_sha256'],
               input_sha256=arm['inputs_sha256'], root_commit=arm['root_commit']))
    print(json.dumps({k: v for k, v in result.items() if k != 'samples_ms'}))


def job_result(state, name):
    for job in state['jobs']:
        if job['name'] != name:
            continue
        if job['status'] != 'completed':
            raise ValueError(f'job {name} is {job["status"]}')
        return read(Path(job['attempts'][-1]['directory']) / 'job/result.json')
    raise KeyError(name)


def collect(a):
    spec = read(a.spec)
    state = read(a.state)
    sequence = spec.get('order') or order(spec['kind'], list(spec['arms']))
    rows = [dict(visit=i, **job_result(state, f"{spec['name']}-{i:02d}-{label}"))
            for i, label in enumerate(sequence)]
    arms = {}
    for row in rows:
        arms.setdefault(row['arm'], []).append(row)
    summary = {label: dict(forward_ms=statistics.median(r['forward_ms'] for r in v),
                           range_pct=span([r['forward_ms'] for r in v]) if len(v) > 1 else 0.0,
                           processes=len(v)) for label, v in arms.items()}
    result = dict(schema=1, protocol='P-DNN', spec=spec, rows=[
        {k: v for k, v in row.items() if k != 'samples_ms'} for row in rows],
        arms=summary, collector_sha256=sha(__file__))
    if spec['kind'] == 'sel':
        ranked = sorted(summary, key=lambda k: summary[k]['forward_ms'])
        best, second = ranked[0], ranked[1]
        threshold = max(0.5, 2 * max(summary[best]['range_pct'], summary[second]['range_pct']))
        gap = (summary[second]['forward_ms'] / summary[best]['forward_ms'] - 1) * 100
        winner = best
        predicted = spec.get('predicted_ns', {})
        if gap <= threshold and predicted:
            winner = min((best, second), key=lambda k: predicted[k])
        result.update(measured_order=ranked, top_gap_pct=gap, threshold_pct=threshold,
                      winner=winner, tie_broken_by_prediction=winner != best,
                      winner_rank=int(winner[1:]),
                      winner_binary_sha256=sha(spec['arms'][winner]['library']))
    else:
        label = list(spec['arms'])[0]
        medians = [row['forward_ms'] for row in rows]
        result.update(winner=label, forward_ms=statistics.median(medians),
                      process_medians=medians, range_pct=span(medians))
    write(a.out, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('spec', 'rows')}, indent=1))


def select(a):
    """The forward solver's measure command: P-DNN over the compiled top-K."""
    out = a.out.resolve()
    out.mkdir(parents=True)
    rows = [r for r in read(a.candidates) if r['status'] == 'compiled']
    if len(rows) < 2:
        raise ValueError('P-DNN selection needs at least two compiled candidates')
    arms = {f"c{r['rank']}": dict(library=r['binary'], export=str(a.export.resolve()),
                                  bridge=str(a.bridge.resolve()), inputs=str(a.inputs.resolve()),
                                  root=str(a.root.resolve())) for r in rows}
    spec = dict(name=a.name, kind='sel', model=a.model, batch=a.batch, gpu=a.gpu, arms=arms,
                predicted_ns={f"c{r['rank']}": r['predicted_ns'] for r in rows})
    write(out / 'spec.json', spec)
    me = str(Path(__file__).resolve())
    subprocess.check_call([PYTHON, me, 'queue', '--spec', str(out / 'spec.json'),
                           '--queue-dir', str(out / 'queue')], cwd=ROOT)
    env = dict(os.environ, **environment(a.gpu))
    subprocess.check_call([PYTHON, '-m', 'tilemega.bench.scheduler', '--queue-dir', str(out / 'queue'),
                           '--out', str(out / 'scheduler'), '--policy', str(POLICY),
                           '--deadline-hours', '12'], cwd=ROOT, env=env)
    subprocess.check_call([PYTHON, me, 'collect', '--spec', str(out / 'spec.json'),
                           '--state', str(out / 'scheduler/state.json'),
                           '--out', str(out / 'result.json')], cwd=ROOT)


def final(a):
    """P-DNN closeout: three fresh processes of the selected artifact."""
    out = a.out.resolve()
    out.mkdir(parents=True)
    arms = {'final': dict(library=str(a.library.resolve()), export=str(a.export.resolve()),
                          bridge=str(a.bridge.resolve()), inputs=str(a.inputs.resolve()),
                          root=str(a.root.resolve()))}
    spec = dict(name=a.name, kind='final', model=a.model, batch=a.batch, gpu=a.gpu, arms=arms)
    write(out / 'spec.json', spec)
    me = str(Path(__file__).resolve())
    subprocess.check_call([PYTHON, me, 'queue', '--spec', str(out / 'spec.json'),
                           '--queue-dir', str(out / 'queue')], cwd=ROOT)
    env = dict(os.environ, **environment(a.gpu))
    subprocess.check_call([PYTHON, '-m', 'tilemega.bench.scheduler', '--queue-dir', str(out / 'queue'),
                           '--out', str(out / 'scheduler'), '--policy', str(POLICY),
                           '--deadline-hours', '12'], cwd=ROOT, env=env)
    subprocess.check_call([PYTHON, me, 'collect', '--spec', str(out / 'spec.json'),
                           '--state', str(out / 'scheduler/state.json'),
                           '--out', str(out / 'result.json')], cwd=ROOT)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('queue', 'measure', 'collect', 'select', 'final'))
    p.add_argument('--spec', type=Path)
    p.add_argument('--queue-dir', type=Path)
    p.add_argument('--state', type=Path)
    p.add_argument('--arm')
    p.add_argument('--fresh', type=int, default=0)
    p.add_argument('--after', nargs='*')
    p.add_argument('--out', type=Path)
    # select (the solver's --measure-cmd prefix supplies these)
    p.add_argument('--model')
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--gpu', type=int)
    p.add_argument('--name')
    p.add_argument('--export', type=Path)
    p.add_argument('--bridge', type=Path)
    p.add_argument('--inputs', type=Path)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--candidates', type=Path)
    p.add_argument('--library', type=Path)
    a = p.parse_args()
    if a.action in ('measure', 'collect', 'select', 'final') and a.out is None:
        p.error(a.action + ' requires --out')
    if a.action == 'queue':
        queue(a)
    elif a.action == 'measure':
        measure(a)
    elif a.action == 'collect':
        collect(a)
    elif a.action == 'final':
        for name in ('model', 'gpu', 'name', 'export', 'bridge', 'inputs', 'library'):
            if getattr(a, name) is None:
                p.error('final requires --' + name)
        final(a)
    else:
        for name in ('model', 'gpu', 'name', 'export', 'bridge', 'inputs', 'candidates'):
            if getattr(a, name) is None:
                p.error('select requires --' + name)
        select(a)


if __name__ == '__main__':
    main()
