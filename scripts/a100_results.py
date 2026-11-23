#!/usr/bin/env python3
"""Collect A100 baseline medians from three independently guarded processes."""
import csv
import json
import math
from pathlib import Path
import statistics

from a100_baselines import BATCHES, ENGINES, MODELS, ROOT, RUN, sha


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def measurement(job, state, inputs):
    if state.get('status') != 'done':
        return None
    folder = Path(state['out'])
    path = folder/f'job/measurements/B{job["batch"]}/measurements.json'
    return guarded_measurement(job, path, folder, inputs)


def guarded_measurement(job, path, folder, inputs, *, selector_bound=False):
    guard = read(folder/'guard_result.json')
    data = read(path)
    identity = folder/'job/physical-device-identity.json'
    if not (guard.get('code') == 0 and guard.get('acceptance') == 'strict'
            and guard.get('ownership_verified') and data.get('ownership_verified')
            and data.get('acceptance') == 'strict' and data.get('shared_checkpoint_identity')
            and data.get('checkpoint', {}).get('seed') == 0 and data.get('load_format') == 'auto'
            and data.get('batch') == job['batch'] and data.get('round_index') == job['round']
            and data.get('tp') == 1 and data.get('prompt_length') == 64 and data.get('max_tokens') == 1024
            and data.get('model_config_sha256') == inputs['models'][job['model']]['config_sha256']
            and data.get('prompt_ids_sha256') == inputs['models'][job['model']]['prompt_sha256']
            and identity.is_file()
            and (data.get('physical_device_identity_sha256') == sha(identity)
                 or selector_bound and 'physical_device_identity_sha256' not in data)
            and read(identity).get('nvml_uuid') == inputs['device']['uuid']):
        raise ValueError(f'invalid strict measurement receipt: {job["name"]}')
    times = {}
    for count in (1, 1024):
        rows = [row for row in data['generation_runs'] if row['N'] == count]
        if len(rows) != 4 or [bool(row['warmup']) for row in rows] != [True, False, False, False]:
            raise ValueError(f'incomplete request timing: {job["name"]} N={count}')
        if any(not math.isfinite(row['wall_seconds']) or row['wall_seconds'] <= 0 for row in rows):
            raise ValueError(f'invalid request timing: {job["name"]} N={count}')
        for row in rows:
            tokens = read(path.parent/row['tokens_file'])
            if len(tokens) != job['batch'] or any(len(seq) != count for seq in tokens):
                raise ValueError(f'invalid output length: {job["name"]} N={count}')
        times[count] = statistics.median(row['wall_seconds'] for row in rows[1:])
    t1, t1024 = times[1], times[1024]
    if t1024 <= t1:
        raise ValueError(f'nonpositive TPOT: {job["name"]}')
    return dict(tpot_ms=1000*(t1024-t1)/1023, t1_s=t1, t1024_s=t1024,
                tok_s=job['batch']*1024/t1024, checkpoint_sha256=data['checkpoint']['checkpoint_sha256'],
                versions=json.dumps(data['versions'], sort_keys=True),
                backend=json.dumps(data['configuration'], sort_keys=True), receipt=sha(path))


def write_csv(path, rows):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    inputs = read(RUN/'baseline_inputs.json')
    jobs = read(RUN/'queue/queue_baselines.json')
    state = read(RUN/'baseline/state.json')
    compiled = read(RUN/'build/verified_artifacts.json')
    occupancy = RUN/'baseline/occupancy.jsonl'
    last = json.loads(occupancy.read_text().splitlines()[-1]) if occupancy.is_file() else {}
    blocked = 'strict GPU ownership requires readable host /proc' in last.get('error', '')
    pending = 'blocked_host_pid_mapping' if blocked else 'pending'
    scope = {}
    identities = {model: set() for model in MODELS}
    for job in jobs:
        if not job.get('model'):
            continue
        key = tuple(job[k] for k in ('model', 'engine', 'batch'))
        step = state.get(job['name'], {})
        record = (job, step, measurement(job, step, inputs))
        scope.setdefault(key, []).append(record)
        if record[2]:
            identities[job['model']].add(record[2]['checkpoint_sha256'])
    if any(len(hashes) > 1 for hashes in identities.values()):
        raise ValueError('cross-framework checkpoint mismatch')
    baseline, tilemega = [], []
    for model in MODELS:
        shared = inputs['models'][model]
        for batch in BATCHES:
            fastest = None
            for engine in ENGINES:
                records = scope[model, engine, batch]
                results = [result for _, _, result in records if result]
                states = [step.get('status', 'pending') for _, step, _ in records]
                status = 'measured' if len(results) == 3 else pending if 'pending' in states else 'incomplete'
                if all(status == 'cancelled' for status in states):
                    status = 'cancelled'
                if not results and all(status == 'failed' for status in states):
                    logs = [RUN/'baseline'/f'{job["name"]}.log' for job, _, _ in records]
                    signatures = ('CUDA out of memory', 'torch.OutOfMemoryError', 'CUDA_ERROR_OUT_OF_MEMORY')
                    if all(path.is_file() and any(sig in path.read_text(errors='replace').rsplit('\nATTEMPT ', 1)[-1]
                                                  for sig in signatures)
                           for path in logs):
                        status = 'OOM'
                row = dict(model=model, framework=engine, batch=batch, status=status, processes=len(results),
                           required_processes=3, correctness='exact token counts' if results else 'pending',
                           tpot_ms='', t1_s='', t1024_s='', tok_s='', backend='', versions='',
                           gpu_uuid=inputs['device']['uuid'], config_sha256=shared['config_sha256'],
                           prompt_sha256=shared['prompt_sha256'], checkpoint_sha256=next(iter(identities[model]), ''),
                           harness_sha256=records[0][0]['harness_sha256'], receipt_sha256='',
                           evidence=' '.join(job['name'] for job, _, _ in records))
                if len(results) == 3:
                    for key in ('tpot_ms', 't1_s', 't1024_s', 'tok_s'):
                        row[key] = statistics.median(result[key] for result in results)
                    for key in ('backend', 'versions'):
                        entries = sorted({result[key] for result in results})
                        row[key] = entries[0] if len(entries) == 1 else json.dumps([json.loads(x) for x in entries])
                    row['receipt_sha256'] = ' '.join(result['receipt'] for result in results)
                    if fastest is None or row['tpot_ms'] < fastest['tpot_ms']:
                        fastest = row
                baseline.append(row)
            artifact = compiled.get(f'{model}_b{batch}', {})
            status = ('pending_sm80' if not artifact else
                      'compiled_pending_gpu' if artifact.get('prefill') else
                      'decode_compiled_pending_prefill')
            tilemega.append(dict(model=model, batch=batch,
                                 status=status,
                                 processes=0, required_processes=3 if model in list(MODELS)[:4] else 5,
                                 correctness='pending', backend=artifact.get('backend', ''),
                                 tpot_ms='', t1_s='', t1024_s='', tok_s='',
                                 fastest_baseline=fastest['framework'] if fastest else '',
                                 fastest_baseline_tpot_ms=fastest['tpot_ms'] if fastest else '', speedup='',
                                 config_sha256=shared['config_sha256'], prompt_sha256=shared['prompt_sha256'],
                                 checkpoint_sha256=next(iter(identities[model]), ''),
                                 artifact_sha256=' '.join(
                                     artifact.get(f'{phase}_sha256', '') for phase in ('prefill', 'decode')).strip(),
                                 gpu_uuid=inputs['device']['uuid'], versions=''))
    out = ROOT/'docs/results'
    write_csv(out/'A100_baselines.csv', baseline)
    write_csv(out/'A100_llm.csv', tilemega)


if __name__ == '__main__':
    main()
