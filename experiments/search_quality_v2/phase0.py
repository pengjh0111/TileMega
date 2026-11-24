"""Rebuild and reproduce two frozen v1 configurations using the v1 protocol."""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

from guarded import ROOT, PYTHON, gpu_attempt, inventory, read, run, sha, write


V1_DOC = ROOT / 'docs/results/sm90_search_quality/qwen3_1_7b_b1'
TARGET = ROOT / 'configs/targets/sm_90_h800.json'
PROMPT = ROOT / 'configs/prompts/qwen_isl64.json'
CASE = ROOT / 'runs/sm90_search_quality_v2/qwen3_1_7b_b1'
IDS = ('q000', 'q079')
V1_TOLERANCE = .015


def inputs(old_run, compiler, paper, path):
    old_search = read(V1_DOC / 'frozen_search.json')
    old_cells = read(old_run / 'cells.json')
    cell = next(row for row in old_cells if row['id'] == 'qwen3_1_7b-B1')
    export = Path(cell['export'])
    seed = Path(cell['seed'])
    checkpoint = old_run / 'checkpoint'
    assert sha(export) == cell['export_sha256'] and sha(seed) == cell['seed_sha256']
    assert sha(TARGET) == old_search['target_sha256']
    assert sha(PROMPT) == old_search['prompt_sha256']
    with (V1_DOC / 'candidates.csv').open() as handle:
        references = {r['original_q_id']: float(r['tpot_ms']) for r in csv.DictReader(handle)
                      if r['original_q_id'] in IDS}
    cards = inventory()
    permitted = cards[-3:]
    card = next(c for c in permitted if c['uuid'] == old_search['gpu_uuid'])
    if card['numa'] < 0:
        raise ValueError('a bound NUMA node is required')

    old_policy = read(old_run / 'guard_policy.json')
    old_by_uuid = {
        uuid: old_policy['idle_power_w_by_gpu'][index]
        for index, uuid in old_policy['search_quality_mapping']['uuid_by_index'].items()
    } if 'uuid_by_index' in old_policy['search_quality_mapping'] else {}
    if not old_by_uuid:
        # V1's live index map is recorded separately from its original eight-card inventory.
        with (old_run / 'gpu-inventory.csv').open() as handle:
            for row in csv.reader(handle, skipinitialspace=True):
                if len(row) >= 2 and row[0].strip().isdigit() and row[1].startswith('GPU-'):
                    index = row[0].strip()
                    if index in old_policy['idle_power_w_by_gpu']:
                        old_by_uuid[row[1]] = old_policy['idle_power_w_by_gpu'][index]
    if any(c['uuid'] not in old_by_uuid for c in permitted):
        raise ValueError('no matching measured idle power reference for an authorized UUID')
    old_policy['allowed_gpu_indices'] = [c['index'] for c in permitted]
    old_policy['allowed_timing_gpu_indices'] = old_policy['allowed_gpu_indices']
    old_policy['idle_power_w_by_gpu'] = {str(c['index']): old_by_uuid[c['uuid']] for c in permitted}
    old_policy['sq2_live_uuid_by_index'] = {str(c['index']): c['uuid'] for c in permitted}
    write(path / 'guard_policy.json', old_policy)

    candidates = {c['original_q_id']: c for c in old_search['candidates']}
    arms = {}
    for q in IDS:
        item = candidates[q]
        canonical = json.dumps(item['coordinates'], sort_keys=True, separators=(',', ':')).encode()
        assert hashlib.sha256(canonical).hexdigest() == item['coordinates_sha256']
        coord = path / q / 'coordinates.json'
        write(coord, item['coordinates'])
        arms[q] = {'coordinate_path': str(coord), 'coordinates_sha256': item['coordinates_sha256'],
                   'v1_tpot_ms': references[q]}
    result = {
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'nvcc': subprocess.check_output(['nvcc', '--version'], text=True),
        'compiler': str(compiler), 'compiler_sha256': sha(compiler),
        'paper_sha256': sha(paper), 'target_sha256': sha(TARGET), 'prompt_sha256': sha(PROMPT),
        'export': str(export), 'export_sha256': sha(export), 'seed': str(seed), 'seed_sha256': sha(seed),
        'checkpoint': str(checkpoint), 'checkpoint_sha256': old_search['checkpoint_sha256'],
        'cards': cards, 'authorized_cards': permitted, 'reproduction_card': card, 'arms': arms,
        'tolerance': V1_TOLERANCE, 'protocol_order': ['q000', 'q079', 'q079', 'q000'],
        'identified_at': time.time(),
    }
    write(path / 'reproduction_inputs.json', result)
    return result


def compile_arm(q, path, spec):
    directory = path / q
    artifact = directory / 'decode.so'
    coord = Path(spec['arms'][q]['coordinate_path'])
    if not (directory / 'build/exit.json').exists():
        command = [spec['compiler'], 'compile', spec['export'], artifact, '--tmexec-config', coord,
                   '--serving', 'decode', '--emit', 'serving', '--batch', '1',
                   '--runtime-target', TARGET, '--nvcc-split-compile', '4']
        run(command, directory / 'build', dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    assert read(directory / 'build/exit.json')['returncode'] == 0, q
    plan = read(str(artifact) + '.plan.json')
    assert plan['native_origin'] == 'fixed' and plan['native_dense'] == read(coord)
    assert plan['mode'] == 'L1' and plan['loop'] == 1
    ident = read(str(artifact) + '.identity.json')
    assert ident['binary_sha256'] == sha(artifact)
    return artifact


def timing_value(attempt, binary, spec, check):
    job = attempt / 'job/measurements'
    path = job / 'B1/measurements.json'
    value = read(path)
    config = value['configuration']
    assert config['decode_sha256'] == sha(binary)
    assert value['checkpoint']['checkpoint_sha256'] == spec['checkpoint_sha256']
    assert value['prompt_ids_sha256'] == spec['prompt_sha256']
    assert value['physical_device_identity']['nvml_uuid'] == spec['reproduction_card']['uuid']
    assert (value['batch'], value['tp'], value['prompt_length'], value['max_tokens']) == (1, 1, 64, 1024)
    assert (config['prefill_binary'], config['prefill_mode'], config['mode'], config['decode_loop']) == ('prompt mode', 'L1', 'L1', 1)
    assert all(row['decode_loop_used'] for row in config['actual_execution'] if row['tokens'] > 1)
    wall = {}
    for n in (1, 1024):
        runs = [row for row in value['generation_runs'] if row['N'] == n]
        assert [row['warmup'] for row in runs] == [True, False, False, False]
        wall[n] = statistics.median(row['wall_seconds'] for row in runs if not row['warmup'])
    tpot = (wall[1024] - wall[1]) / 1023
    assert math.isclose(tpot, value['tpot_seconds'], abs_tol=1e-12)
    if check:
        c1 = read(job / 'c1.json')
        c2 = read(job / 'c2.json')
        assert c1['positions'] == 256 and c1['pass'] and c2['same_binary'] and c2['pass']
    return {'tpot_ms': 1000 * tpot, 'pid': value['pid'], 'measurement': str(path),
            'measurement_sha256': sha(path), 'checks_requested': check}


def measure(q, binary, spec, path, round_index):
    folder = path / q / f'round{round_index}'
    if (folder / 'accepted.json').exists():
        return read(folder / 'accepted.json')
    check = round_index == 0
    command = [PYTHON, '-m', 'tilemega.serving.r16_run', '--model', spec['checkpoint'],
               '--prefill', 'prompt', '--decode', binary, '--prompt-ids', PROMPT, '--batch', 1,
               '--max-tokens', 1024, '--warmup', 1, '--repeats', 3,
               '--mode', 'L1', '--decode-loop', 1, '--prefill-mode', 'L1',
               '--round-index', round_index, '--out', '@ATTEMPT@/job/measurements']
    if check:
        command.extend(['--check-step-loop', '--hf-check', '--hf-positions', 256])
    for i in range(10000):
        dest = folder / f'attempt{i:04d}'
        if dest.exists():
            guard_file = dest / 'guard/guard_result.json'
            if not guard_file.exists():
                continue
            guard = read(guard_file)
        else:
            guard = gpu_attempt(command, dest, path / 'guard_policy.json', spec['reproduction_card'])
        if guard['code'] == 75:
            time.sleep(5)
            continue
        if guard['code'] != 0:
            raise RuntimeError(f'{q} process failed with code {guard["code"]}: {dest}')
        result = timing_value(dest, binary, spec, check)
        write(folder / 'accepted.json', result)
        print(q, round_index, 'valid', result['tpot_ms'], flush=True)
        return result
    raise RuntimeError(f'{q} guard retries exhausted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-run', type=Path, required=True)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--paper', type=Path, required=True)
    args = parser.parse_args()
    path = CASE / 'phase0_reproduction'
    path.mkdir(parents=True, exist_ok=True)
    spec_path = path / 'reproduction_inputs.json'
    spec = read(spec_path) if spec_path.exists() else inputs(args.old_run, args.compiler, args.paper, path)
    assert str(args.compiler) == spec['compiler'] and sha(args.compiler) == spec['compiler_sha256']
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        binaries = dict(zip(IDS, pool.map(lambda q: compile_arm(q, path, spec), IDS), strict=True))
    samples = {q: [] for q in IDS}
    for q in spec['protocol_order']:
        samples[q].append(measure(q, binaries[q], spec, path, len(samples[q])))
    rows = {}
    for q in IDS:
        measured = statistics.median(row['tpot_ms'] for row in samples[q])
        reference = spec['arms'][q]['v1_tpot_ms']
        error = measured / reference - 1
        rows[q] = {'reference_ms': reference, 'measured_ms': measured, 'relative_delta': error,
                   'pass': abs(error) <= V1_TOLERANCE, 'processes': samples[q],
                   'binary_sha256': sha(binaries[q])}
    passed = all(row['pass'] for row in rows.values())
    write(path / 'summary.json', {'pass': passed, 'rows': rows, 'completed_at': time.time()})
    print('Phase 0 reproduction', 'pass' if passed else 'STOP', rows, flush=True)
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
