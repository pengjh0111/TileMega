#!/usr/bin/env python3
"""End-to-end P-SEL inside the native compiler's guarded A800 allocation."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

from a100_baselines import BATCHES, MODELS, inputs, inspect_inputs, sha
from a800_baselines import ENVS, INVENTORY, calibrated_device, device_policy
from a800_native import artifact
from sm90r3_llm import order, selection_result, span
from tilemega.bench.scheduler import write_json
from tilemega.serving.baseline_common import require_guard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=MODELS)
    parser.add_argument('--batch', type=int, required=True, choices=BATCHES)
    parser.add_argument('--prefill', required=True)
    parser.add_argument('--target', required=True, type=Path)
    parser.add_argument('--candidates', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if (not os.environ.get('TILEMEGA_STRICT_HOST_GUARD_PID')
            or os.environ.get('SLURM_JOB_PARTITION') != 'gpu_a800'):
        parser.error('P-SEL requires the strict guard inside a gpu_a800 allocation')
    require_guard(tp=1)
    spec = json.loads(args.target.read_text())
    if not spec.get('sm80_native_constants'):
        raise ValueError('P-SEL requires a measured calibration target')
    device = calibrated_device(device_policy(), spec)
    if args.prefill == 'prompt':
        if args.model != 'olmoe_1b_7b':
            raise ValueError('this model needs its independent prefill artifact')
        prefill = 'prompt'
    else:
        prefill = str(artifact(Path(args.prefill))[0])
    arms = {}
    for row in json.loads(args.candidates.read_text()):
        if row['status'] != 'compiled':
            continue
        binary, identity, plan = artifact(Path(row['binary']))
        if (identity['binary_sha256'] != row['binary_sha256']
                or identity['target_sha256'] != sha(args.target)
                or plan['runtime_batch'] != args.batch):
            raise ValueError('P-SEL candidate does not match the planned workload')
        arms[f"c{row['rank']}"] = dict(row, binary=str(binary))
    if len(arms) < 2:
        raise ValueError('P-SEL needs two or more compiled candidates')
    args.out.mkdir(parents=True, exist_ok=False)

    config, prompt = inputs(args.model)
    manifest = inspect_inputs(INVENTORY)[args.model]
    checkpoint_limit = str(math.ceil(manifest['checkpoint_bytes']/2**30))
    samples = {label: [] for label in arms}
    checkpoint_hashes = set()
    sequence = list(dict.fromkeys(order('sel', list(arms), args.model)))
    for index, choice in enumerate(sequence):
        row = arms[choice]
        folder = args.out/f'{index:02d}-{choice}'
        folder.mkdir()
        measure = folder/'measurements'
        command = [str(ENVS/'vllm/bin/python'), '-m', 'tilemega.serving.random_checkpoint',
            '--config', str(config), '--out', str(folder/'checkpoint'), '--seed', '0',
            '--max-gib', checkpoint_limit, '--', str(ENVS/'tilemega/bin/python'),
            '-m', 'tilemega.serving.r16_run', '--prefill', prefill, '--decode', row['binary'],
            '--prompt-ids', str(prompt), '--batch', str(args.batch), '--max-tokens', '1024',
            '--warmup', '1', '--repeats', '3', '--mode', 'L1', '--decode-loop', '1',
            '--prefill-mode', 'L1', '--out', str(measure), '--round-index', str(index)]
        write_json(folder/'command.json', command)
        with (folder/'process.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        receipt = measure/f'B{args.batch}'/'measurements.json'
        result = json.loads(receipt.read_text())
        if (not result.get('shared_checkpoint_identity') or result.get('acceptance') != 'strict'
                or result['configuration']['decode_sha256'] != row['binary_sha256']
                or result['prompt_ids_sha256'] != manifest['prompt_sha256']
                or result['model_config_sha256'] != manifest['config_sha256']):
            raise ValueError('P-SEL candidate lacks comparable measurement evidence')
        checkpoint_hashes.add(result['checkpoint']['checkpoint_sha256'])
        samples[choice].append(float(result['tpot_seconds'])*1e3)
    if len(checkpoint_hashes) != 1:
        raise ValueError('P-SEL checkpoint identity changed between candidates')
    summary = {label: dict(tpot_ms=statistics.median(values), range_pct=span(values),
                           processes=len(values)) for label, values in samples.items()}
    predicted = {label: row['predicted_ns'] for label, row in arms.items()}
    result = selection_result(summary, predicted)
    winner = arms[result['winner']]
    write_json(args.out/'result.json', dict(result, schema=1, protocol='P-SEL-single-process',
        winner_rank=winner['rank'], winner_binary_sha256=sha(winner['binary']), gpu_uuid=device['uuid'],
        checkpoint_sha256=next(iter(checkpoint_hashes)), arms=summary, predicted_ns=predicted,
        launch_order=sequence, formal_timing=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
