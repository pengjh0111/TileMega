#!/usr/bin/env python3
"""End-to-end P-SEL inside the native compiler's guarded A800 allocation."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

from a100_baselines import BATCHES, MODELS, inputs, inspect_inputs, sha
from a800_baselines import ENVS, INVENTORY, calibrated_device, device_policy, hardware_class
from a800_native import artifact, verify_request_tokens
from sm90r3_llm import order, selection_result, span
from tilemega.bench.scheduler import write_json
from tilemega.serving.baseline_common import require_guard
from tilemega.serving.random_checkpoint import ALGORITHM, validate_live_checkpoint


def compatible_calibration(before, after, previous_hash):
    old, new = before.copy(), after.copy()
    old_prices, new_prices = (d.pop('sm80_native_constants').copy() for d in (old, new))
    old_decisions, new_decisions = (d.pop('sm80_native_decisions') for d in (old, new))
    if old != new:
        return False
    service_fields = {'dense_gemm_service', 'fused_expert_service', 'attention_fresh_service',
                      'residual_copy_header_equivalence'}
    changed = {key for key in old_prices.keys() | new_prices.keys()
               if old_prices.get(key) != new_prices.get(key)}
    # The caller has already proved that the rebuilt device code is identical.
    if old_decisions == new_decisions and changed <= service_fields:
        return True
    old_mma = old_prices.pop('mma_mainloop')
    new_mma = new_prices.pop('mma_mainloop')
    supplement = new_mma.get('supplements', [])[-1:]
    return (new_decisions == dict(old_decisions, reuse_payload=True) and old_prices == new_prices
            and bool(supplement) and supplement[0]['parent_profile_sha256'] == previous_hash
            and all(new_mma[stat].get(key) == value
                    for stat in ('median_ns', 'p99_ns') for key, value in old_mma[stat].items()))


def prior_measurements(arms, model, batch, prefill, target, manifest, device):
    path = os.environ.get('TILEMEGA_PSEL_PRIOR')
    if not path:
        return {}
    prefill_hash = 'prompt mode' if prefill == 'prompt' else sha(Path(prefill))
    target_hash = sha(target)
    by_binary = {r['binary_sha256']: label for label, r in arms.items()}
    reused = {}
    for entry in json.loads(Path(path).read_text()):
        files = {key: Path(entry[key]) for key in ('comparison', 'measurement', 'guard', 'planned')}
        if any(sha(file) != entry[f'{key}_sha256'] for key, file in files.items()):
            raise ValueError('previous selection evidence changed')
        proof, result, guard, plan = (json.loads(files[key].read_text()) for key in
                                ('comparison', 'measurement', 'guard', 'planned'))
        label = by_binary.get(proof['candidate_sha256'])
        if (label is None or label in reused or not proof['pass'] or proof['diagnostic']
                or not all(proof[k] for k in ('same_coordinates', 'same_workload',
                                             'same_source', 'same_instruction_encoding'))
                or guard['code'] != 0 or guard['acceptance'] != 'strict'
                or not guard['ownership_verified']):
            raise ValueError('previous selection does not certify the same native candidate')
        cfg = result['configuration']
        old_target_hash = cfg['identities']['decode']['target_sha256']
        if old_target_hash != target_hash:
            previous = Path(entry['reference_target'])
            if sha(previous) != old_target_hash:
                raise ValueError('previous selection target hash does not match its process')
            before, after = (json.loads(p.read_text()) for p in (previous, target))
            if not compatible_calibration(before, after, old_target_hash):
                raise ValueError('previous selection has different runtime hardware or target options')
        if (plan['model'] != model or hardware_class(plan['device']) != hardware_class(device)
                or result['batch'] != batch or result['tp'] != 1
                or result['prompt_length'] != 64 or result['max_tokens'] != 1024
                or result['acceptance'] != 'strict' or not result['ownership_verified']
                or not result['shared_checkpoint_identity'] or result['checkpoint']['seed'] != 0
                or cfg['prefill_sha256'] != prefill_hash
                or cfg['decode_sha256'] != proof['reference_sha256']
                or result['prompt_ids_sha256'] != manifest['prompt_sha256']
                or result['model_config_sha256'] != manifest['config_sha256']):
            raise ValueError('previous selection has different request inputs or execution')
        verify_request_tokens(files['measurement'], batch, 3)
        reused[label] = (dict(entry, gpu_uuid=plan['device']['uuid']), result)
    return reused


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=MODELS)
    parser.add_argument('--batch', type=int, required=True, choices=BATCHES)
    parser.add_argument('--prefill', required=True)
    parser.add_argument('--target', required=True, type=Path)
    parser.add_argument('--candidates', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--shared-checkpoint', type=Path, help=argparse.SUPPRESS)
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
    rank = os.environ.get('TILEMEGA_PSEL_RANK')
    if rank is not None:
        label = f'c{rank}'
        if label not in arms or os.environ.get('TILEMEGA_PSEL_PRIOR'):
            raise ValueError('split P-SEL requires a compiled rank and a fresh process')
        arms = {label: arms[label]}
    config, prompt = inputs(args.model)
    manifest = inspect_inputs(INVENTORY)[args.model]
    checkpoint_limit = str(math.ceil(manifest['checkpoint_bytes']/2**30))
    samples = {label: [] for label in arms}
    process_receipts = {}
    checkpoint_hashes = set()
    reused = prior_measurements(arms, args.model, args.batch, prefill, args.target, manifest, device)
    for choice, (entry, result) in reused.items():
        samples[choice].append(float(result['tpot_seconds'])*1e3)
        process_receipts[choice] = entry['measurement_sha256']
        checkpoint_hashes.add(result['checkpoint']['checkpoint_sha256'])
    sequence = [choice for choice in dict.fromkeys(order('sel', list(arms), args.model))
                if choice not in reused]
    checkpoint = (args.out/'shared-checkpoint').resolve()
    if args.shared_checkpoint is None:
        args.out.mkdir(parents=True, exist_ok=False)
        if sequence:
            # Retain one read-only seed-0 checkpoint supervisor for the whole
            # allocation. The CUDA executors below still use separate processes.
            command = [str(ENVS/'vllm/bin/python'), '-m', 'tilemega.serving.random_checkpoint',
                       '--config', str(config), '--out', str(checkpoint), '--seed', '0',
                       '--max-gib', checkpoint_limit, '--', sys.executable, str(Path(__file__).resolve()),
                       *sys.argv[1:], '--shared-checkpoint', str(checkpoint)]
            return subprocess.run(command).returncode
    else:
        if (args.shared_checkpoint.resolve(strict=True) != checkpoint or
                os.environ.get('TILEMEGA_RANDOM_CHECKPOINT') != str(checkpoint)):
            raise ValueError('the selection worker requires its allocation-owned checkpoint')
        live = validate_live_checkpoint(checkpoint)
        if (live['supervisor_pid'] != os.getppid() or live['algorithm'] != ALGORITHM
                or live['config_sha256'] != manifest['config_sha256']
                or live['seed'] != 0 or live['total_bytes'] != manifest['checkpoint_bytes']):
            raise ValueError('the shared selection checkpoint changed its frozen workload')
        checkpoint_hashes.add(live['checkpoint_sha256'])
    for index, choice in enumerate(sequence):
        row = arms[choice]
        folder = args.out/f'{index:02d}-{choice}'
        folder.mkdir()
        measure = folder/'measurements'
        command = [str(ENVS/'tilemega/bin/python'), '-m', 'tilemega.serving.r16_run',
            '--prefill', prefill, '--decode', row['binary'],
            '--prompt-ids', str(prompt), '--batch', str(args.batch), '--max-tokens', '1024',
            '--warmup', '1', '--repeats', '3', '--mode', 'L1', '--decode-loop', '1',
            '--prefill-mode', 'L1', '--out', str(measure), '--round-index', str(index)]
        write_json(folder/'command.json', command)
        with (folder/'process.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        receipt = measure/f'B{args.batch}'/'measurements.json'
        verify_request_tokens(receipt, args.batch, 3)
        result = json.loads(receipt.read_text())
        if (not result.get('shared_checkpoint_identity') or result.get('acceptance') != 'strict'
                or result['configuration']['decode_sha256'] != row['binary_sha256']
                or result['prompt_ids_sha256'] != manifest['prompt_sha256']
                or result['model_config_sha256'] != manifest['config_sha256']):
            raise ValueError('P-SEL candidate lacks comparable measurement evidence')
        checkpoint_hashes.add(result['checkpoint']['checkpoint_sha256'])
        process_receipts[choice] = sha(receipt)
        samples[choice].append(float(result['tpot_seconds'])*1e3)
    if len(checkpoint_hashes) != 1:
        raise ValueError('P-SEL checkpoint identity changed between candidates')
    summary = {label: dict(tpot_ms=statistics.median(values), range_pct=span(values),
                           processes=len(values)) for label, values in samples.items()}
    predicted = {label: row['predicted_ns'] for label, row in arms.items()}
    result = (dict(winner=next(iter(arms)), split_rank_measurement=True) if rank is not None
              else selection_result(summary, predicted))
    winner = arms[result['winner']]
    write_json(args.out/'result.json', dict(result, schema=1, protocol='P-SEL-single-process',
        protocol_version='a800-single-v2', kind='selection', process_receipts=process_receipts,
        winner_rank=winner['rank'], winner_binary_sha256=sha(winner['binary']), gpu_uuid=device['uuid'],
        checkpoint_sha256=next(iter(checkpoint_hashes)), arms=summary, predicted_ns=predicted,
        launch_order=sequence, prior_measurements={k:v[0] for k,v in reused.items()},
        formal_timing=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
