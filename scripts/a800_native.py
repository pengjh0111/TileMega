#!/usr/bin/env python3
"""Run one immutable SM80 prefill/decode pair through guarded correctness or timing."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

from a100_baselines import BATCHES, MODELS, ROOT, device_environment, inputs, inspect_inputs, sha
from a800_baselines import ENVS, INVENTORY, RUN, device_policy
from tilemega.bench.scheduler import main as schedule, write_json


def artifact(path):
    path = path.resolve(strict=True)
    identity = json.loads(Path(str(path)+'.identity.json').read_text())
    manifest = Path(str(path)+'.plan.json')
    if (
        identity['arch'] != 'sm_80' or identity['nvcc_arch'] != 'sm_80' or identity['trace']
        or identity['binary_sha256'] != sha(path) or identity['manifest_sha256'] != sha(manifest)
    ):
        raise ValueError(f'expected an immutable uninstrumented SM80 artifact: {path}')
    return path, identity, json.loads(manifest.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'run'))
    parser.add_argument('--stage', choices=('correctness', 'measure'), required=True)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--batch', type=int, choices=BATCHES, required=True)
    parser.add_argument('--prefill', required=True)
    parser.add_argument('--decode', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--budget-s', type=int, default=840)
    parser.add_argument('--timeout-s', type=int, default=600)
    args = parser.parse_args()

    if not 0 < args.timeout_s + 111 < args.budget_s or args.round < 0:
        parser.error('reserve time for strict admission and child cleanup, and use a nonnegative round')
    decode, decode_id, plan = artifact(args.decode)
    with args.target.open() as stream:
        target = json.load(stream)
    expected = target['device_provenance']['uuid']
    if target['arch_tag'] != 'sm_80' or 'A800' not in target['device_provenance']['name']:
        parser.error('require a separate A800 target from the guarded device probe')
    if plan.get('runtime_batch') != args.batch or plan.get('native_origin') not in ('fixed', 'solver'):
        parser.error('decode plan must be native and match the specified batch')
    if args.prefill == 'prompt':
        if args.model != 'olmoe_1b_7b':
            parser.error('the frozen model protocol uses an independent prefill library')
        prefill, prefill_id = 'prompt', {}
    else:
        prefill, prefill_id, prefill_plan = artifact(Path(args.prefill))
        if not (prefill_plan.get('phase') == 'prefill' and
                prefill_plan['batch_lo'] <= args.batch <= prefill_plan['batch_hi']):
            parser.error('prefill plan must match the batch')

    identity = dict(model=args.model, batch=args.batch, round=args.round, stage=args.stage,
                    target_sha256=sha(args.target), device_uuid=expected,
                    decode_sha256=decode_id['binary_sha256'],
                    prefill_sha256=prefill_id.get('binary_sha256', 'prompt'))
    scope = RUN/'native'/expected/args.model/f'B{args.batch}'
    if args.stage == 'measure':
        if (plan['native_origin'] != 'solver' or not target.get('sm80_native_constants')
                or decode_id['target_sha256'] != identity['target_sha256']
                or plan['native_selection'].get('selection') != 'measure'):
            parser.error('formal measurement requires the solver-selected artifact and its calibrated target')
        receipts = [json.loads(path.read_text()) for path in scope.glob('correctness-*.json')]
        fixed = ('device_uuid', 'decode_sha256', 'prefill_sha256', 'target_sha256')
        rounds = {result['round'] for result in receipts if result['pass']
                  and all(result[key] == identity[key] for key in fixed)}
        if len(rounds) < 5:
            parser.error('formal timing requires five fresh accepted correctness processes for this pair')
    config, prompt = inputs(args.model)
    description = inspect_inputs(INVENTORY)[args.model]
    name = f'native-{args.stage}-r{args.round}'
    if args.mode == 'plan':
        print(json.dumps(dict(identity, **description, prefill=str(prefill), decode=str(decode),
                              timeout_s=args.timeout_s, max_tokens=1024, step=name), indent=2))
        return 0

    receipt = scope/f'{args.stage}-{args.round}.json'
    if receipt.exists():
        raise FileExistsError('this artifact pair already has a receipt for the selected round')
    policy = device_policy()
    dev = device_environment(policy, 'A800')
    if dev['uuid'] != expected:
        raise RuntimeError('native run requires the A800 UUID recorded in the target')
    py = str(ENVS/'tilemega/bin/python')
    seed_py = str(ENVS/'vllm/bin/python')
    run = scope/'allocations'/os.environ['SLURM_JOB_ID']
    run.mkdir(parents=True, exist_ok=False)
    policy_path = run/'guard-policy.json'
    write_json(policy_path, policy)
    child = [py, '-m', 'tilemega.serving.r16_run', '--prefill', str(prefill), '--decode', str(decode),
             '--prompt-ids', str(prompt), '--batch', str(args.batch), '--max-tokens', '1024',
             '--mode', 'L1', '--prefill-mode', 'L1', '--decode-loop', '1',
             '--warmup', '1', '--repeats', '3' if args.stage == 'measure' else '1',
             '--round-index', str(args.round), '--out', '{attempt_dir}/job/measurements',
             '--system-label', 'TM-sm80', '--policy', str(policy_path)]
    if args.stage == 'correctness':
        child += ['--check-step-loop', '--hf-check']
    command = ['taskset', '--cpu-list', dev['cpus'], py,
               str(ROOT/'scripts/run_r16_with_device_identity.py'),
               '--out', '{attempt_dir}/job', '--guard-policy', str(policy_path),
               '--expected-uuid', expected, '--expected-pci', dev['pci'],
               '--numa-node', dev['numa_node'], '--', seed_py,
               '-m', 'tilemega.serving.random_checkpoint', '--config', str(config),
               '--out', '{attempt_dir}/checkpoint', '--seed', '0',
               '--max-gib', str(math.ceil(description['checkpoint_bytes']/2**30)), '--', *child]
    env = dict(TILEMEGA_DEVICE_INDEX=str(dev['index']), TILEMEGA_NUMA_NODE=dev['numa_node'],
               TILEMEGA_GPU_PID_MODE='strict', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    os.environ.update(TILEMEGA_DEVICE_INDEX=env['TILEMEGA_DEVICE_INDEX'],
                      TILEMEGA_GPU_LOCK=policy['gpu_lock_path'])
    queue = run/'queue'
    queue.mkdir()
    state = run/'results'
    needs = min(dev['memory_mib'] - policy['max_hidden_mib'],
                math.ceil(description['checkpoint_bytes']/2**20) + 8192)
    write_json(queue/'queue_native.json', [dict(name=name, gpu=True, timing=args.stage == 'measure',
               env=env, command=command, timeout_s=args.timeout_s, needs_free_mib=needs, cwd=str(ROOT))])
    schedule(['--queue-dir', str(queue), '--out', str(state), '--policy', str(policy_path),
              '--deadline-hours', str(args.budget_s/3600), '--release-on-blocked'])
    result = json.loads((state/'state.json').read_text())[name]
    if result['status'] != 'done':
        return 1
    evidence = Path(result['out'])/'job/measurements'
    if args.stage == 'correctness':
        c1, c2 = [json.loads((evidence/f'{label}.json').read_text()) for label in ('c1', 'c2')]
        if not (c1['pass'] and c2['pass'] and c2['same_binary']
                and {trial['decode_loop_used'] for trial in c2['checks']} == {False, True}):
            return 1
    write_json(receipt, dict(identity, **{'pass': True},
               job=run.name, evidence=str(evidence)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
