#!/usr/bin/env python3
"""Run one immutable SM80 prefill/decode pair through guarded correctness or timing."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from a100_baselines import BATCHES, MODELS, ROOT, inputs, inspect_inputs, sha
from a800_baselines import ENVS, INVENTORY, RUN, calibrated_device, device_policy
from tilemega.bench.scheduler import main as schedule, write_json


def allocation_checkpoint():
    job = os.environ['SLURM_JOB_ID']
    if not job.isdecimal():
        raise ValueError('shared checkpoint requires a Slurm job identifier')
    return (RUN/'native-pair-checkpoints'/job).resolve()


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


def verify_request_tokens(measurement, batch, repeats):
    import collections

    result = json.loads(measurement.read_text())
    groups = collections.defaultdict(list)
    for request in result['generation_runs']:
        length = request['N']
        if length not in (1, 1024):
            raise ValueError('unexpected native request length')
        tokens = json.loads((measurement.parent/request['tokens_file']).read_text())
        if len(tokens) != batch or any(len(row) != length for row in tokens):
            raise ValueError('native request returned an incomplete token sequence')
        groups[length].append((bool(request['warmup']), tokens))
    for length in (1, 1024):
        runs = groups[length]
        if len(runs) != 1 + repeats or [warmup for warmup, _ in runs] != [True] + [False] * repeats:
            raise ValueError('native process did not complete the frozen timing protocol')
        if any(tokens != runs[0][1] for _, tokens in runs[1:]):
            raise ValueError(f'native greedy outputs differ between requests at N={length}')
    if groups[1][0][1] != [row[:1] for row in groups[1024][0][1]]:
        raise ValueError('native short and full requests disagree on first tokens')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'run'))
    parser.add_argument('--stage', choices=('correctness', 'measure', 'diagnose', 'tune-prefill', 'tune-decode', 'candidate-timing'), required=True)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--batch', type=int, choices=BATCHES, required=True)
    parser.add_argument('--prefill', required=True)
    parser.add_argument('--decode', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--hf-reference', action='store_true')
    parser.add_argument('--numeric-gate', action='store_true',
                        help='verify the SM90 teacher-forced and same-binary executor contract')
    parser.add_argument('--budget-s', type=int, default=840)
    parser.add_argument('--timeout-s', type=int, default=600)
    parser.add_argument('--prefill-mode', choices=('L1', 'L2'), default='L1')
    tools = parser.add_mutually_exclusive_group()
    tools.add_argument('--sanitizer', type=Path)
    tools.add_argument('--cuda-gdb', type=Path)
    parser.add_argument('--diagnostic-config', type=Path)
    parser.add_argument('--seed-suite', choices=('vllm', 'tilemega'), default='vllm')
    parser.add_argument('--expected-checkpoint-sha256')
    parser.add_argument('--reuse-checkpoint', action='store_true',
                        help='load the sealed checkpoint owned by the paired allocation')
    parser.add_argument('--host-preparation', type=Path)
    parser.add_argument('--private-stream', action='store_true')
    parser.add_argument('--routing-export', type=Path)
    parser.add_argument('--routing-tokens', type=Path)
    args = parser.parse_args()

    if not 0 < args.timeout_s + 111 < args.budget_s or args.round < 0:
        parser.error('reserve time for strict admission and child cleanup, and use a nonnegative round')
    if args.seed_suite != 'vllm' and not args.expected_checkpoint_sha256:
        parser.error('an alternate generator must reproduce the verified checkpoint identity')
    if args.reuse_checkpoint and (not args.expected_checkpoint_sha256 or args.diagnostic_config):
        parser.error('checkpoint reuse requires the frozen, full model')
    if args.hf_reference and args.stage != 'correctness':
        parser.error('teacher-forced reference is a correctness process')
    if args.numeric_gate and args.stage != 'correctness':
        parser.error('the numeric gate belongs to a correctness process')
    if args.routing_export and (args.stage != 'correctness' or args.model not in
                                ('olmoe_1b_7b', 'qwen3_30b_a3b')):
        parser.error('routing evidence belongs to MoE correctness')
    if args.routing_tokens and (not args.routing_export or not args.numeric_gate):
        parser.error('fixed routing tokens require the R11 numeric gate and export')
    check_hf = args.stage == 'correctness' and (args.round == 0 or args.hf_reference or args.numeric_gate)
    diagnostic = args.stage == 'diagnose'
    tuning = args.stage in ('tune-prefill', 'tune-decode')
    independent = args.stage in ('measure', 'candidate-timing')
    timing = independent or tuning
    if (not diagnostic and (args.prefill_mode != 'L1' or args.sanitizer or args.cuda_gdb)
            or diagnostic and args.prefill == 'prompt'):
        parser.error('prefill executor and sanitizer diagnostics require an independent prefill artifact')
    for tool in (args.sanitizer, args.cuda_gdb):
        if tool and not os.access(tool, os.X_OK):
            parser.error(f'diagnostic executable is unavailable: {tool}')
    if args.diagnostic_config and (not diagnostic or args.model != 'qwen3_30b_a3b'):
        parser.error('the diagnostic config is reserved for a reduced Qwen3-30B prefill probe')
    decode, decode_id, plan = artifact(args.decode)
    with args.target.open() as stream:
        target = json.load(stream)
    if target['arch_tag'] != 'sm_80' or 'A800' not in target['device_provenance']['name']:
        parser.error('require a separate A800 target from the guarded device probe')
    if plan.get('runtime_batch') != args.batch or plan.get('native_origin') not in ('fixed', 'solver'):
        parser.error('decode plan must be native and match the specified batch')
    if args.prefill == 'prompt':
        if args.stage == 'tune-prefill':
            parser.error('prefill tuning requires an independent library')
        if args.model != 'olmoe_1b_7b':
            parser.error('the frozen model protocol uses an independent prefill library')
        prefill, prefill_id = 'prompt', {}
    else:
        prefill, prefill_id, prefill_plan = artifact(Path(args.prefill))
        if not (prefill_plan.get('phase') == 'prefill' and
                prefill_plan['batch_lo'] <= args.batch <= prefill_plan['batch_hi']):
            parser.error('prefill plan must match the batch')

    kind = ('diagnostic' if diagnostic else 'paired_tuning' if tuning else
            'formal_timing' if independent else 'correctness')
    identity = dict(protocol_version='a800-single-v2', kind=kind, model=args.model,
                    batch=args.batch, round=args.round, stage=args.stage,
                    decode_source_sha=decode_id.get('source', {}).get('head'),
                    decode_source_digest=decode_id.get('source', {}).get('source_digest'),
                    decode_coordinate_sha256=decode_id.get('plan', {}).get('coordinates_sha256'),
                    target_sha256=sha(args.target),
                    decode_sha256=decode_id['binary_sha256'],
                    prefill_sha256=prefill_id.get('binary_sha256', 'prompt'))
    if args.routing_export:
        args.routing_export = args.routing_export.resolve(strict=True)
        identity['routing_export_sha256'] = sha(args.routing_export)
    if args.routing_tokens:
        args.routing_tokens = args.routing_tokens.resolve(strict=True)
        identity['routing_tokens_sha256'] = sha(args.routing_tokens)
    if args.host_preparation:
        args.host_preparation = args.host_preparation.resolve(strict=True)
    artifact_key = hashlib.sha256(json.dumps(
        [identity[key] for key in ('target_sha256', 'decode_sha256', 'prefill_sha256')],
        separators=(',', ':')).encode()).hexdigest()[:20]
    native_root = RUN/'native'
    relative_scope = Path(args.model)/f'B{args.batch}'/artifact_key
    stage = f'diagnose-{args.prefill_mode.lower()}' if diagnostic else args.stage
    receipt_name = f'{stage}-{args.round}.json'
    if timing:
        selected = (plan['native_origin'] == 'solver' and
                    plan['native_selection'].get('selection') == 'measure')
        if (not (selected or args.stage == 'tune-decode') or not target.get('sm80_native_constants')
                or decode_id['target_sha256'] != identity['target_sha256']):
            parser.error('timing requires the solver-selected decode and its calibrated target')
    if independent and args.mode == 'run':
        receipts = [json.loads(path.read_text()) for path in
                    native_root.glob(f'*/{relative_scope}/correctness-*.json')]
        fixed = ('decode_sha256', 'prefill_sha256', 'target_sha256')
        accepted = [result for result in receipts if result['pass']
                    and all(result[key] == identity[key] for key in fixed)]
        if len({result['tokens_sha256'] for result in accepted}) > 1:
            parser.error('fresh correctness processes disagreed on generated tokens')
        if not any(result.get('hf_pass') for result in accepted):
            parser.error('formal timing requires an accepted teacher-forced reference')
    config, prompt = inputs(args.model)
    description = inspect_inputs(INVENTORY)[args.model]
    if args.diagnostic_config:
        source = json.loads((config/'config.json').read_text())
        config = args.diagnostic_config.resolve(strict=True)
        reduced = json.loads((config/'config.json').read_text())
        layers = reduced.get('num_hidden_layers')
        if type(layers) is not int or not 0 < layers < source['num_hidden_layers'] or (
                reduced != dict(source, num_hidden_layers=layers)):
            parser.error('the diagnostic model may only reduce the number of layers')
        description = dict(description, config_sha256=sha(config/'config.json'))
    if args.host_preparation:
        from tilemega.serving.host_preparation import validate_artifact_binding

        contents = args.host_preparation.read_bytes()
        identities = (prefill_id, decode_id) if prefill_id else (decode_id,)
        validate_artifact_binding(json.loads(contents), json.loads((config/'config.json').read_text()),
                                  args.batch, plan['capacity'], identities)
        identity['host_preparation_sha256'] = hashlib.sha256(contents).hexdigest()
    name = f'native-{stage}-r{args.round}'
    tokens = 1 if diagnostic else 1024
    hf_positions = min(256, tokens) if args.numeric_gate and args.model == 'qwen3_30b_a3b' else tokens
    if args.numeric_gate:
        identity.update(numeric_gate='sm90-c1-c2-v1', hf_positions=hf_positions)
    if args.reuse_checkpoint:
        identity['checkpoint_supervision'] = 'allocation-sealed'
    if args.mode == 'plan':
        print(json.dumps(dict(identity, **description, prefill=str(prefill), decode=str(decode),
                              calibration_uuid=target['device_provenance']['uuid'],
                              timeout_s=args.timeout_s, max_tokens=tokens, step=name, teacher_forced=check_hf,
                              sanitizer=str(args.sanitizer) if args.sanitizer else None,
                              cuda_gdb=str(args.cuda_gdb) if args.cuda_gdb else None), indent=2))
        return 0

    if any(native_root.glob(f'*/{relative_scope}/{receipt_name}')):
        print('this artifact pair already has an accepted receipt for this round')
        return 0
    policy = device_policy()
    dev = calibrated_device(policy, target)
    scope = native_root/dev['uuid']/relative_scope
    receipt = scope/receipt_name
    py = str(ENVS/'tilemega/bin/python')
    seed_py = str(ENVS/args.seed_suite/'bin/python')
    seed_hash = ['--expected-sha256', args.expected_checkpoint_sha256] if args.expected_checkpoint_sha256 else []
    checkpoint_env = {}
    if args.reuse_checkpoint:
        from tilemega.serving.random_checkpoint import ALGORITHM, validate_live_checkpoint

        checkpoint = allocation_checkpoint()
        if os.environ.get('TILEMEGA_RANDOM_CHECKPOINT') != str(checkpoint):
            raise ValueError('checkpoint must belong to the current allocation')
        live = validate_live_checkpoint(checkpoint)
        if (not live['sealed_memfds'] or live['algorithm'] != ALGORITHM or live['seed'] != 0
                or live['checkpoint_sha256'] != args.expected_checkpoint_sha256
                or live['config_sha256'] != description['config_sha256']
                or live['total_bytes'] != description['checkpoint_bytes']):
            raise ValueError('the allocation checkpoint differs from the frozen model')
        checkpoint_env['TILEMEGA_RANDOM_CHECKPOINT'] = str(checkpoint)
    run = scope/'allocations'/os.environ['SLURM_JOB_ID']/name
    run.mkdir(parents=True, exist_ok=False)
    policy_path = run/'guard-policy.json'
    write_json(policy_path, policy)
    child = [py, '-m', 'tilemega.serving.r16_run', '--prefill', str(prefill), '--decode', str(decode),
             '--prompt-ids', str(prompt), '--batch', str(args.batch), '--max-tokens', str(tokens),
             '--mode', 'L1', '--prefill-mode', args.prefill_mode, '--decode-loop', '1',
             '--warmup', '1', '--repeats', '3' if timing else '1',
             '--round-index', str(args.round), '--out', '{attempt_dir}/job/measurements',
             '--system-label', 'TM-sm80', '--policy', str(policy_path)]
    if args.private_stream:
        child.append('--private-stream')
    if args.host_preparation:
        child.extend(('--host-preparation', str(args.host_preparation),
                      '--host-preparation-sha256', identity['host_preparation_sha256']))
    if args.stage == 'correctness':
        child.append('--check-step-loop')
        if args.numeric_gate:
            child.extend(('--check-executors', '--check-advertised-executors'))
        if args.routing_export:
            child.extend(('--routing-export', str(args.routing_export)))
        if args.routing_tokens:
            child.extend(('--routing-tokens', str(args.routing_tokens)))
        if check_hf:
            child.extend(('--hf-check', '--hf-positions', str(hf_positions)))
    if args.sanitizer:
        child = [str(args.sanitizer), '--tool', 'memcheck', '--kernel-name',
                 f'kns=tilemega_{args.prefill_mode.lower()}_kernel', '--launch-count', '1',
                 '--error-exitcode', '86', '--log-file', '{attempt_dir}/prefill-memcheck.log', *child]
    if args.cuda_gdb:
        child = [str(args.cuda_gdb), '-nx', '-batch', '-ex', 'set confirm off',
                 '-ex', 'set disable-randomization off', '-ex', 'set debuginfod enabled off',
                 '-ex', 'set cuda api_failures stop', '-ex', 'run', '-ex', 'info cuda kernels',
                 '-ex', 'info cuda threads', '-ex', 'bt', '-ex', 'p/x $pc',
                 '-ex', 'info symbol $pc', '-ex', 'x/10i $pc-0x60', '--args', *child]
    if not args.reuse_checkpoint:
        child = [seed_py, '-m', 'tilemega.serving.random_checkpoint', '--config', str(config),
                 '--out', '{attempt_dir}/checkpoint', '--seed', '0',
                 '--max-gib', str(math.ceil(description['checkpoint_bytes']/2**30)), *seed_hash, '--', *child]
    command = ['taskset', '--cpu-list', dev['cpus'], py,
               str(ROOT/'scripts/run_r16_with_device_identity.py'),
               '--out', '{attempt_dir}/job', '--guard-policy', str(policy_path),
               '--expected-uuid', dev['uuid'], '--expected-pci', dev['pci'],
               '--numa-node', dev['numa_node'], '--', *child]
    env = dict(TILEMEGA_DEVICE_INDEX=str(dev['index']), TILEMEGA_NUMA_NODE=dev['numa_node'],
               TILEMEGA_GPU_PID_MODE='strict', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', **checkpoint_env)
    if diagnostic:
        env['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ.update(TILEMEGA_DEVICE_INDEX=env['TILEMEGA_DEVICE_INDEX'],
                      TILEMEGA_GPU_LOCK=policy['gpu_lock_path'])
    queue = run/'queue'
    queue.mkdir()
    state = run/'results'
    needs = min(dev['memory_mib'] - policy['max_hidden_mib'],
                math.ceil(description['checkpoint_bytes']/2**20) + 8192)
    write_json(queue/'queue_native.json', [dict(name=name, gpu=True, timing=timing,
               env=env, command=command, timeout_s=args.timeout_s, heartbeat_timeout_s=min(300, args.timeout_s),
               needs_free_mib=needs, cwd=str(ROOT))])
    schedule(['--queue-dir', str(queue), '--out', str(state), '--policy', str(policy_path),
              '--deadline-hours', str(args.budget_s/3600), '--release-on-blocked'])
    result = json.loads((state/'state.json').read_text())[name]
    if result['status'] != 'done':
        return 1
    evidence = Path(result['out'])/'job/measurements'
    measurement = evidence/f'B{args.batch}'/'measurements.json'
    if not diagnostic:
        verify_request_tokens(measurement, args.batch, 3 if timing else 1)
    hf_pass = False
    if args.stage == 'correctness':
        c2_path = evidence/'c2.json'
        c2 = json.loads(c2_path.read_text())
        if not (c2['pass'] and c2['same_binary']
                and {trial['decode_loop_used'] for trial in c2['checks']} == {False, True}):
            return 1
        if check_hf:
            c1_path = evidence/'c1.json'
            c1 = json.loads(c1_path.read_text())
            hf_pass = c1['pass']
            if not hf_pass:
                return 1
        if args.numeric_gate:
            from a800_r11_gate import executor_agreement

            if c1['positions'] != args.batch*hf_positions or not executor_agreement(c2):
                return 1
            identity.update(executor_agree=True, executor_proof_sha256=sha(c2_path),
                            hf_proof_sha256=sha(c1_path))
        if args.routing_export:
            routes = json.loads((evidence/'routing.json').read_text())
            route_inputs = (sha(args.routing_tokens) if args.routing_tokens else sha(evidence/'reference_tokens.json'))
            if (routes['export_sha256'] != identity['routing_export_sha256']
                    or routes['trajectory_sha256'] != route_inputs):
                raise ValueError('routing evidence changed its model export or trajectory')
            identity['router_indices_sha256'] = routes['indices_sha256']
            identity['routing_trajectory_sha256'] = routes['trajectory_sha256']
            identity['routing_proof_sha256'] = sha(evidence/'routing.json')
    write_json(receipt, dict(identity, **{'pass': True}, job=os.environ['SLURM_JOB_ID'], hf_pass=hf_pass,
               process_receipt_sha256=sha(measurement) if measurement.is_file() else sha(evidence/'reference_tokens.json'),
               device=dev, device_uuid=dev['uuid'],
               evidence=str(evidence), tokens_sha256=sha(evidence/'reference_tokens.json')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
