#!/usr/bin/env python3
"""Run CTG pipeline search, linked occupancy and measured selection under Slurm."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from a100_baselines import BATCHES, MODELS, ROOT, sha
from a800_baselines import ENVS, RUN, calibrated_device, device_policy
from a800_native import artifact
from tilemega.bench.scheduler import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'prepare', 'reprepare', 'run', 'finalize', 'bind-selector'))
    parser.add_argument('--model', required=True, choices=MODELS)
    parser.add_argument('--batch', required=True, type=int, choices=BATCHES)
    parser.add_argument('--export', required=True, type=Path)
    parser.add_argument('--target', required=True, type=Path)
    parser.add_argument('--prefill', required=True)
    parser.add_argument('--cli', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--pin-theta', type=Path)
    parser.add_argument('--seed-coordinates', type=Path)
    parser.add_argument('--submission-plan', type=Path)
    parser.add_argument('--prepared-shortlist', type=Path)
    parser.add_argument('--reprepare-receipt', type=Path)
    parser.add_argument('--finalize-receipt', type=Path)
    parser.add_argument('--defer-finalize', action='store_true')
    parser.add_argument('--measure-rank', type=int)
    parser.add_argument('--search-budget-ms', type=int, default=600000)
    parser.add_argument('--compile-jobs', type=int, choices=(1, 2), default=2)
    parser.add_argument('--top', type=int, choices=range(2, 7), default=2)
    parser.add_argument('--timeout-s', type=int, default=720)
    args = parser.parse_args()
    source, target, cli = (path.resolve(strict=True) for path in (args.export, args.target, args.cli))
    output = args.out.resolve()
    if args.timeout_s <= 0 or args.search_budget_ms <= 0 or output.suffix != '.so' or output.exists():
        parser.error('provide a positive process budget and a new .so output')
    spec = json.loads(target.read_text())
    fingerprint = spec['device_provenance']
    if (spec['arch_tag'] != 'sm_80' or 'A800' not in fingerprint['name']
            or spec.get('sm80_native_constants', {}).get('gpu_uuid') != fingerprint['uuid']
            or not os.access(cli, os.X_OK)):
        parser.error('provide an executable compiler and a calibrated A800 target')
    if args.prefill != 'prompt':
        library, identity, _ = artifact(Path(args.prefill))
        prefill, prefill_sha = str(library), identity['binary_sha256']
    elif args.model == 'olmoe_1b_7b':
        prefill = prefill_sha = 'prompt'
    else:
        parser.error('this model requires an independent prefill library')

    py = str(ENVS/'tilemega/bin/python')
    select = shlex.join((py, str(ROOT/'scripts/a800_select.py'), '--model', args.model,
                         '--batch', str(args.batch), '--prefill', prefill, '--target', str(target)))
    if args.measure_rank is not None and (
            args.mode not in ('plan', 'run') or not args.defer_finalize
            or not args.prepared_shortlist or not 0 <= args.measure_rank < args.top):
        parser.error('a rank measurement requires an existing deferred shortlist')
    if args.mode == 'bind-selector':
        if (os.environ.get('SLURM_JOB_ID') or not args.prepared_shortlist or args.submission_plan):
            parser.error('bind a separate selector for an existing shortlist on the login node')
        prepared = args.prepared_shortlist.resolve(strict=True)
        document = json.loads(prepared.read_text())
        abi_path = prepared.parent/'weight-abi.json'
        abi = json.loads(abi_path.read_text())
        scope = document['scope']
        old = shlex.split(scope['measure_command'])
        new = shlex.split(select)
        if (document['schema'] != 1 or scope['compiler_sha256'] != sha(cli)
                or scope['target_sha256'] != sha(target) or scope['export_sha256'] != sha(source)
                or abi['prefill_sha256'] != prefill_sha
                or scope['batch'] != args.batch or len(old) != len(new)
                or old[0] != new[0] or Path(old[1]).name != 'a800_select.py' or old[2:] != new[2:]):
            parser.error('selector rebinding must preserve the executable and all frozen arguments')
        scope['measure_command'] = select
        document['selector_rebind'] = dict(
            prepared_sha256=sha(prepared), previous_command=shlex.join(old),
            selector_sha256=sha(ROOT/'scripts/a800_select.py'), weight_abi=str(abi_path),
            weight_abi_sha256=sha(abi_path))
        output.parent.mkdir(parents=True, exist_ok=True)
        bound = output.with_suffix('.prepared.json')
        if bound.exists():
            parser.error('retain the previous selector binding')
        write_json(bound, document)
        return 0
    command = [str(cli), 'compile', str(source), str(output), '--solve', str(target),
               '--serving', 'decode', '--emit', 'serving', '--batch', str(args.batch),
               '--tp', '1', '--runtime-target', str(target), '--capacity', '1088',
               '--prompt-len', '64', '--past-range', '64:1086', '--search', 'pipeline',
               '--selection', 'measure', '--measure-top', str(args.top), '--measure-cmd', select,
               '--nvcc-split-compile', str(args.compile_jobs), '--search-budget-ms', str(args.search_budget_ms)]
    if args.pin_theta:
        pinned = args.pin_theta.resolve(strict=True)
        command.extend(('--pin-theta', str(pinned)))
    if args.seed_coordinates:
        seed = args.seed_coordinates.resolve(strict=True)
        command.extend(('--seed-coordinates', str(seed)))
    if bool(args.finalize_receipt) != (args.mode == 'finalize'):
        parser.error('finalize requires a verified GPU selection receipt')
    if args.defer_finalize and (args.mode not in ('plan', 'run') or not args.prepared_shortlist):
        parser.error('defer finalization only for a prepared GPU selection')
    if args.defer_finalize:
        command.extend(('--defer-finalize', '1'))
    if bool(args.reprepare_receipt) != (args.mode == 'reprepare'):
        parser.error('reprepare requires the verified occupancy correction receipt')
    if args.mode in ('prepare', 'reprepare'):
        if (os.environ.get('SLURM_JOB_ID') or args.submission_plan
                or bool(args.prepared_shortlist) != (args.mode == 'reprepare')):
            parser.error('prepare on the login node and supply the original shortlist only for reprepare')
        command.extend(('--prepare-shortlist', '1'))
        if args.reprepare_receipt:
            command.extend(('--prepared-shortlist', str(args.prepared_shortlist.resolve(strict=True)),
                            '--reprepare-receipt', str(args.reprepare_receipt.resolve(strict=True))))
        result = subprocess.run(['timeout', '--kill-after=15s', str(args.timeout_s), *command], cwd=ROOT)
        if result.returncode:
            return result.returncode
        from a100_baselines import inputs
        from a800_shortlist_abi import check_shortlist
        path = Path(f'{output}.search')
        config = json.loads((inputs(args.model)[0]/'config.json').read_text())
        report = check_shortlist(prefill, path/'prepared.json', config, args.batch)
        write_json(path/'weight-abi.json', dict(prefill_sha256=prefill_sha, candidates=report))
        return 0
    if args.prepared_shortlist:
        prepared = args.prepared_shortlist.resolve(strict=True)
        command.extend(('--prepared-shortlist', str(prepared)))
    if args.mode == 'finalize':
        if (os.environ.get('SLURM_JOB_ID') or args.submission_plan or not args.prepared_shortlist):
            parser.error('finalize the prepared winner on the login node')
        command.extend(('--finalize-receipt', str(args.finalize_receipt.resolve(strict=True))))
        return subprocess.run(['timeout', '--kill-after=15s', str(args.timeout_s), *command], cwd=ROOT).returncode
    heartbeat_timeout_s = min(540, args.timeout_s)
    selector = ROOT/'scripts/a800_select.py'
    receipt = dict(protocol_version='a800-single-v2', kind='selection', model=args.model,
                   batch=args.batch, prefill_sha256=prefill_sha, select_sha256=sha(selector),
                   source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                   command=command, target=str(target), target_sha256=sha(target),
                   export_sha256=sha(source), cli_sha256=sha(cli), calibration_uuid=fingerprint['uuid'],
                   timeout_s=args.timeout_s, heartbeat_timeout_s=heartbeat_timeout_s)
    prior = os.environ.get('TILEMEGA_PSEL_PRIOR')
    if prior:
        previous = Path(prior).resolve(strict=True)
        receipt['prior_selection'] = str(previous)
        receipt['prior_selection_sha256'] = sha(previous)
    if args.pin_theta:
        receipt['pin_theta_sha256'] = sha(pinned)
    if args.seed_coordinates:
        receipt['seed_coordinates_sha256'] = sha(seed)
    if args.prepared_shortlist:
        receipt['prepared_shortlist_sha256'] = sha(prepared)
        document = json.loads(prepared.read_text())
        rebound = document.get('selector_rebind', {})
        abi_path = Path(rebound.get('weight_abi', prepared.parent/'weight-abi.json'))
        if abi_path.is_file():
            abi = json.loads(abi_path.read_text())
            if abi['prefill_sha256'] != prefill_sha or (rebound.get('weight_abi_sha256') and
                    rebound['weight_abi_sha256'] != sha(abi_path)):
                raise ValueError('prepared host ABI differs from the frozen selector')
            compiled = {row['binary_sha256'] for row in document['receipt']['candidates']
                        if row['status'] == 'compiled'}
            host = {row['binary_sha256']: [row['host_preparation'], row['host_preparation_sha256']]
                    for row in abi['candidates'] if 'host_preparation' in row}
            if host:
                if host.keys() != compiled or any(sha(path) != digest for path, digest in host.values()):
                    raise ValueError('prepared host recipes differ from the compiled shortlist')
                receipt['host_preparation'] = host
    if args.measure_rank is not None:
        receipt['measure_rank'] = args.measure_rank
    if args.mode == 'plan':
        print(json.dumps(receipt, indent=2))
        return 0

    if args.submission_plan and json.loads(args.submission_plan.read_text()) != receipt:
        raise ValueError('queued search artifacts, source, or configuration changed')
    policy = device_policy()
    dev = calibrated_device(policy, spec)
    out = RUN/'search'/os.environ['SLURM_JOB_ID']
    out.mkdir(parents=True, exist_ok=False)
    policy_path = out/'guard-policy.json'
    write_json(policy_path, policy)
    write_json(out/'planned.json', dict(receipt, device=dev))
    env = dict(os.environ, TILEMEGA_DEVICE_INDEX=str(dev['index']),
               TILEMEGA_GPU_LOCK=policy['gpu_lock_path'], TILEMEGA_NUMA_NODE=dev['numa_node'],
               TILEMEGA_TIMING='1', TILEMEGA_GPU_PID_MODE='strict', OMP_NUM_THREADS='2',
               MAX_JOBS='2', CMAKE_BUILD_PARALLEL_LEVEL='2')
    env.pop('TILEMEGA_PSEL_RANK', None)
    env.pop('TILEMEGA_PSEL_HOST_PREPARATION', None)
    if 'host_preparation' in receipt:
        env['TILEMEGA_PSEL_HOST_PREPARATION'] = json.dumps(receipt['host_preparation'])
    if args.measure_rank is not None:
        env['TILEMEGA_PSEL_RANK'] = str(args.measure_rank)
    wrapper = [sys.executable, str(ROOT/'scripts/run_r16_with_device_identity.py'),
               '--out', str(out/'job'), '--guard-policy', str(policy_path),
               '--expected-uuid', dev['uuid'], '--expected-pci', dev['pci'],
               '--numa-node', dev['numa_node'], '--allow-no-measurements', '--']
    guarded = [sys.executable, '-m', 'tilemega.bench.gpu_guard', '--policy', str(policy_path),
               '--out', str(out), '--needs-free-mib', '16384',
               '--timeout-s', str(args.timeout_s), '--heartbeat-timeout-s', str(heartbeat_timeout_s),
               '--', 'taskset', '--cpu-list', dev['cpus'],
               *wrapper, *command]
    return subprocess.run(guarded, cwd=ROOT, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
