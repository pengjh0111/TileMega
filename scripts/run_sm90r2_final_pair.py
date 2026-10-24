#!/usr/bin/env python3
"""Stage and collect final TileMega/reference pairs with same-session blanks."""
import argparse
import copy
import os
from pathlib import Path
import statistics
import subprocess
import time
import uuid

import run_sm90r2_final_e2e as final
from run_r16_with_device_identity import annotate
from sm90d1_common import ROOT, PYTHON, read, sha, write

INDEX = 'configs/serving/sm90d1/reference/artifacts/index.json'
REFERENCE_HEAD = '966f0b9ab3936964c7bc4ca9c64ad467552747b0'


def validate_reference_identity(identity, binding, config, phase, batch, binary_sha256):
    plan = identity['plan']
    if (identity['trace'] or plan['phase'] != phase or plan['batch_lo'] != batch or plan['batch_hi'] != batch
            or plan['capacity'] != 1088 or plan['seq'] != (64 if phase == 'prefill' else 1)
            or identity['source']['head'] != REFERENCE_HEAD or binding['head'] != REFERENCE_HEAD
            or identity['artifact_id'] != binding['artifact_id'] or binary_sha256 != binding['binary_sha256']
            or final.prefill_signature(plan) != config['coordinates'][phase]):
        raise ValueError('reference artifact is not the frozen indexed build: ' + phase)


def validate_reference(record, candidate, base):
    index_path = final.checked_file(record['reference_index'], base)
    committed = subprocess.check_output(['git', '-C', str(ROOT), 'show',
                                        candidate['manifest']['source_head'] + ':' + INDEX])
    if index_path.read_bytes() != committed:
        raise ValueError('reference index differs from the frozen repository index')
    index = read(index_path)
    config_path = final.checked_file(record['reference_config'], base); config = read(config_path)
    m = candidate['manifest']
    if (index['source_head'] != REFERENCE_HEAD or (config['model'], config['batch']) != (m['model'], m['batch'])
            or config['model_config_sha256'] != m['model_config']['sha256']
            or config['prompt_ids_sha256'] != m['prompt']['sha256']
            or config['execution'] != dict(decode='L1', loop=1, prefill='L1')):
        raise ValueError('reference model/prompt/execution does not match the candidate')
    artifacts = {}
    for phase in ('prefill', 'decode'):
        path = final.checked_file(record['reference_' + phase], base)
        identity = final.verify(path)
        bindings = [r for r in index['bindings'] if r['config_sha256'] == sha(config_path)
                    and r['phase'] == phase]
        if len(bindings) != 1:
            raise ValueError('reference index must bind exactly one artifact for each phase')
        validate_reference_identity(identity, bindings[0], config, phase, m['batch'], sha(path))
        artifacts[phase] = dict(binary=path)
    if read(str(artifacts['prefill']['binary']) + '.plan.json') != read(candidate['files']['prefill_reference_plan']):
        raise ValueError('candidate prefill must preserve this indexed reference plan')
    reference = copy.deepcopy(candidate)
    reference['manifest'].update(prefill=dict(binary=record['reference_prefill']),
                                 decode=dict(binary=record['reference_decode']))
    reference['artifacts'] = artifacts
    return reference


def validate_manifest(path):
    path = path.resolve(); record = read(path); base = path.parent
    if record['schema'] != 1 or record['kind'] != 'sm90r2_final_pair':
        raise ValueError('unsupported TileMega final pair manifest')
    candidate = final.validate_manifest(final.checked_file(record['candidate_manifest'], base))
    processes = record['processes']
    if type(processes) is not int or processes not in (3, 5) or processes < final.process_count(candidate):
        raise ValueError('final pairs need three processes, or five for Llama-8B/near-threshold retries')
    reference = validate_reference(record, candidate, base)
    actual = subprocess.check_output(['git', '-C', str(ROOT), 'show',
                                     candidate['manifest']['source_head'] + ':scripts/' + Path(__file__).name])
    if actual != Path(__file__).read_bytes():
        raise ValueError('paired runner differs from the bound final source HEAD')
    return dict(path=path, manifest=record, manifest_sha256=sha(path), candidate=candidate,
                reference=reference, processes=processes, runner_sha256=sha(__file__))


def pair_specs(processes):
    return [dict(kind='blank_before', fresh_process=0)] + [
        dict(kind='comparison', fresh_process=i) for i in range(processes)] + [
        dict(kind='blank_after', fresh_process=0)]


def prepare_queue(bound, gpu, out):
    candidate = bound['candidate']; policy = read(candidate['files']['guard_policy'])
    node = final.validate_gpu(gpu, policy)
    if out.exists():
        raise FileExistsError('retain the previous paired preparation')
    session = uuid.uuid4().hex; specs = pair_specs(bound['processes']); jobs = []; previous = []
    for i, spec in enumerate(specs):
        name = f'final-pair-{session[:12]}-{i:02d}'
        command = ['numactl', f'--cpunodebind={node}', PYTHON,
                   str(ROOT / 'scripts/run_r16_with_device_identity.py'),
                   '--guard-policy', str(candidate['files']['guard_policy']), '--out', '{attempt_dir}/pair',
                   '--expected-uuid', policy['idle_power_mapping_provenance']['uuid_by_gpu'][str(gpu)],
                   '--expected-pci', f'0000:{final.PCI[gpu]}:00.0', '--numa-node', str(node),
                   '--allow-no-measurements', '--', PYTHON, str(Path(__file__).resolve()),
                   'run', '--manifest', str(bound['path']), '--preparation', str(out),
                   '--pair-index', str(i), '--out', '{attempt_dir}/pair']
        jobs.append(dict(name=name, gpu=True, timing=True, timeout_s=7200, needs_free_mib=32768,
                         cwd=str(ROOT), after=previous, env=dict(CUDA_VISIBLE_DEVICES=str(gpu),
                         TILEMEGA_DEVICE_INDEX=str(gpu), TILEMEGA_NUMA_NODE=str(node),
                         TILEMEGA_GPU_PID_MODE='namespace_limited', PYTHONPATH=str(ROOT / 'python'),
                         OMP_NUM_THREADS='8'), command=command))
        previous = [name]
    out.mkdir(parents=True); write(out / 'queue/queue_final_pair.json', jobs)
    write(out / 'preparation.json', dict(schema=1, session_id=session, gpu=gpu, specs=specs,
          manifest_sha256=bound['manifest_sha256'], runner_sha256=bound['runner_sha256'],
          candidate_runner_sha256=candidate['runner_sha256'], jobs=[r['name'] for r in jobs],
          queue_sha256=sha(out / 'queue/queue_final_pair.json'), gpu_jobs_started=0,
          active_queue_publication=False, scheduler_started=False))


def checked_preparation(bound, directory):
    prepared = read(directory / 'preparation.json')
    if (prepared['manifest_sha256'] != bound['manifest_sha256']
            or prepared['runner_sha256'] != bound['runner_sha256']
            or prepared['candidate_runner_sha256'] != bound['candidate']['runner_sha256']
            or prepared['specs'] != pair_specs(bound['processes'])
            or len(prepared['jobs']) != len(prepared['specs'])
            or sha(directory / 'queue/queue_final_pair.json') != prepared['queue_sha256']):
        raise ValueError('paired preparation changed after admission')
    return prepared


def reference_run(bound, out, fresh):
    result = out / 'reference-process.json'
    if result.exists() or (out / 'measurements').exists():
        raise FileExistsError('retain the prior reference process')
    command = final.command_for(bound, out, fresh)
    evidence = dict(kind='sm90r2_frozen_reference_process', status='running', started=time.time(),
                    command=command, user_final_e2e_amendment=final.AMENDMENT)
    write(result, evidence); code = subprocess.call(command, cwd=ROOT)
    evidence.update(exit_code=code, ended=time.time(), status='failed' if code else 'awaiting_outer_guard')
    if not code:
        evidence.update(final.metrics_from_job(bound, out, fresh))
    write(result, evidence)
    return code


def run_pair(bound, preparation, index, out):
    prepared = checked_preparation(bound, preparation)
    if not 0 <= index < len(prepared['specs']):
        raise ValueError('invalid paired job index')
    spec = prepared['specs'][index]; candidate = bound['candidate']
    policy = read(candidate['files']['guard_policy']); gpu = int(os.environ['TILEMEGA_DEVICE_INDEX'])
    node = final.validate_gpu(gpu, policy); final.require_guard(policy, gpu); final.require_numa(node)
    physical_path = out / 'physical-device-identity.json'; physical = read(physical_path)
    if (gpu != prepared['gpu'] or physical['wrapper_pid'] != os.getppid()
            or physical['wrapper_sha256'] != sha(ROOT / 'scripts/run_r16_with_device_identity.py')
            or physical['gpu_index'] != gpu or physical['numa_node'] != node
            or physical['cuda_uuid'] != policy['idle_power_mapping_provenance']['uuid_by_gpu'][str(gpu)].removeprefix('GPU-')
            or physical['cuda_pci_bus_id'].lower() != f'0000:{final.PCI[gpu]}:00.0'
            or physical['guard_policy_sha256'] != candidate['manifest']['guard_policy']['sha256']
            or os.environ.get('TILEMEGA_TIMING') != '1' or os.environ.get('TILEMEGA_NUMA_NODE') != str(node)):
        raise ValueError('paired process lacks the declared physical wrapper/device/NUMA binding')
    if any(os.environ.get(k) for k in ('TILEMEGA_RUNTIME_MANIFEST', 'TILEMEGA_M0_BARRIER_DIR')):
        raise ValueError('final pairs cannot inherit a different runtime or baseline barrier')
    path = out / 'paired-process.json'
    if path.exists():
        raise FileExistsError('retain both arms; retries need a new complete pair attempt')
    evidence = dict(schema=1, session_id=prepared['session_id'], spec=spec, status='running',
                    manifest_sha256=bound['manifest_sha256'], preparation_sha256=sha(preparation / 'preparation.json'),
                    runner_sha256=bound['runner_sha256'], arms=[], started=time.time(),
                    user_final_e2e_amendment=final.AMENDMENT)
    write(path, evidence)
    for role in ('reference', 'second'):
        arm = out / role; arm.mkdir()
        # Both helpers run in this physical wrapper's direct child, not another wrapper subprocess.
        (arm / physical_path.name).write_bytes(physical_path.read_bytes())
        is_candidate = role == 'second' and spec['kind'] == 'comparison'
        artifact = candidate if is_candidate else bound['reference']
        code = (final.run(candidate, arm, spec['fresh_process'], processes=bound['processes']) if is_candidate
                else reference_run(artifact, arm, spec['fresh_process']))
        if code:
            evidence.update(status='failed', exit_code=code, ended=time.time()); write(path, evidence)
            return code
        annotate(arm, physical_path)
        metrics = final.metrics_from_job(artifact, arm, spec['fresh_process'])
        evidence['arms'].append(dict(role='candidate' if is_candidate else 'reference', **metrics))
        write(path, evidence)
    first, second = evidence['arms']
    if (first['gpu_process_pid'] == second['gpu_process_pid'] or not first['checkpoint']
            or first['checkpoint'] != second['checkpoint']):
        raise ValueError('paired arms require fresh GPU processes with the same checkpoint')
    validate_manifest(bound['path'])
    evidence.update(status='awaiting_outer_guard', exit_code=0, ended=time.time()); write(path, evidence)
    return 0


def process_range(values):
    if not values or any(not isinstance(v, (int, float)) or not final.math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('positive finite fresh-process timings are required')
    return (max(values) - min(values)) / statistics.median(values) * 100


def paired_decision(pairs, blanks):
    if (len(pairs) not in (3, 5) or len(blanks) != 2
            or any(len(pair) != 2 for pair in pairs + blanks)):
        raise ValueError('final comparison needs three/five pairs and both bookend blanks')
    reference = [r[0] for r in pairs]; candidate = [r[1] for r in pairs]
    blank = [v for pair in blanks for v in pair]
    candidate_range = process_range(candidate); reference_range = process_range(reference)
    blank_range = process_range(blank); session_range = process_range(reference + blank)
    threshold = max(.5, session_range, reference_range, candidate_range)
    deltas = [(after / before - 1) * 100 for before, after in pairs]
    delta = statistics.median(deltas)
    needs_five = len(pairs) < 5 and abs(delta) <= 2 * threshold
    return dict(paired_change_pct=delta, process_paired_changes_pct=deltas,
                reference_process_range_pct=reference_range, candidate_process_range_pct=candidate_range,
                blank_process_range_pct=blank_range, same_session_reference_range_pct=session_range,
                decision_threshold_pct=threshold, needs_five_processes=needs_five,
                decision='needs_five_processes' if needs_five else
                         'indistinguishable' if abs(delta) <= threshold else 'faster' if delta < 0 else 'slower',
                g_perf_scope='one-cell paired TPOT diagnostic; no global G-PERF claim')


def collect(bound, preparation, state_path, out):
    prepared = checked_preparation(bound, preparation); states = read(state_path)
    if out.exists():
        raise FileExistsError('retain the prior paired collection')
    pairs, blanks, records, pids, uuids = [], [], [], set(), set()
    checkpoints = []; main_metrics = dict(reference=[], candidate=[])
    for i, (name, spec) in enumerate(zip(prepared['jobs'], prepared['specs'])):
        state = states[name]
        if state['status'] != 'done' or state['exit_code'] != 0:
            raise ValueError('paired scheduler job is incomplete: ' + name)
        attempt = Path(state['out']); job = attempt / 'pair'
        guard = read(attempt / 'guard_result.json'); pair = read(job / 'paired-process.json')
        physical = read(job / 'physical-device-identity.json'); wrapper = read(job / 'physical-device-command.json')
        if (guard['code'] != 0 or wrapper['actual_returncode'] != 0
                or pair['status'] != 'awaiting_outer_guard' or pair['exit_code'] != 0
                or pair['session_id'] != prepared['session_id'] or pair['spec'] != spec
                or pair['manifest_sha256'] != bound['manifest_sha256']
                or pair['runner_sha256'] != bound['runner_sha256']
                or pair['preparation_sha256'] != sha(preparation / 'preparation.json')
                or physical['gpu_index'] != prepared['gpu'] or len(pair['arms']) != 2):
            raise ValueError('paired attempt lacks successful bound guard/wrapper/session evidence')
        times = []
        for arm_index, role in enumerate(('reference', 'second')):
            artifact = bound['candidate'] if role == 'second' and spec['kind'] == 'comparison' else bound['reference']
            arm = job / role; row = final.metrics_from_job(artifact, arm, spec['fresh_process'])
            value = read(row['measurement'])
            process = read(arm / ('final-e2e-process.json' if artifact is bound['candidate'] else 'reference-process.json'))
            expected_role = 'candidate' if artifact is bound['candidate'] else 'reference'
            if (value['physical_device_identity'] != physical
                    or value['physical_device_identity_sha256'] != sha(job / 'physical-device-identity.json')
                    or process['exit_code'] != 0 or process['status'] != 'awaiting_outer_guard'
                    or process['command'] != final.command_for(artifact, arm, spec['fresh_process'])
                    or process['user_final_e2e_amendment'] != final.AMENDMENT
                    or pair['arms'][arm_index] != dict(role=expected_role, **row)
                    or row['gpu_process_pid'] in pids):
                raise ValueError('paired arms lack fresh matching physical execution evidence')
            if artifact is bound['candidate'] and (
                    process['manifest_sha256'] != artifact['manifest_sha256']
                    or process['runner_sha256'] != artifact['runner_sha256']
                    or process['fresh_process'] != spec['fresh_process']):
                raise ValueError('candidate arm lost its strict final manifest/source provenance')
            pids.add(row['gpu_process_pid']); checkpoints.append(row['checkpoint'])
            times.append(row['metrics']['tpot_seconds'])
            if spec['kind'] == 'comparison':
                main_metrics['reference' if role == 'reference' else 'candidate'].append(row['metrics'])
        (pairs if spec['kind'] == 'comparison' else blanks).append(times)
        uuids.add(physical['cuda_uuid'])
        records.append(dict(job=name, spec=spec, paired_receipt=str(job / 'paired-process.json'),
                            paired_receipt_sha256=sha(job / 'paired-process.json'),
                            guard_sha256=sha(attempt / 'guard_result.json'), ownership_verified=guard['ownership_verified']))
    if len(uuids) != 1 or not checkpoints[0] or any(c != checkpoints[0] for c in checkpoints):
        raise ValueError('paired cohort changed its physical GPU or checkpoint')
    write(out, dict(schema=1, kind='sm90r2_final_tilemega_pair', session_id=prepared['session_id'],
          evidence='verified guarded same-card TileMega/reference paired timing',
          manifest_sha256=bound['manifest_sha256'], model=bound['candidate']['manifest']['model'],
          batch=bound['candidate']['manifest']['batch'], processes_per_arm=bound['processes'],
          physical_uuid=uuids.pop(), user_final_e2e_amendment=final.AMENDMENT, records=records,
          ownership_verified=all(r['ownership_verified'] for r in records),
          arm_metrics={role: {key: statistics.median(r[key] for r in rows) for key in rows[0]}
                       for role, rows in main_metrics.items()},
          paired_tpot_seconds=pairs, reference_blank_tpot_seconds=blanks, **paired_decision(pairs, blanks)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('validate', 'queue', 'run', 'collect'))
    parser.add_argument('--manifest', type=Path, required=True); parser.add_argument('--out', type=Path)
    parser.add_argument('--gpu', type=int); parser.add_argument('--preparation', type=Path)
    parser.add_argument('--state', type=Path); parser.add_argument('--pair-index', type=int)
    args = parser.parse_args(); bound = validate_manifest(args.manifest)
    if args.action == 'validate':
        print('Validated candidate final contract and frozen TileMega reference; no GPU work started.')
        return 0
    if args.out is None or not args.out.resolve().is_relative_to(ROOT / 'runs/sm90r2'):
        parser.error('--out must retain a new result under runs/sm90r2')
    out = args.out.resolve()
    if args.action == 'queue':
        if args.gpu is None: parser.error('queue requires --gpu')
        prepare_queue(bound, args.gpu, out)
    else:
        if args.preparation is None: parser.error('--preparation is required')
        if args.action == 'run':
            if args.pair_index is None: parser.error('run requires --pair-index')
            return run_pair(bound, args.preparation.resolve(), args.pair_index, out)
        if args.state is None: parser.error('collect requires --state')
        collect(bound, args.preparation.resolve(), args.state.resolve(), out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
