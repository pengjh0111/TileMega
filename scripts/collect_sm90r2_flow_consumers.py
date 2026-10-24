#!/usr/bin/env python3
"""Collect connected consumer responses and observer bounds, never pure compute."""
import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import median

ACTIVE = (1, 32, 64, 128)
FIELDS = ('fresh', 'active_ctas', 'observe_empty', 'trial', 'cta', 'warp',
          'entry', 'returned', 'empty0_wait_begin', 'empty0_observed_ready',
          'empty1_wait_begin', 'empty1_observed_ready')
OBSERVERS = FIELDS[8:]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def quantile(values, fraction):
    ordered = sorted(values)
    require(bool(ordered), 'empty samples')
    return ordered[min(len(ordered) - 1, math.floor(fraction * len(ordered)))]


def summarize(values):
    return dict(samples=len(values), median=median(values), p99=quantile(values, .99))


def parse_row(row, fresh):
    require(set(row) == set(FIELDS), 'wrong raw columns')
    require(all(isinstance(v, str) and v.isascii() and v.isdecimal()
                for v in row.values()), 'raw values must be unsigned decimal integers')
    r = {k: int(v) for k, v in row.items()}
    require(r['fresh'] == fresh and r['active_ctas'] in ACTIVE
            and r['observe_empty'] in (0, 1) and 1 <= r['trial'] <= 16
            and 0 <= r['cta'] < r['active_ctas'] and 0 <= r['warp'] < 4,
            'mixed process or invalid observation coordinates')
    require(0 < r['entry'] < r['returned'] < 2**64, 'invalid consumer timestamp order')
    times = tuple(r[k] for k in OBSERVERS)
    if r['observe_empty']:
        require(0 < times[0] <= times[1] <= times[2] <= times[3] < 2**64,
                'empty observer timestamp order failed')
    else:
        require(times == (0, 0, 0, 0), 'off control contains empty observations')
    return r


def cta_observation(rows):
    require(set(rows) == set(range(4)), 'missing consumer warp')
    require(len({tuple(r[k] for k in OBSERVERS) for r in rows.values()}) == 1,
            'inconsistent duplicate observer stamps across warps')
    first = rows[0]
    result = {k: first[k] for k in ('fresh', 'active_ctas', 'observe_empty', 'trial', 'cta')}
    for warp, row in sorted(rows.items()):
        result[f'entry_w{warp}'] = row['entry']
        result[f'returned_w{warp}'] = row['returned']
    result.update({k: first[k] for k in OBSERVERS})
    result['entry_min'] = min(r['entry'] for r in rows.values())
    result['entry_max'] = max(r['entry'] for r in rows.values())
    result['returned_min'] = min(r['returned'] for r in rows.values())
    result['returned_max'] = max(r['returned'] for r in rows.values())
    result['connected_cta_envelope_ns'] = result['returned_max'] - result['entry_min']
    if first['observe_empty']:
        # All 128 readers arrive after entry and before return. Observer-ready
        # is an upper bound; wait-begin need not precede the release itself.
        for slot in (0, 1):
            lower = result['entry_max']
            upper = min(result['returned_min'], first[f'empty{slot}_observed_ready'])
            require(lower <= upper, 'empty release has impossible causal bounds')
            result[f'empty{slot}_release_lower'] = lower
            result[f'empty{slot}_release_upper'] = upper
    return result


def rows_from_raw(path, fresh):
    groups = defaultdict(dict)
    with Path(path).open() as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == list(FIELDS), 'wrong raw header')
        for row in reader:
            r = parse_row(row, fresh)
            key = tuple(r[k] for k in ('active_ctas', 'observe_empty', 'trial', 'cta'))
            require(r['warp'] not in groups[key], 'duplicate CTA/warp observation')
            groups[key][r['warp']] = r
    expected = {(a, o, t, c) for a in ACTIVE for o in (0, 1)
                for t in range(1, 17) for c in range(a)}
    require(set(groups) == expected, 'incomplete trial/CTA/on-off coverage')
    return {key: cta_observation(rows) for key, rows in groups.items()}


def csv_write(path, rows):
    require(bool(rows), 'empty output')
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def option(command, name):
    require(command.count(name) == 1, 'missing or duplicate queue option: ' + name)
    return command[command.index(name) + 1]


def validate_process(folder, fresh, state, job, binding, paths, metadata_validator=None):
    require(state['status'] == 'done' and state['exit_code'] == 0,
            'five accepted completed processes required')
    guard = read(folder / 'guard_result.json')
    unit = folder / 'unit'
    command = read(unit / 'physical-device-command.json')
    process = read(unit / 'probe-process.json')
    identity = read(unit / 'physical-device-identity.json')
    meta = read(unit / 'probe/metadata.json')
    expected = [a.replace('{attempt_dir}', str(folder)) for a in job['command']]
    require(expected.count('--') == 1, 'ambiguous physical command separator')
    require(command['command'] == expected[expected.index('--') + 1:]
            and command['actual_returncode'] == 0, 'actual child differs from queued command')
    require(command['command'][1:] == [str(paths['runner']), '--fresh', str(fresh),
            '--out', str(unit / 'probe')], 'wrong probe runner/fresh/output command')
    policy_path = Path(option(expected, '--guard-policy'))
    wrapper_path = Path(expected[3])
    require(wrapper_path.name == 'run_r16_with_device_identity.py'
            and expected[:2] == ['numactl', '--cpunodebind=' + str(identity['numa_node'])],
            'wrong physical wrapper/NUMA launch')
    policy = read(policy_path)
    gpu = identity['gpu_index']
    require(gpu in (0, 1, 2, 3, 4, 7) and gpu in policy['allowed_timing_gpu_indices']
            and job['env']['CUDA_VISIBLE_DEVICES'] == str(gpu)
            and job['env']['TILEMEGA_DEVICE_INDEX'] == str(gpu)
            and job['env']['TILEMEGA_NUMA_NODE'] == str(identity['numa_node']),
            'wrong declared physical GPU/NUMA/timing domain')
    require(identity['nvml_uuid'] == option(expected, '--expected-uuid')
            and identity['cuda_uuid'] == identity['nvml_uuid'].removeprefix('GPU-')
            and identity['cuda_pci_bus_id'].lower() == option(expected, '--expected-pci').lower()
            and identity['numa_node'] == int(option(expected, '--numa-node'))
            and identity['sm_count'] == 132 and identity['probe_exit_code'] == 0
            and identity['outer_guard_holds_percard_lock'] is True
            and identity['cuda_driver_queries_without_context'] is True
            and identity['wrapper_sha256'] == sha(wrapper_path)
            and identity['guard_policy_sha256'] == sha(policy_path), 'physical device binding failed')
    require(guard['code'] == 0 and guard['reason'] == 'child exit 0'
            and isinstance(guard['ownership_verified'], bool), 'guard rejected process')
    if not guard['ownership_verified']:
        require(guard['acceptance'] == 'provisional_container_pid_namespace'
                and job['env']['TILEMEGA_GPU_PID_MODE'] == 'namespace_limited',
                'unrecognized provisional guard')
    require(process['fresh'] == fresh == meta['fresh']
            and process['ppid'] == identity['wrapper_pid']
            and process['boot_id'] == identity['boot_id']
            and process['pid'] != process['ppid'] and process['pid'] > 0
            and int((folder / 'pgid').read_text()) == identity['wrapper_pid']
            and state['started'] <= identity['created'] <= process['created'] <= state['ended'],
            'fresh process/parent/boot/time binding failed')
    require(process['binary_sha256'] == sha(paths['binary']) == job['artifact_sha256']
            and job['source_sha256'] == sha(paths['source'])
            and job['performance_eligible'] is False, 'binary/source binding failed')
    require(meta['numerical_failures'] == 0 and math.isfinite(meta['frequency_drift_pct'])
            and 0 <= meta['frequency_drift_pct'] <= 2, 'numerical/frequency bracket rejected')
    if metadata_validator is not None:
        metadata_validator(meta)
    else:
        require(tuple(meta[k] for k in ('grid', 'active_ctas_max', 'threads', 'shared_bytes',
                'registers', 'local_bytes', 'observed_residency')) == (132, 128, 160, 199808, 142, 0, 1)
                and meta['observation'] == 'actual_Consume_entry_to_return_with_static_operands_and_dummy_producer'
                and meta['includes'] == ['MMA', 'full_empty_protocol', 'finalize', 'epilogue', 'generation_publication']
                and meta['excludes'] == ['memory_transfer', 'grid_barrier']
                and meta['performance_eligible'] is False, 'wrong consumer geometry/observation')
    history = [json.loads(line) for line in (folder / 'guard.jsonl').read_text().splitlines()]
    pre = [v for v in history if v['phase'] == 'preflight']
    require(len(pre) == policy['samples'] and {v['sample'] for v in pre} == set(range(policy['samples']))
            and history[-1]['phase'] == 'after' and history[-1]['exit_code'] == 0
            and all(v['gpu_index'] == gpu for v in history), 'guard history incomplete/wrong device')
    files = [folder / 'guard_result.json', folder / 'guard.jsonl', folder / 'pgid',
             unit / 'physical-device-command.json', unit / 'physical-device-identity.json',
             unit / 'probe-process.json', unit / 'probe/metadata.json', unit / 'probe/raw.csv']
    if metadata_validator is not None:
        files.append(unit / 'probe/ctas.csv')
    return dict(fresh=fresh, pid=process['pid'], ppid=process['ppid'], boot_id=process['boot_id'],
                physical_identity=identity, metadata=meta, guard=guard, attempt=str(folder),
                binary_sha256=process['binary_sha256'],
                receipts={str(p): sha(p) for p in files},
                wrapper_sha256=sha(wrapper_path), guard_policy_sha256=sha(policy_path))


def metrics_for_process(observations, fresh):
    samples = defaultdict(list)
    paired = []
    trials = []
    for (active, observe, trial, cta), r in sorted(observations.items()):
        key = (active, str(observe))
        samples[key + ('cta', 'connected_entry_to_return_envelope_ns')].append(r['connected_cta_envelope_ns'])
        for warp in range(4):
            samples[key + ('warp', 'entry_to_return_ns')].append(r[f'returned_w{warp}'] - r[f'entry_w{warp}'])
        if observe:
            for slot in (0, 1):
                samples[key + ('cta', f'empty{slot}_observer_wait_ns')].append(
                    r[f'empty{slot}_observed_ready'] - r[f'empty{slot}_wait_begin'])
                samples[key + ('cta', f'empty{slot}_release_upper_from_entry_ns')].append(
                    r[f'empty{slot}_release_upper'] - r['entry_min'])
                samples[key + ('cta', f'empty{slot}_release_bound_width_ns')].append(
                    r[f'empty{slot}_release_upper'] - r[f'empty{slot}_release_lower'])
            off = observations[active, 0, trial, cta]
            pair = dict(fresh=fresh, active_ctas=active, trial=trial, cta=cta)
            pair.update({'off_' + k: v for k, v in off.items() if k not in ('fresh', 'active_ctas', 'trial', 'cta', 'observe_empty')})
            pair.update({'on_' + k: v for k, v in r.items() if k not in ('fresh', 'active_ctas', 'trial', 'cta', 'observe_empty')})
            pair['on_over_off_envelope_ratio'] = r['connected_cta_envelope_ns'] / off['connected_cta_envelope_ns']
            paired.append(pair)
            samples[(active, 'paired_control', 'cta', 'on_over_off_envelope_ratio')].append(pair['on_over_off_envelope_ratio'])
    for active in ACTIVE:
        for trial in range(1, 17):
            pair = dict(fresh=fresh, active_ctas=active, trial=trial)
            for observe in (0, 1):
                population = [observations[active, observe, trial, c] for c in range(active)]
                begin = min(r['entry_min'] for r in population)
                end = max(r['returned_max'] for r in population)
                envelope = end - begin
                samples[(active, str(observe), 'trial', 'population_entry_to_return_envelope_ns')].append(envelope)
                for name, value in [('entry_min', begin), ('returned_max', end), ('envelope_ns', envelope)]:
                    pair[('on_' if observe else 'off_') + name] = value
            pair['on_over_off_envelope_ratio'] = pair['on_envelope_ns'] / pair['off_envelope_ns']
            trials.append(pair)
            samples[(active, 'paired_control', 'trial', 'on_over_off_population_envelope_ratio')].append(pair['on_over_off_envelope_ratio'])
    rows = [dict(fresh=fresh, active_ctas=key[0], observe_empty=key[1], scope=key[2],
                 metric=key[3], **summarize(values)) for key, values in sorted(samples.items())]
    return rows, paired, trials


def aggregate_processes(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ('active_ctas', 'observe_empty', 'scope', 'metric'))
        grouped[key].append(row)
    result = []
    for key, values in sorted(grouped.items()):
        require(len(values) == 5 and {r['fresh'] for r in values} == set(range(5)),
                'each summary needs five distinct process statistics')
        result.append(dict(active_ctas=key[0], observe_empty=key[1], scope=key[2], metric=key[3],
            processes=5, median_of_process_medians=median(r['median'] for r in values),
            median_of_process_p99=median(r['p99'] for r in values),
            min_process_median=min(r['median'] for r in values),
            max_process_median=max(r['median'] for r in values),
            aggregation='process_statistics_not_pooled'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'preserve prior collection')
    state_bytes, binding_bytes, queue_bytes = (p.read_bytes() for p in (args.state, args.bindings, args.queue))
    state, binding, queue = (json.loads(v) for v in (state_bytes, binding_bytes, queue_bytes))
    for path, digest in binding['files'].items():
        require(sha(path) == digest, 'frozen probe input changed: ' + path)
    paths = {key: next(Path(p) for p in binding['files'] if Path(p).name == name)
             for key, name in [('binary', 'probe'), ('runner', 'run.py'),
                               ('source', 'sm90r2_flow_consumer_probe.cu'), ('review', 'function-review.json')]}
    review = read(paths['review'])
    require(review['production_function_byte_equal'] is True
            and review['production_resource_metadata_equal'] is True
            and review['probe_binary_sha256'] == sha(paths['binary'])
            and review['comparison']['unique'] is True
            and review['comparison']['role'] == 'Consume'
            and review['comparison']['left_flags'] == review['comparison']['right_flags'] == [['true', 'false']],
            'missing bound exact outlined Consume comparison')
    jobs = {j['name']: j for j in queue}
    names = [f'flow-consumer-v1-f{i}' for i in range(5)]
    require(set(jobs) == set(names) and len(queue) == 5, 'wrong five-process queue')
    selected = {name: state[name] for name in names}
    evidence, process_rows, pairs, trials = [], [], [], []
    for fresh, name in enumerate(names):
        folder = Path(selected[name]['out'])
        receipt = validate_process(folder, fresh, selected[name], jobs[name], binding, paths)
        observations = rows_from_raw(folder / 'unit/probe/raw.csv', fresh)
        rows, paired, trial = metrics_for_process(observations, fresh)
        evidence.append(receipt)
        process_rows.extend(rows)
        pairs.extend(paired)
        trials.extend(trial)
    require(len({(v['boot_id'], v['pid']) for v in evidence}) == 5
            and len({v['ppid'] for v in evidence}) == 5, 'not five fresh child/wrapper processes')
    require(len({tuple(v['physical_identity'][k] for k in ('gpu_index', 'cuda_uuid', 'cuda_pci_bus_id', 'numa_node'))
                 for v in evidence}) == 1, 'mixed physical device cohort')
    require({name: read(args.state)[name] for name in names} == selected, 'completed selected scheduler rows changed')
    for path, digest in binding['files'].items():
        require(sha(path) == digest, 'probe input changed during collection: ' + path)
    for receipt in evidence:
        for path, digest in receipt['receipts'].items():
            require(sha(path) == digest, 'process evidence changed during collection: ' + path)
    args.out.mkdir(parents=True)
    (args.out / 'inputs').mkdir()
    for name, data in [('state.json', state_bytes), ('bindings.json', binding_bytes), ('queue.json', queue_bytes)]:
        (args.out / 'inputs' / name).write_bytes(data)
    (args.out / 'raw').mkdir()
    for receipt in evidence:
        folder = Path(receipt['attempt'])
        shutil.copyfile(folder / 'unit/probe/raw.csv', args.out / 'raw' / f"f{receipt['fresh']}.csv")
    csv_write(args.out / 'process-statistics.csv', process_rows)
    csv_write(args.out / 'metrics.csv', aggregate_processes(process_rows))
    csv_write(args.out / 'paired-cta-observations.csv', pairs)
    csv_write(args.out / 'paired-trial-observations.csv', trials)
    write(args.out / 'receipt.json', dict(schema=1,
        evidence='verified connected coupled-response diagnostic observations only',
        bindings_sha256=hashlib.sha256(binding_bytes).hexdigest(),
        state_sha256_at_read=hashlib.sha256(state_bytes).hexdigest(),
        queue_sha256=hashlib.sha256(queue_bytes).hexdigest(), selected_scheduler_rows=selected,
        processes=evidence, numerical_passes=5, numerical_processes=5,
        ownership_verified=all(v['guard']['ownership_verified'] for v in evidence),
        guard_scope='preserved original guarded acceptance; provisional remains provisional',
        function_comparison=review, function_comparison_scope='exact one outlined Consume only',
        source_scope='bound CU/head/runner/build receipt; no complete header snapshot is present in original bindings',
        observation='Consume entry-to-return response coupled to concurrent dummy producer',
        empty_bounds='all-warp entry max <= release <= min(all-warp return min, observer ready); wait-begin is not a release lower bound',
        control_pairing='same process/active/trial/CTA; separate launches, all off trials precede all on trials',
        quantile='sorted[min(n-1,floor(0.99*n))], matching transfer collector',
        aggregation='per-process median/p99 first, then median across five processes; not pooled',
        pure_compute_observed=False, observer_cost_subtracted=False,
        physical_capacity_or_latency_fit='not_performed', production_kernel_equivalent=False,
        performance_acceptance=False))
    print(json.dumps(dict(processes=5, cta_pairs=len(pairs), trial_pairs=len(trials),
                          process_statistic_rows=len(process_rows), out=str(args.out))))


if __name__ == '__main__':
    main()
