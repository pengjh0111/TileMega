#!/usr/bin/env python3
"""Validate/collect paired consumer responses with explicit prefed readiness."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import median
from collect_sm90r2_flow_consumers import (sha, read, write, require, summarize, csv_write,
                        validate_process)

ACTIVE = (1, 32, 64, 128)
FEEDS = ('concurrent', 'prefed')
PROBE_SHA = 'cb900b2555c4f557c27a9772e39557320caecb6f679846eb592a1a5884359c0f'
COORD = ('active_ctas', 'feed_mode', 'observe_empty', 'trial', 'cta')
CTA_FIELDS = ('fresh', 'active_ctas', 'feed_mode', 'observe_empty', 'trial', 'cta',
              'smid', 'has_work', 'full_ready_mask', 'full0_confirmed',
              'full1_confirmed', 'setup_acquired', 'v3_arrive', 'v3_acquired')
RAW_FIELDS = ('fresh', 'active_ctas', 'feed_mode', 'observe_empty', 'trial', 'cta',
              'smid', 'warp', 'entry', 'returned', 'full_ready_mask',
              'full0_confirmed', 'full1_confirmed', 'setup_acquired', 'v3_arrive',
              'v3_acquired', 'empty0_wait_begin', 'empty0_observed_ready',
              'empty1_wait_begin', 'empty1_observed_ready')
OBSERVER = RAW_FIELDS[-4:]
COMMON = ('fresh',) + COORD + ('smid', 'full_ready_mask', 'full0_confirmed',
                              'full1_confirmed', 'setup_acquired', 'v3_arrive', 'v3_acquired')


def launch_order():
    return [(a, FEEDS[fi ^ (t & 1)], oi ^ ((t >> 1) & 1), t)
            for a in ACTIVE for t in range(1, 17)
            for fi in range(2) for oi in range(2)]


def parse(row, fields, fresh):
    require(set(row) == set(fields), 'wrong CSV columns')
    require(row['feed_mode'] in FEEDS, 'invalid feed mode')
    require(all(isinstance(v, str) and v.isascii() and v.isdecimal()
                for k, v in row.items() if k != 'feed_mode'), 'non-unsigned CSV value')
    r = {k: v if k == 'feed_mode' else int(v) for k, v in row.items()}
    require(r['fresh'] == fresh and r['active_ctas'] in ACTIVE and
            r['observe_empty'] in (0, 1) and 1 <= r['trial'] <= 16 and
            0 <= r['cta'] < 132 and all(0 <= v < 2**64 for k, v in r.items()
                                      if k != 'feed_mode'), 'invalid process/coordinates')
    require(0 < r['setup_acquired'] <= r['v3_arrive'] <= r['v3_acquired'],
            'invalid native V3 timestamp order')
    if r['feed_mode'] == 'prefed' and r['cta'] < r['active_ctas']:
        require(r['full_ready_mask'] == 3 and
                0 < r['full0_confirmed'] <= r['full1_confirmed'] <= r['setup_acquired'],
                'prefed full-ready confirmation failed')
    else:
        require((r['full_ready_mask'], r['full0_confirmed'], r['full1_confirmed']) == (0, 0, 0),
                'unexpected prefeed confirmation')
    if fields == CTA_FIELDS:
        require(r['has_work'] == int(r['cta'] < r['active_ctas']), 'wrong CTA work role')
    else:
        require(r['cta'] < r['active_ctas'] and 0 <= r['warp'] < 4 and
                r['setup_acquired'] <= r['entry'] < r['returned'] <= r['v3_arrive'],
                'invalid Consume timestamp/coordinates')
        stamps = tuple(r[k] for k in OBSERVER)
        if r['observe_empty']:
            require(0 < stamps[0] <= stamps[1] <= stamps[2] <= stamps[3] <= r['v3_arrive'],
                    'invalid empty observer order')
        else:
            require(stamps == (0, 0, 0, 0), 'off control has observer stamps')
    return r


def load_csv(path, fields, fresh):
    with Path(path).open() as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == list(fields), 'wrong CSV header: ' + str(path))
        return [parse(row, fields, fresh) for row in reader]


def validate_rows(raw_rows, cta_rows):
    expected_launches = launch_order()
    expected_ctas = [launch + (c,) for launch in expected_launches for c in range(132)]
    require([tuple(r[k] for k in COORD) for r in cta_rows] == expected_ctas,
            'incomplete/duplicate/reordered CTA cohort or interleaved context toggles')
    expected_raw = [launch + (c, w) for launch in expected_launches
                    for c in range(launch[0]) for w in range(4)]
    require([tuple(r[k] for k in COORD + ('warp',)) for r in raw_rows] == expected_raw,
            'incomplete/duplicate/reordered raw cohort or interleaved context toggles')
    ctas = {tuple(r[k] for k in COORD): r for r in cta_rows}
    previous_end = 0
    for launch in expected_launches:
        group = [ctas[launch + (c,)] for c in range(132)]
        require(len({r['smid'] for r in group}) == 132,
                '132 resident CTAs lack distinct actual SMIDs')
        require(max(r['v3_arrive'] for r in group) <= min(r['v3_acquired'] for r in group),
                'V3 acquire predates a CTA release arrival')
        require(previous_end <= min(r['setup_acquired'] for r in group),
                'interleaved launches overlap or have reversed global time')
        previous_end = max(r['v3_acquired'] for r in group)
        if launch[1] == 'prefed':
            require(max(r['full1_confirmed'] for r in group) <= min(r['setup_acquired'] for r in group),
                    'setup V3 acquired before every prefed full confirmation')
    groups = defaultdict(dict)
    for row in raw_rows:
        key = tuple(row[k] for k in COORD)
        require(all(row[k] == ctas[key][k] for k in COMMON), 'raw/CTA duplicate metadata differs')
        groups[key][row['warp']] = row
    observations = {}
    for key, warps in groups.items():
        require(set(warps) == set(range(4)), 'missing warp')
        require(len({tuple(r[k] for k in OBSERVER) for r in warps.values()}) == 1,
                'observer copies disagree across warps')
        r = dict(ctas[key]);r.update({k: warps[0][k] for k in OBSERVER})
        r.update(entry_min=min(v['entry'] for v in warps.values()),
                 entry_max=max(v['entry'] for v in warps.values()),
                 returned_min=min(v['returned'] for v in warps.values()),
                 returned_max=max(v['returned'] for v in warps.values()))
        r['connected_cta_envelope_ns'] = r['returned_max'] - r['entry_min']
        r['warp_durations'] = [warps[w]['returned'] - warps[w]['entry'] for w in range(4)]
        if r['observe_empty']:
            for slot in (0, 1):
                r[f'empty{slot}_release_lower'] = r['entry_max']
                r[f'empty{slot}_release_upper'] = min(r['returned_min'], r[f'empty{slot}_observed_ready'])
                require(r[f'empty{slot}_release_lower'] <= r[f'empty{slot}_release_upper'],
                        'impossible empty-release causal bounds')
        observations[key] = r
    return observations, ctas


def validate_metadata(meta):
    require(tuple(meta.get(k) for k in ('schema', 'grid', 'contexts', 'threads', 'shared_bytes',
            'record_offset', 'slots', 'iterations', 'registers', 'local_bytes', 'observed_residency',
            'measured_trials')) == (2, 132, 16, 160, 199808, 448, 2, 2, 142, 0, 1, 16),
            'wrong v2 geometry/resources')
    require(meta.get('active_ctas') == list(ACTIVE) and meta.get('feed_modes') == list(FEEDS)
            and meta.get('observer_modes') == [False, True]
            and meta.get('observation') == 'actual_Consume_entry_to_return_with_static_operands_and_dummy_producer'
            and meta.get('includes') == ['MMA', 'full_empty_protocol', 'finalize', 'epilogue', 'generation_publication']
            and meta.get('excludes') == ['memory_transfer', 'grid_barriers_outside_timed_Consume']
            and meta.get('prefed_is_pure_compute') is False
            and meta.get('performance_eligible') is False
            and meta.get('prefeed_confirmation') == 'both_full_phases_waited_and_ready_mask_3_before_setup_V3_and_timed_Consume_entry'
            and meta.get('idle_CTAs') == 'resident_at_native_V3_release_acquire_until_all_active_CTAs_finish',
            'wrong v2 observation/control declaration')


def source_binding(paths, binding):
    snapshot = read(paths['snapshot']);head = snapshot['source_head']
    require(head == binding['source_head'] and snapshot['repo_diff'] == '',
            'source snapshot not a clean bound head')
    source = paths['source'].read_bytes()
    headers = snapshot['header_hashes']
    require(all(k in headers for k in ('include/tilemega/Backend/Sm90DenseGemm.h',
            'include/tilemega/Backend/Sm90DenseRing.cuh', 'include/tilemega/Backend/Sm90DenseOperands.cuh',
            'include/tilemega/Codegen/tasks/Sm90DenseProgram.cuh')), 'missing critical source hashes')
    repo = paths['source'].parent.parent
    names = list(headers) + [str(paths['source'].relative_to(repo))]
    command = ['git', '-C', str(repo), 'cat-file', '--batch']
    data = subprocess.check_output(command, input=''.join(head + ':' + p + '\n' for p in names).encode())
    cursor = 0
    for name in names:
        end = data.index(b'\n', cursor);info = data[cursor:end].split();cursor = end + 1
        require(len(info) == 3 and info[1] == b'blob', 'missing bound source object: ' + name)
        size = int(info[2]);blob = data[cursor:cursor + size];cursor += size + 1
        expected = headers.get(name, hashlib.sha256(source).hexdigest())
        require(hashlib.sha256(blob).hexdigest() == expected, 'snapshot/head source mismatch: ' + name)
    text = source.decode()
    require('if(event.arrivals!=Grid)' in text and 'V3 arrival count mismatch' in text and
            'for(auto const& event:event_host)' in text,
            'missing executed host counter check')
    build = read(paths['build']);cmd = build['command']
    require(build['status'] == 'completed' and build['exit_code'] == 0 and
            cmd[cmd.index('-o') + 1] == str(paths['binary'].relative_to(repo)) and
            str(paths['source'].relative_to(repo)) in cmd and '-rdc=true' in cmd and
            'arch=compute_90a,code=sm_90a' in cmd, 'wrong bound build receipt')
    return dict(source_head=head, own_headers_verified=len(headers),
                counter_evidence='source-bound per-launch host check of both counters==132 plus guarded child exit0; counter values not separately serialized',
                external_header_scope='third-party includes not fully snapshotted')


def process_metrics(observations, ctas, fresh):
    samples = defaultdict(list);pairs = [];trials = [];placements = [];by_sm = defaultdict(list)
    def add(a, feed, observe, scope, metric, value):
        samples[(a, feed, str(observe), scope, metric)].append(value)
    for (a, feed, observe, t, c), row in ctas.items():
        role = 'active' if row['has_work'] else 'idle'
        span = row['v3_acquired'] - row['v3_arrive']
        add(a, feed, observe, role + '_cta', 'native_v3_arrive_to_acquire_span_ns', span)
        place = dict(row, native_v3_arrive_to_acquire_span_ns=span)
        obs = observations.get((a, feed, observe, t, c))
        place['connected_cta_envelope_ns'] = obs['connected_cta_envelope_ns'] if obs else ''
        placements.append(place)
        by_sm[(a, feed, observe, row['smid'], role, 'native_v3_arrive_to_acquire_span_ns')].append(span)
        if obs:
            by_sm[(a, feed, observe, row['smid'], role, 'connected_cta_envelope_ns')].append(obs['connected_cta_envelope_ns'])
    for (a, feed, observe, t, c), row in observations.items():
        add(a, feed, observe, 'cta', 'connected_entry_to_return_envelope_ns', row['connected_cta_envelope_ns'])
        add(a, feed, observe, 'cta', 'entry_skew_ns', row['entry_max'] - row['entry_min'])
        add(a, feed, observe, 'cta', 'return_skew_ns', row['returned_max'] - row['returned_min'])
        for value in row['warp_durations']:
            add(a, feed, observe, 'warp', 'paired_entry_to_return_ns', value)
        if observe:
            for slot in (0, 1):
                add(a, feed, observe, 'cta', f'empty{slot}_observer_wait_ns', row[f'empty{slot}_observed_ready'] - row[f'empty{slot}_wait_begin'])
                add(a, feed, observe, 'cta', f'empty{slot}_release_upper_from_entry_ns', row[f'empty{slot}_release_upper'] - row['entry_min'])
            off = observations[a, feed, 0, t, c]
            pair = dict(fresh=fresh, active_ctas=a, feed_mode=feed, trial=t, cta=c,
                        off_smid=off['smid'], on_smid=row['smid'], same_smid=off['smid'] == row['smid'],
                        off_envelope_ns=off['connected_cta_envelope_ns'], on_envelope_ns=row['connected_cta_envelope_ns'])
            pair['on_over_off_envelope_ratio'] = pair['on_envelope_ns'] / pair['off_envelope_ns'];pairs.append(pair)
            add(a, feed, 'paired_control', 'cta', 'on_over_off_envelope_ratio', pair['on_over_off_envelope_ratio'])
    for a in ACTIVE:
        for t in range(1, 17):
            env = {}
            for feed in FEEDS:
                for observe in (0, 1):
                    pop = [observations[a, feed, observe, t, c] for c in range(a)]
                    begin = min(r['entry_min'] for r in pop);end = max(r['returned_max'] for r in pop)
                    env[feed, observe] = end - begin
                    add(a, feed, observe, 'trial', 'population_entry_to_return_envelope_ns', end - begin)
                ratio = env[feed, 1] / env[feed, 0]
                trials.append(dict(fresh=fresh, active_ctas=a, trial=t, comparison='observer_on_over_off',
                                   fixed_context=feed, numerator_envelope_ns=env[feed, 1], denominator_envelope_ns=env[feed, 0], ratio=ratio))
                add(a, feed, 'paired_control', 'trial', 'on_over_off_population_envelope_ratio', ratio)
            for observe in (0, 1):
                ratio = env['prefed', observe] / env['concurrent', observe]
                trials.append(dict(fresh=fresh, active_ctas=a, trial=t, comparison='prefed_over_concurrent',
                                   fixed_context=str(observe), numerator_envelope_ns=env['prefed', observe], denominator_envelope_ns=env['concurrent', observe], ratio=ratio))
                add(a, 'paired_feed', observe, 'trial', 'prefed_over_concurrent_population_envelope_ratio', ratio)
    stats = [dict(fresh=fresh, active_ctas=k[0], feed_mode=k[1], observe_empty=k[2], scope=k[3],
                  metric=k[4], **summarize(v)) for k, v in sorted(samples.items())]
    smstats = [dict(fresh=fresh, active_ctas=k[0], feed_mode=k[1], observe_empty=k[2], smid=k[3], role=k[4],
                    metric=k[5], **summarize(v)) for k, v in sorted(by_sm.items())]
    return stats, pairs, trials, placements, smstats


def aggregate(rows, freshes):
    groups = defaultdict(list)
    names = ('active_ctas', 'feed_mode', 'observe_empty', 'scope', 'metric')
    for row in rows:groups[tuple(row[k] for k in names)].append(row)
    result = []
    for key, values in sorted(groups.items()):
        require(len(values) == len(freshes) and {r['fresh'] for r in values} == set(freshes),
                'mixed/missing process statistic')
        result.append(dict(zip(names, key), processes=len(values),
            median_of_process_medians=median(r['median'] for r in values),
            median_of_process_p99=median(r['p99'] for r in values),
            min_process_median=min(r['median'] for r in values),
            max_process_median=max(r['median'] for r in values), aggregation='process_statistics_not_pooled'))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('state', 'bindings', 'queue', 'out'):p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--allow-partial', action='store_true');args = p.parse_args()
    require(not args.out.exists(), 'preserve previous output')
    collector_paths = [Path(__file__).resolve(), Path(__file__).resolve().with_name('collect_sm90r2_flow_consumers.py')]
    collector_hashes = {str(path): sha(path) for path in collector_paths}
    inputs = {name: getattr(args, name).read_bytes() for name in ('state', 'bindings', 'queue')}
    state, binding, queue = (json.loads(inputs[name]) for name in ('state', 'bindings', 'queue'))
    jobs = {j['name']: j for j in queue};names = [f'flow-consumer-prefed-v2-f{i}' for i in range(5)]
    require(len(queue) == 5 and set(jobs) == set(names) and set(state) == set(names), 'wrong five-process cohort')
    require(all(state[n]['status'] in ('pending', 'running', 'done') for n in names), 'cohort contains failed/rejected state')
    selected = {n: state[n] for n in names if state[n]['status'] == 'done'}
    require(selected and (args.allow_partial or len(selected) == 5), 'five completed processes required; use --allow-partial for a provisional subset')
    for path, digest in binding['files'].items():require(sha(path) == digest, 'frozen input changed: ' + path)
    wanted = dict(binary='probe', runner='run.py', source='sm90r2_flow_consumer_prefed_probe.cu',
                  review='function-review.json', snapshot='source-snapshot.json', build='build-public.json')
    paths = {}
    for key, name in wanted.items():
        candidates = [Path(v) for v in binding['files'] if Path(v).name == name]
        require(len(candidates) == 1, 'ambiguous/missing input: ' + name);paths[key] = candidates[0]
    source = source_binding(paths, binding);review = read(paths['review']);comparison = review['comparison']
    require(review['production_function_byte_equal'] is True and review['production_resource_metadata_equal'] is True
            and review['probe_binary_sha256'] == sha(paths['binary']) == PROBE_SHA and comparison['unique'] is True
            and comparison['role'] == 'Consume' and comparison['bytecode_equal'] is True
            and comparison['resource_metadata_equal'] is True
            and comparison['left_flags'] == comparison['right_flags'] == [['true', 'false']], 'exact Consume audit failed')
    evidence = [];stats = [];pairs = [];trials = [];placements = [];smstats = []
    for fresh, name in enumerate(names):
        if name not in selected:continue
        folder = Path(selected[name]['out'])
        receipt = validate_process(folder, fresh, selected[name], jobs[name], binding, paths, validate_metadata)
        raw = load_csv(folder / 'unit/probe/raw.csv', RAW_FIELDS, fresh)
        cta = load_csv(folder / 'unit/probe/ctas.csv', CTA_FIELDS, fresh)
        observations, ctas = validate_rows(raw, cta)
        values = process_metrics(observations, ctas, fresh)
        for destination, rows in zip((stats, pairs, trials, placements, smstats), values):destination.extend(rows)
        evidence.append(receipt)
    freshes = [e['fresh'] for e in evidence]
    require(len({(e['boot_id'], e['pid']) for e in evidence}) == len(evidence)
            and len({e['ppid'] for e in evidence}) == len(evidence), 'not fresh child/wrapper processes')
    require(len({tuple(e['physical_identity'][k] for k in ('gpu_index', 'cuda_uuid', 'cuda_pci_bus_id', 'numa_node')) for e in evidence}) == 1,
            'mixed physical cohort')
    require({n: read(args.state)[n] for n in selected} == selected, 'completed scheduler rows changed')
    for path, digest in binding['files'].items():require(sha(path) == digest, 'input changed during collection')
    for e in evidence:
        for path, digest in e['receipts'].items():require(sha(path) == digest, 'process evidence changed during collection')
    for path, digest in collector_hashes.items():require(sha(path) == digest, 'collector source changed while running')
    args.out.mkdir(parents=True);(args.out / 'inputs').mkdir();(args.out / 'raw').mkdir()
    for name, data in inputs.items():(args.out / 'inputs' / (name + '.json')).write_bytes(data)
    for e in evidence:
        for filename in ('raw.csv', 'ctas.csv'):
            shutil.copyfile(Path(e['attempt']) / 'unit/probe' / filename,
                            args.out / 'raw' / f"f{e['fresh']}-{filename}")
    for filename, rows in [('process-statistics.csv', stats), ('metrics.csv', aggregate(stats, freshes)),
            ('paired-cta-observer-controls.csv', pairs), ('paired-trial-controls.csv', trials),
            ('sm-placement.csv', placements), ('smid-statistics.csv', smstats)]:csv_write(args.out / filename, rows)
    write(args.out / 'receipt.json', dict(schema=2, collector_files=collector_hashes, complete_cohort=len(evidence) == 5,
        numerical_passes=len(evidence), numerical_processes_expected=5, freshes=freshes,
        pending=[n for n in names if n not in selected], selected_scheduler_rows=selected,
        bindings_sha256=hashlib.sha256(inputs['bindings']).hexdigest(), queue_sha256=hashlib.sha256(inputs['queue']).hexdigest(),
        processes=evidence, source_binding=source, function_comparison=review,
        ownership_verified=all(e['guard']['ownership_verified'] for e in evidence),
        guard_scope='original guarded acceptance preserved; provisional stays provisional',
        scope='connected Consume response with concurrent or explicitly full-ready prefed dummy producer; no real transfers',
        control_pairing='same fresh/active/trial/CTA across separate interleaved launches; actual SMID retained and may differ',
        barrier_scope='arrive-to-acquire spans include release RMW and acquire polling; acquire stamp precedes final CTA join',
        counter_scope=source['counter_evidence'],
        empty_bounds='max sampled warp entry <= empty release <= min sampled warp return and observer-ready; observer can lag',
        aggregation='per-process median/p99 then median across accepted processes; full cohort requires five; no pooling',
        quantile='sorted[min(n-1,floor(0.99*n))]',
        pure_compute_observed=False, observer_cost_subtracted=False, marginal_phase_subtraction=False,
        latency_or_physical_capacity_fit=False, production_kernel_equivalent=False, performance_acceptance=False))
    print(json.dumps(dict(processes=len(evidence), complete=len(evidence) == 5, out=str(args.out),
                          cta_observations=len(placements), process_statistic_rows=len(stats))))


if __name__ == '__main__':main()
