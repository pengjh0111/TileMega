#!/usr/bin/env python3
"""Reproduce MX-5's stage-zero decision tables from immutable raw samples."""
import argparse
import csv
import io
import json
from pathlib import Path
import re
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('runs/mx5/probes'))
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--decisions', type=Path, required=True)
    args = parser.parse_args()
    raw = args.runs
    groups = {}
    rows = []
    provenance = json.loads((raw / 'source_provenance.json').read_text())
    head = provenance['base_head']

    def add(values, round_name, implementation, sources):
        identity = provenance['implementations'][implementation]
        if identity['sources'] != sources:
            raise ValueError('measurement source list differs from its recorded identity')
        row = dict(values, round=round_name, executor='mx5_probe', implementation=implementation,
                   base_head=head, source_sha256=identity['source_sha256'], card='7', evidence='verified')
        rows.append(row)

    for family in ['sync', 'load']:
        source = f'experiments/mx5/{family}_probe.cu'
        family_groups = {}
        for round_number in range(1, 6):
            path = raw / 'five_process' / f'{family}_round{round_number}.csv'
            samples = list(csv.DictReader(path.open()))
            if len(samples) != (51 if family == 'sync' else 45):
                raise ValueError('incomplete sample: ' + str(path))
            for row in samples:
                key_fields = ['probe', 'variant', 'n_kib'] if family == 'sync' else ['shape', 'active_ctas', 'outstanding_kib']
                key = tuple(row[k] for k in key_fields)
                family_groups.setdefault(key, []).append(row)
                add(row, f'round{round_number}', family + '_probe', [source])
        for key, values in family_groups.items():
            if len(values) != 5:
                raise ValueError(f'{family} {key}: expected five processes')
            row = values[0].copy()
            fields = ['operation_ns', 'load_ready_ns', 'post_operation_ns'] if family == 'sync' else ['gbps', 'per_cta_gbps', 'latency_ns']
            for field in fields:
                row[field] = statistics.median(float(v[field]) for v in values)
            add(row, 'median', family + '_probe', [source])
            groups[(family,) + key] = row
    guard = json.loads((raw / 'five_process/guard/guard_result.json').read_text())
    if guard['code']:
        raise ValueError('probe guard rejected the samples')

    def sync(probe, variant, n, field='operation_ns'):
        return float(groups[('sync', probe, variant, str(n))][field])

    load8 = sync('L8', 'no_operation', 8, 'load_ready_ns')
    load0 = sync('L8', 'no_operation', 0, 'load_ready_ns')
    threshold = .1 * (load8 - load0)
    cta_variants = [key[2] for key in groups if key[:2] == ('sync', 'S1') and key[3] == '0']
    cta_deltas = {v: sync('S1', v, 8) - sync('S1', v, 0) for v in cta_variants}
    poll_variants = ['volatile', 'atomic_add_zero', 'atomic_load_relaxed']
    release_variants = ['fence_atomic_add', 'atomic_add_release']
    ratios = {f'{p}/{v}': sync(p, v, 8) / sync(p, v, 0)
              for p, variants in [('S3', poll_variants), ('S4', release_variants)] for v in variants}
    threads = 576 if max(ratios.values()) >= 1.3 else 512
    release = 'release_atomic' if ratios['S4/atomic_add_release'] <= 1.1 else 'fence_atomic_add'
    lds_delta = sync('S2', 'lds_release', 8) - sync('S2', 'lds_release', 0)
    prefetch = 'P-A' if min(cta_deltas.values()) <= threshold else 'P-B' if lds_delta <= threshold else 'P-C'
    exchange = json.loads((raw / 'build/full_s1/exchange_wait_check.json').read_text())
    exchange_delta = sync('S2b', 'syncwarp_exchange', 8) - sync('S2b', 'syncwarp_exchange', 0)
    wave_exchange = 'X-A' if exchange_delta <= threshold and not exchange['global_waits_in_exchange'] else 'X-B'
    fragment_ratios = {}
    for shape in ['K', 'Vt']:
        for active in [26, 52, 104]:
            for outstanding in [8, 16, 32, 64, 128]:
                num = float(groups[('load', shape, str(active), str(outstanding))]['gbps'])
                den = float(groups[('load', 'W', str(active), str(outstanding))]['gbps'])
                fragment_ratios[f'{shape}/{active}/{outstanding}'] = num / den
    kv = 'rows_vt' if min(fragment_ratios.values()) >= .9 else 'block_fragments'
    poll_n = 0 if threads == 576 else 8
    poll = min(poll_variants, key=lambda v: sync('S3', v, poll_n))

    sources = ['tools/probes/maca/sync_litmus.cu',
               'include/tilemega/Codegen/executor/LastArriverV2.cuh',
               'include/tilemega/Codegen/tasks/mx/StageSyncV2.cuh']
    litmus = {}
    for mode in ['barrier_v2', 'la_v2', 'lds_release']:
        for round_number in range(1, 6):
            text = (raw / 'litmus/five_process' / f'{mode}_round{round_number}.log').read_text()
            match = re.search(r'iterations=(\d+).*errors=(\d+) lasts=(\d+) us/iter=([\d.]+) PASS', text)
            if not match or int(match[1]) < 2000 or int(match[2]) != 0 or 'ctas=104 threads=576' not in text:
                raise ValueError(f'litmus rejected: {mode} process {round_number}')
            if mode == 'la_v2' and match[1] != match[3]:
                raise ValueError('last-arriver count differs')
            add(dict(probe='S5', variant=mode, errors=match[2], iterations=match[1],
                     median_ns=float(match[4]) * 1000, ctas=104, threads=576),
                f'round{round_number}', 'sync_litmus_v2', sources)
        litmus[mode] = dict(processes=5, iterations_per_process=2000, ctas=104, errors=0)
    if json.loads((raw / 'litmus/five_process/guard/guard_result.json').read_text())['code']:
        raise ValueError('litmus guard rejected the samples')

    sizes = json.loads((raw / 'i1/code_sizes.json').read_text())
    instruction_groups = {}
    for round_number in range(1, 6):
        path = raw / 'i1/five_process' / f'round{round_number}.csv'
        sample = list(csv.DictReader(path.open()))
        if len(sample) != 10:
            raise ValueError('incomplete I1 sample')
        for row in sample:
            row['code_bytes'] = sizes[row['class']]['bytes']
            row['program_code_bytes'] = sizes['program']['bytes']
            add(row, f'round{round_number}', 'mx4_instruction_order', ['experiments/mx4/stage_bench.cu'])
            instruction_groups.setdefault((row['class'], row['mode']), []).append(row)
    for values in instruction_groups.values():
        if len(values) != 5:
            raise ValueError('I1 requires five processes per key')
        row = values[0].copy()
        for field in ['median_ns', 'p90_ns']:
            row[field] = statistics.median(float(v[field]) for v in values)
        add(row, 'median', 'mx4_instruction_order', ['experiments/mx4/stage_bench.cu'])
    if json.loads((raw / 'i1/five_process/guard/guard_result.json').read_text())['code']:
        raise ValueError('I1 guard rejected the samples')
    builtin = json.loads((raw / 's0/results.json').read_text())
    for row in builtin['probes']:
        add(dict(probe='S0', variant=row['builtin'], compiled=int(row['returncode'] == 0)),
            'compile', 'builtin_probe', ['experiments/mx5/builtin_probe.py'])
        rows[-1]['card'] = 'cpu'
    if any(row['returncode'] for row in builtin['probes']):
        raise ValueError('a synchronization builtin did not compile')

    decisions = dict(threads=threads, release=release, prefetch=prefetch, wave_exchange=wave_exchange,
                     kv_layout=kv, poll=poll, load8_ns=load8, load0_ns=load0, threshold_ns=threshold,
                     cta_delta_ns=cta_deltas, lds_delta_ns=lds_delta, exchange_delta_ns=exchange_delta,
                     load_ratios=ratios, fragment_ratios=fragment_ratios, litmus=litmus,
                     builtin_count=builtin['builtin_count'], base_head=head)
    fields = sorted({k for row in rows for k in row})
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    for path, body in [(args.csv, stream.getvalue()), (args.decisions, json.dumps(decisions, indent=2) + '\n')]:
        if path.exists() and path.read_text() != body:
            raise ValueError('refusing to replace prior results: ' + str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists(): path.write_text(body)
    print(json.dumps(decisions, indent=2))


if __name__ == '__main__':
    main()
