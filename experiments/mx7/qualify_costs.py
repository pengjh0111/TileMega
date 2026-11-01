"""Supplement competitive noisy keys once and qualify a complete TP cost table."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def identity(key):
    return json.dumps(key, sort_keys=True)


def competitive_noise(rows):
    groups = defaultdict(list)
    for row in rows:
        key = row['key']
        group = (key['model'], key['b'], key['stage'], key.get('length'))
        groups[group].append(row)
    result = []
    for group in groups.values():
        best = min(row['median_ns'] for row in group)
        result += [row for row in group if row['spread'] > .02 and row['median_ns'] <= best * 1.02]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--constants', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--card', type=int, choices=(4, 5, 6, 7), default=7)
    args = parser.parse_args()
    qualification = args.out.with_suffix('.qualification.json')
    if args.out.exists() or qualification.exists():
        parser.error('preserve existing qualified costs')
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    request = json.loads(args.request.read_text())
    catalog = json.loads(args.catalog.read_text())
    spec = importlib.util.spec_from_file_location('mx7_aggregate', ROOT / 'experiments/mx7/aggregate_costs.py')
    aggregator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aggregator)
    # Revalidate raw provenance before qualifying or supplementing a catalog.
    validated = aggregator.aggregate(request)
    for field in ('rows', 'source_digest', 'requested_keys', 'resource_rejections'):
        if validated[field] != catalog[field]:
            raise ValueError('catalog differs from its source-bound process records')
    noisy = competitive_noise(catalog['rows'])
    build_rows = json.loads(Path(request['all_keys']['path']).read_text())
    index = {identity(row['key']): row for row in build_rows}
    supplements = []
    environment = dict(os.environ, PYTHONPATH=str(ROOT / 'python'), NEEDS_MIB='12288')
    for number, begin in enumerate(range(0, len(noisy), 64)):
        folder = work / f'noise{number:04d}'
        folder.mkdir()
        keys = [index[identity(row['key'])] for row in noisy[begin:begin + 64]]
        key_path = folder / 'keys.json'
        save(key_path, keys)
        records = []
        for round_number in (4, 5):
            run = folder / f'round{round_number}'
            run.mkdir()
            raw = run / 'raw.jsonl'
            command = list(map(str, [ROOT / 'scripts/mx_gpu_job.sh', args.card, '900', run / 'guard',
                                    sys.executable, request['driver']['path'], '--keys', key_path,
                                    '--out', raw, '--round', round_number]))
            start = time.time()
            with (run / 'stdout.log').open('x') as stream:
                child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                status = child.wait()
            process = run / 'PROCESS.json'
            save(process, dict(argv=command, card=args.card, PID=child.pid, start=start,
                               end=time.time(), returncode=status))
            if status:
                raise RuntimeError('noise supplement failed; retain ' + str(run))
            records.append(dict(round=round_number, raw=ref(raw), process=ref(process),
                                guard=ref(run / 'guard/guard_result.json')))
        supplements.append(dict(keys=ref(key_path), records=records))
    if noisy:
        all_keys = work / 'supplemented_keys.json'
        save(all_keys, [index[identity(row['key'])] for row in noisy])
        supplementary_request = dict(request, all_keys=ref(all_keys), jobs=supplements)
        save(work / 'SUPPLEMENTARY_REQUEST.json', supplementary_request)
        extra = aggregator.aggregate(supplementary_request, rounds=(4, 5))
        if extra['resource_rejections']:
            raise ValueError('noise supplement changed resource qualification')
        extras = {identity(row['key']): row for row in extra['rows']}
        for row in noisy:
            added = extras[identity(row['key'])]
            if added['resources'] != row['resources'] or added['source_digest'] != row['source_digest']:
                raise ValueError('supplement source/resources differ')
            records = row['rounds'] + added['rounds']
            if len({record['PID'] for record in records}) != 5:
                raise ValueError('supplement requires five fresh processes')
            values = [record['cost_ns'] for record in records]
            row.update(rounds=records, processes=5, median_ns=statistics.median(values),
                       p90_ns=statistics.median(record['p90_ns'] for record in records))
            row['spread'] = (max(values) - min(values)) / row['median_ns']
    for row in catalog['rows']:
        row['noisy'] = row['spread'] > .02
    constants = json.loads(args.constants.read_text())
    if set(constants) != {'t_ar', 't_head_x'}:
        raise ValueError('AR and head constants are required')
    from tilemega.tp.selection_tp import stage_lists
    for tp in (2, 4):
        for batch in (1, 2, 4, 8, 16):
            stage_lists(catalog['rows'], f'qwen3_1_7b_tp{tp}', batch, constants)
    catalog.update(tp_constants=constants, constants_evidence=ref(args.constants),
                   noise_supplement=ref(work / 'SUPPLEMENTARY_REQUEST.json') if noisy else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save(args.out, catalog)
    save(qualification, dict(solver_eligible=True, cost_table_sha256=sha(args.out),
         kind='mx7_tp_measured_stage_costs', complete_scopes=10,
         source_digest=catalog['source_digest'], history=[ref(args.catalog)],
         supplement_requested_keys=len(noisy),
         noisy_keys=[dict(key=row['key'], p90_ns=row['p90_ns']) for row in catalog['rows'] if row['noisy']]))
    print('qualified', len(catalog['rows']), 'rows, supplements', len(noisy), flush=True)


if __name__ == '__main__':
    main()
