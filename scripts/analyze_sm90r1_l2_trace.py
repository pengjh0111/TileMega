#!/usr/bin/env python3
"""Stage ledger from actual L2 task dependencies and diagnostic timestamps."""
import argparse
import collections
import csv
import json
from pathlib import Path
import statistics

from tilemega.bench.trace_analysis import analyze_task_space_trace, read_tsv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    identity = json.loads((a.trace / 'trace.json').read_text())
    if identity.get('mode') != 'L2' or not identity.get('diagnostic_only'):
        raise ValueError('expected a separately instrumented L2 forward trace')
    reconstruction, spaces, links = analyze_task_space_trace(a.trace / 'v2')
    runtime = {int(r['stage']): r for r in read_tsv(a.trace / 'v2/runtime_stages.tsv')}
    profiles = collections.defaultdict(list)
    for row in read_tsv(a.trace / 'task_profile.tsv'):
        row = {key: int(value) if key != 'kind' else value for key, value in row.items()}
        duration = row['run_end'] - row['run_begin']
        if duration < 0 or not 0 <= row['epilogue_ns'] <= duration:
            raise ValueError('invalid sampled epilogue interval')
        profiles[row['stage']].append(dict(epilogue_ns=row['epilogue_ns'],
                                          mainloop_plus_dispatch_ns=duration-row['epilogue_ns']))
    critical = collections.defaultdict(lambda: collections.Counter())
    for row in links:
        critical[row['stage']].update({key: row[key] for key in
                                      ('wall_ns', 'task_fixed_plus_mainloop_ns',
                                       'publish_ns', 'wait_hop_ns', 'barrier_ns', 'idle_ns')})
    rows = []
    for row in spaces:
        stage = row['stage']
        samples = profiles[stage]
        rows.append(dict(**row, kind=runtime[stage]['kind'],
                         critical_path_ns=critical[stage]['wall_ns'],
                         critical_wait_ns=critical[stage]['wait_hop_ns'],
                         critical_publish_ns=critical[stage]['publish_ns'],
                         profile_samples=len(samples),
                         sampled_epilogue_median_ns=statistics.median(x['epilogue_ns'] for x in samples)
                             if samples else None,
                         sampled_mainloop_plus_dispatch_median_ns=statistics.median(
                             x['mainloop_plus_dispatch_ns'] for x in samples) if samples else None))
    rows.sort(key=lambda row: row['critical_path_ns'], reverse=True)
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / 'stages.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (a.out / 'summary.json').write_text(json.dumps(dict(
        evidence='verified trace reconstruction', diagnostic_only=True, timing_valid=False,
        reconstruction=reconstruction, top10=rows[:10],
        dependencies=str((a.trace / 'v2/runtime_dependencies.cuh').resolve()),
        scope='all-task wait/run/publish; sampled epilogue; remaining sampled body includes mainloop and dispatch'),
        indent=2) + '\n')


if __name__ == '__main__':
    main()
