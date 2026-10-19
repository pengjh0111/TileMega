#!/usr/bin/env python3
"""Collect only completed guarded traces and preserve per-stage raw arithmetic."""
import argparse
from pathlib import Path

from analyze_sm90d1_ledger import analyze
from sm90d1_common import ROOT, RUN, read, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--tag', required=True)
    a = p.parse_args()
    record = read(a.state)[a.job]
    if record['status'] != 'done':
        raise ValueError('trace has not completed its guard')
    raw = Path(record['out'])
    if read(raw / 'guard_result.json')['code']:
        raise ValueError('trace guard rejected the run')
    c = read(a.config)
    selected = read(raw / 'job/selected-config.json')
    binary = RUN / 'build' / a.tag / f"{c['model']}-B{c['batch']}" / 'decode.so'
    reports = []
    for folder in sorted((raw / 'job/trace').glob('past*')):
        details, closure = analyze(folder, read(ROOT / c['model_dir'] / 'config.json'),
                                   c['batch'], read(str(binary) + '.plan.json'))
        report = dict(stages=details, steps=closure, trace=read(folder / 'trace.json'),
                      selected=selected, guard=str(raw / 'guard_result.json'))
        write(folder / 'ledger.json', report)
        reports.append(dict(past=int(folder.name[4:]), stage_rows=len(details),
                            steps=len(closure), complete_profiles=sum(r['profile_complete'] for r in details)))
    write(raw / 'job/ledger-summary.json', reports)


if __name__ == '__main__':
    main()
