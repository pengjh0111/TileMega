#!/usr/bin/env python3
"""Apply the D1 ledger arithmetic to accepted static-program stage traces."""
import argparse
from pathlib import Path

from analyze_sm90d1_ledger import analyze
from collect_sm90d1_diagnostics import latest_tsv
from sm90d1_common import ROOT, read, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--job', required=True)
    a = p.parse_args()
    row = read(a.state)[a.job]
    if row['status'] != 'done':
        raise ValueError('trace must complete before analysis')
    raw = Path(row['out'])
    if read(raw / 'guard_result.json')['code']:
        raise ValueError('guard rejected the trace')
    selected = read(raw / 'job/selected-config.json')
    if selected['stage'] != 'trace':
        raise ValueError('a stage-only trace is required')
    config = read(selected['config'])
    command = selected['command']
    binary = command[command.index('--decode') + 1]
    plan = read(binary + '.plan.json')
    for folder in sorted((raw / 'job/trace').glob('past*')):
        metadata = read(folder / 'trace.json')
        if metadata.get('protocol_version') != 3:
            raise ValueError('nonwrapping trace protocol is required')
        for name in ('stage_trace.tsv', 'step_trace.tsv'):
            latest_tsv(dict(folder=folder, meta=metadata), name)
        stages, steps = analyze(folder, read(ROOT / config['model_dir'] / 'config.json'),
                                config['batch'], plan)
        write(folder / 'ledger.json', dict(stages=stages, steps=steps, trace=metadata,
                                         selected=selected, diagnostic_only=True))


if __name__ == '__main__':
    main()
