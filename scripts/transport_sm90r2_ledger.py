#!/usr/bin/env python3
"""Reuse stage measurements only after both control and trace instruction audits."""
import argparse
import csv
from pathlib import Path
import sys

from sm90d1_common import ROOT, read, sha, write


def table(path):
    return list(csv.DictReader(path.open()))


def emit(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-ledger', type=Path, required=True)
    p.add_argument('--source-overhead', type=Path, required=True)
    p.add_argument('--source-tag', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--trace-audits', type=Path, required=True)
    p.add_argument('--control-audits', type=Path, required=True)
    p.add_argument('--measured-ledger', type=Path, required=True)
    p.add_argument('--measured-overhead', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out = a.out.resolve()
    if a.out.exists() or not a.out.is_relative_to(ROOT / 'runs/sm90r2'):
        raise ValueError('choose a new evidence directory')
    sys.path.insert(0, str(ROOT / 'python'))
    from tilemega.build.identity import verify
    source, old_overhead = table(a.source_ledger), table(a.source_overhead)
    measured, new_overhead = table(a.measured_ledger), table(a.measured_overhead)
    ledger, overhead, receipts = [], [], []
    cells = {(r['model'], r['batch']) for r in old_overhead if r['tag'] == a.source_tag+'-stage'}
    for model, batch in sorted(cells):
        cell = f'{model}-B{batch}'; proofs = []
        for trace, directory in ((False, a.control_audits), (True, a.trace_audits)):
            path = directory / cell / 'comparison.json'; comparison = read(path)
            if trace and not comparison.get('diagnostic'):
                raise ValueError('trace comparison was not made in diagnostic mode')
            for key, tag in (('reference', a.source_tag), ('candidate', a.tag)):
                binary = ROOT / f'runs/sm90r2/build/{tag+"-stage" if trace else tag}/{cell}/decode.so'
                identity = verify(binary)
                if identity['binary_sha256'] != comparison[key+'_sha256'] or bool(identity['trace']) != trace:
                    raise ValueError('instruction audit no longer binds the compared artifact')
            proofs.append(comparison['pass'])
            receipts.append(dict(model=model, batch=int(batch), trace=trace,
                                 path=str(path), sha256=sha(path), equivalent=comparison['pass']))
        reuse = all(proofs)
        tag = a.source_tag if reuse else a.tag
        selected = [r for r in (source if reuse else measured)
                    if (r['model'], r['batch'], r['tag']) == (model, batch, tag+'-stage')]
        timing = [r for r in (old_overhead if reuse else new_overhead)
                  if (r['model'], r['batch'], r['tag']) == (model, batch, tag+'-stage')]
        if len(selected) != 8 or len(timing) != 1:
            raise ValueError(f'{cell}: missing eight stage classes or matched timing control')
        for rows, output in ((selected, ledger), (timing, overhead)):
            for row in rows:
                output.append(dict(row, tag=a.tag+'-stage', measured_tag=tag+'-stage',
                    transport='instruction_identity' if reuse else 'remeasured'))
    a.out.mkdir(parents=True)
    emit(a.out/'stage_ledger.csv', ledger); emit(a.out/'trace_overhead.csv', overhead)
    write(a.out/'transport.json', dict(evidence='verified source and device instruction identity; '
        'transport is diagnostic reuse, not a new performance measurement', audits=receipts,
        inputs=[dict(path=str(path), sha256=sha(path)) for path in
                (a.source_ledger, a.source_overhead, a.measured_ledger, a.measured_overhead)],
        output_sha256={name:sha(a.out/name) for name in ('stage_ledger.csv','trace_overhead.csv')}))
    print(f'{len(cells)} cells; {sum(r["transport"]=="instruction_identity" for r in overhead)} reused')


if __name__ == '__main__':
    main()
