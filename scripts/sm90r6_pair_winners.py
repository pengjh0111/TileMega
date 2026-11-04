#!/usr/bin/env python3
"""Compare distinct selected coordinates with P-PAIR blocks on one card."""
import argparse
import itertools
import statistics
from pathlib import Path

from sm90d1_common import read, sha, write
from sm90r3_llm import processes, span
from sm90r4_dispatch import run_unit


def prepare(arms, model, batch):
    unique = {}; aliases = {}; inputs = {}
    for label, arm in arms.items():
        binary = Path(arm['binary']).resolve()
        identity_path = Path(str(binary) + '.identity.json')
        identity = read(identity_path); digest = sha(binary)
        c2 = read(arm['c2'])
        if (identity['binary_sha256'] != digest or not c2['passed'] or
                len(c2['processes']) < 5 or c2['steps'] != 1024 or
                c2['model'] != model or c2['batch'] != batch or
                c2['artifacts']['candidate']['sha256'] != digest):
            raise ValueError('winner identity or C-2 mismatch: ' + label)
        for path in (binary, identity_path, Path(arm['c2'])):
            inputs[str(path.resolve())] = sha(path)
        coordinates = identity['plan']['native_dense']
        representative = next((k for k, v in unique.items()
                               if v['coordinates'] == coordinates), None)
        if representative is None:
            representative = label
            unique[label] = dict(coordinates=coordinates, decode=str(binary), prefill='prompt')
        aliases[label] = representative
    pairs = list(itertools.combinations(unique, 2))
    sequence = [label for pair in pairs for label in list(pair) * processes(model)]
    return unique, aliases, pairs, sequence, inputs


def summarize(rows, pairs, model):
    count = processes(model); results = []
    if len(rows) != len(pairs) * 2 * count:
        raise ValueError('incomplete pair blocks')
    for index, pair in enumerate(pairs):
        block = rows[index * 2 * count:(index + 1) * 2 * count]
        if [r['arm'] for r in block] != list(pair) * count:
            raise ValueError('P-PAIR alternation changed')
        summaries = {}
        tokens = [read(r['tokens']) for r in block]
        if any(t != tokens[0] for t in tokens):
            raise ValueError('P-PAIR output tokens differ')
        for label in pair:
            values = [r['tpot_ms'] for r in block if r['arm'] == label]
            summaries[label] = dict(tpot_ms=statistics.median(values),
                                    range_pct=span(values), processes=count)
        ref, cand = [summaries[k] for k in pair]
        threshold = max(.5, 2 * max(ref['range_pct'], cand['range_pct']))
        change = (cand['tpot_ms'] / ref['tpot_ms'] - 1) * 100
        results.append(dict(reference=pair[0], candidate=pair[1], arms=summaries,
                            change_pct=change, threshold_pct=threshold,
                            decision='indistinguishable' if abs(change) <= threshold
                            else 'faster' if change < 0 else 'slower'))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--batch', type=int, choices=(1, 16), required=True)
    p.add_argument('--arms', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    unique, aliases, pairs, sequence, inputs = prepare(read(a.arms), a.model, a.batch)
    a.out.mkdir(parents=True, exist_ok=False)
    inputs.update({str(a.arms.resolve()): sha(a.arms), str(Path(__file__).resolve()): sha(__file__)})
    write(a.out / 'inputs.json', inputs)
    result = dict(protocol='P-PAIR', model=a.model, batch=a.batch, aliases=aliases,
                  pairs=[], reason='identical selected coordinates')
    if pairs:
        # One dispatch unit keeps every pair block on the same physical card.
        spec = dict(name='sm90r6-pairs-' + a.out.parent.name, kind='sel',
                    protocol='P-PAIR blocks', model=a.model, batch=a.batch,
                    gpu='auto', concurrency='trial', order=sequence,
                    arms={k: {f: v[f] for f in ('decode', 'prefill')} for k, v in unique.items()})
        write(a.out / 'spec.json', spec)
        raw = run_unit(a.out / 'spec.json', a.out / 'unit', 'trial')
        rows = raw['rows']; gpus = {r['gpu'] for r in rows}
        if len(gpus) != 1 or 6 in gpus:
            raise ValueError('P-PAIR did not stay on one eligible card')
        checkpoints = set()
        for row in rows:
            paths = list(Path(row['tokens']).parent.rglob('measurements.json'))
            if len(paths) != 1:
                raise ValueError('ambiguous checkpoint evidence')
            checkpoints.add(read(paths[0])['checkpoint']['checkpoint_sha256'])
        if len(checkpoints) != 1:
            raise ValueError('P-PAIR checkpoints differ')
        result.update(pairs=summarize(rows, pairs, a.model), reason='distinct selected coordinates',
                      gpu=gpus.pop(), checkpoint=checkpoints.pop(),
                      raw=str((a.out / 'unit/result.json').resolve()),
                      ownership='provisional_container_pid_namespace', concurrency='trial')
    if any(sha(path) != digest for path, digest in inputs.items()):
        raise ValueError('P-PAIR inputs changed')
    write(a.out / 'result.json', result)


if __name__ == '__main__':
    main()
