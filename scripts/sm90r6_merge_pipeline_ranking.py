#!/usr/bin/env python3
"""Report R5 TP candidate ranking and the lowest-three-cell flow decomposition."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def tau(rows, field):
    concordant = discordant = tie_x = tie_y = 0
    for i, a in enumerate(rows):
        for b in rows[:i]:
            x = (a[field] > b[field]) - (a[field] < b[field])
            av, bv = a['input']['measured_tpot_ms'], b['input']['measured_tpot_ms']
            y = (av > bv) - (av < bv)
            concordant += x*y > 0
            discordant += x*y < 0
            tie_x += not x and bool(y)
            tie_y += not y and bool(x)
    denominator = (concordant + discordant + tie_x)*(concordant + discordant + tie_y)
    if not denominator:
        raise ValueError('Kendall tau is undefined for this candidate set')
    return (concordant-discordant)/math.sqrt(denominator)


def merge(paths, decompositions, output):
    groups, by_id, provenance = defaultdict(list), {}, {}
    for source in paths:
        digest = sha(source)
        for row in read(source):
            cell = row['input']; identity = cell['id']
            if identity in by_id:
                raise ValueError('duplicate candidate prediction')
            for field in ('coordinates', 'export', 'measurement', 'binary'):
                if sha(cell[field]) != cell[field+'_sha256']:
                    raise ValueError('ranking source changed: '+cell[field])
            coordinates = json.loads(Path(cell['coordinates']).read_text())
            # Native serialization can fill defaults, but every recorded field must agree.
            if any(row['coordinates'].get(k) != v for k,v in coordinates.items()):
                raise ValueError('ranking changed the measured coordinates')
            measured = json.loads(Path(cell['measurement']).read_text())
            value = (measured['tpot_ms'] if cell['candidate'] == 'REF-R3-derived' else
                     measured['medians'][cell['candidate']])
            if float(format(value, '.10g')) != cell['measured_tpot_ms'] or not math.isfinite(value) or value <= 0:
                raise ValueError('ranking measurement mismatch')
            # The C++ JSON writer uses ten significant digits; rank the exact source medians.
            cell['measured_tpot_ms'] = value
            if any(not math.isfinite(row[k]) or row[k] <= 0 for k in ('pool_ns','replay_ns','additive_ns')):
                raise ValueError('invalid ranking prediction')
            key = cell['tp'], cell['batch']
            if cell['model'] != 'qwen3_1_7b':
                raise ValueError('unexpected TP dataset model')
            by_id[identity] = row; groups[key].append(row)
            provenance[identity] = str(source), digest
    expected = {(tp,b) for tp in (2,4) for b in (1,2,4,8,16)}
    if set(groups) != expected or len(by_id) != 42:
        raise ValueError('expected the 40 measured candidates and two REF-R3-derived coordinates')
    for (tp,b), group in groups.items():
        candidates = {r['input']['candidate'] for r in group}
        if candidates != {'c0','c1','c2','c3'} | ({'REF-R3-derived'} if b == 1 else set()):
            raise ValueError('incomplete TP cell')
    result, correlations = [], {}
    def append(check, key, candidate, metric, value, reference='', note='', raw='', digest=''):
        result.append(dict(check=check, model='qwen3_1_7b', batch=key[1], tp=key[0],
            candidate=candidate, metric=metric, value=value, reference=reference, ratio='',
            status='reported; TP subset', note=note, raw=raw, raw_sha256=digest))
    sources = ';'.join(str(p) for p in paths)
    digests = ';'.join(sha(p) for p in paths)
    for key, group in sorted(groups.items()):
        correlations[key] = tau(group, 'replay_ns')
        for field in ('replay_ns', 'additive_ns'):
            append('V2-TP-subset', key, 'R5 P-SEL + REF-R3-derived' if key[1] == 1 else 'R5 P-SEL',
                field.removesuffix('_ns')+'_Kendall_tau_b', tau(group,field), len(group),
                'original measured TP dataset; additive constants/structure are in-sample; MoE dataset audit pending',
                sources, digests)
        for row in group:
            cell = row['input']; raw, digest = provenance[cell['id']]
            for field in ('pool_ns','replay_ns','additive_ns'):
                append('V2-TP-predictions', key, cell['candidate'], field, row[field],
                    cell['measured_tpot_ms']*1e6, 'reference is measured TPOT in ns; no fitting', raw, digest)
            append('V4-TP-subset', key, cell['candidate'], 'pool_evaluation_ms', row['pool_evaluation_ms'],
                200, 'includes CTG/release/evaluation; cache reused within each collector process', raw, digest)
    for field in ('replay_ns','additive_ns'):
        append('V2-TP-subset', ('all','all'), '42 candidates in 10 cells',
            field.removesuffix('_ns')+'_mean_tau_b', statistics.mean(tau(g,field) for g in groups.values()),
            note='unweighted cell mean; additive is in-sample; not the complete V2 matrix', raw=sources, digest=digests)
    worst = sorted(groups, key=lambda k:(correlations[k],k))[:3]
    decomposed = {}
    for row in read(decompositions):
        identity = row['input']['id']
        if identity not in by_id or identity in decomposed:
            raise ValueError('unexpected or duplicate decomposition candidate')
        original = by_id[identity]
        if any(row[k] != original[k] for k in ('pool_ns','replay_ns','additive_ns','coordinates')):
            raise ValueError('decomposition changed the prediction')
        d = row['decomposition']
        if d['original_ns'] != row['pool_ns']:
            raise ValueError('decomposition original differs from PoolScore')
        if not math.isclose(sum(d[k] for k in ('synchronization_ns','fixed_ns','contention_ns','chain_ns')),
                            d['original_ns'], abs_tol=1e-6):
            raise ValueError('decomposition does not reconstruct the zero-floor flow')
        decomposed[identity] = row
    if set(decomposed) != {r['input']['id'] for k in worst for r in groups[k]}:
        raise ValueError('decomposition must cover exactly the lowest three cells')
    for key in worst:
        pairs = [(a,b) for a in groups[key] for b in groups[key] if a['replay_ns'] < b['replay_ns'] and
                 a['input']['measured_tpot_ms'] > b['input']['measured_tpot_ms']]
        if not pairs:
            raise ValueError('lowest-cell report requires a measured rank inversion')
        a,b = max(pairs, key=lambda p:p[0]['input']['measured_tpot_ms']-p[1]['input']['measured_tpot_ms'])
        ca,cb = a['input'],b['input']; da,db = decomposed[ca['id']]['decomposition'],decomposed[cb['id']]['decomposition']
        note = (f"A-B; A={ca['candidate']} measured {ca['measured_tpot_ms']} ms; "
                f"B={cb['candidate']} measured {cb['measured_tpot_ms']} ms; largest measured-gap inversion; "
                'DecomposeFlow counterfactual deltas are diagnostic')
        for field in ('original_ns','synchronization_ns','fixed_ns','contention_ns','chain_ns','pg_upper_bound_ns'):
            append('V2-TP-decomposition', key, ca['candidate']+' minus '+cb['candidate'], 'delta_'+field,
                da[field]-db[field], note=note, raw=str(decompositions), digest=sha(decompositions))
    if output.exists():
        with output.open() as stream:
            existing = list(csv.DictReader(stream))
        result = [r for r in existing if r['check'] not in
                  ('V2-TP-subset','V2-TP-predictions','V4-TP-subset','V2-TP-decomposition')] + result
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(result[0]), lineterminator='\n')
        writer.writeheader();writer.writerows(result)
    print(json.dumps(dict(candidates=len(by_id), cells=len(groups), decomposed_cells=worst,
        replay_mean_tau=statistics.mean(correlations.values()),
        additive_mean_tau=statistics.mean(tau(g,'additive_ns') for g in groups.values()), rows=len(result))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, nargs='+', required=True)
    p.add_argument('--decompositions', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args();merge(a.predictions,a.decompositions,a.out)
