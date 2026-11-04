#!/usr/bin/env python3
"""Run the continuation's DT-A0 diagnostics after a completed P-ABL cell."""
import argparse
import json
import math
from pathlib import Path
import subprocess

from sm90d1_common import ROOT, read, sha, write


def triggered(measured):
    a, f = (measured['arms'][arm] for arm in ('A0', 'F'))
    change = (f['tpot_ms'] / a['tpot_ms'] - 1) * 100
    threshold = max(.5, 2 * max(a['range_pct'], f['range_pct']))
    comparison = measured['comparisons']['F']
    if (not math.isclose(change, comparison['change_pct'], abs_tol=1e-9) or
            threshold != comparison['threshold_pct']):
        raise ValueError('P-ABL comparison does not match retained arm measurements')
    return change < -threshold


def summarize(receipt, predictions, winner, frozen):
    candidates = receipt['evaluated']
    if len(predictions) != len(candidates):
        raise ValueError('replay ranking requires the complete original E')
    for candidate, prediction in zip(candidates, predictions):
        if (candidate['coordinates'] != prediction['coordinates'] or
                candidate['pool_ns'] != prediction['pool_ns']):
            raise ValueError('diagnostic evaluation changed the retained candidate or score')
    fi = [i for i, c in enumerate(candidates) if c['coordinates'] == frozen]
    ai = [i for i, c in enumerate(candidates) if c['coordinates'] == winner]
    if len(fi) != 1 or len(ai) != 1:
        raise ValueError('DT-A0 requires unique retained F and A0 candidates in E')
    f, a = predictions[fi[0]], predictions[ai[0]]
    for value in (a, f):
        d = value['decomposition']
        if (d['original_ns'] != value['pool_ns'] or not math.isclose(
                sum(d[k] for k in ('synchronization_ns', 'fixed_ns', 'contention_ns', 'chain_ns')),
                d['original_ns'], abs_tol=1e-6)):
            raise ValueError('DecomposeFlow components do not reconstruct the original score')
    structure = candidates[fi[0]]['structural_class']
    return dict(F_present_in_E=True,
        F_pool_rank=1+sum(c['pool_ns'] < f['pool_ns'] for c in predictions),
        F_replay_rank=1+sum(c['replay_ns'] < f['replay_ns'] for c in predictions),
        F_structure=structure,
        F_structure_in_finalists=any(c['structural_class'] == structure for c in receipt['finalists']),
        E_size=len(candidates), replay_rank_scope='diagnostic replay of every retained E candidate; search unchanged',
        delta_A0_minus_F={k: a['decomposition'][k]-f['decomposition'][k] for k in a['decomposition']},
        delta_replay_ns=a['replay_ns']-f['replay_ns'])


def diagnose(root, cell, evaluator, out):
    out.mkdir(parents=True, exist_ok=False)
    directory = root/'ablation/pabl-v1'/cell
    measured_path = directory/'measurement/result.json'
    measured = read(measured_path)
    if measured['protocol'] != 'P-ABL' or measured['source_sha256'] != sha(directory/'arms.json'):
        raise ValueError('DT-A0 requires the completed original P-ABL manifest')
    evidence = dict(measurement=str(measured_path), measurement_sha256=sha(measured_path),
                    triggered=triggered(measured), action='report only; no new search or selection')
    if not evidence['triggered']:
        write(out/'result.json', evidence)
        return evidence
    source_path = root/'a0/finalists-v1'/cell/'source.json'
    source = read(source_path)
    if sha(source['receipt']) != source['receipt_sha256']:
        raise ValueError('A0 search receipt changed')
    receipt = read(source['receipt']); arms = read(directory/'arms.json')
    coordinates = {}; paths = [measured_path, directory/'arms.json', source_path,
                              Path(source['receipt']), evaluator, Path(__file__)]
    for arm in ('A0', 'F'):
        binary = Path(arms[arm]['binary']); identity_path = Path(str(binary)+'.identity.json')
        identity = read(identity_path)
        if identity['binary_sha256'] != sha(binary):
            raise ValueError('P-ABL selected binary changed')
        coordinates[arm] = identity['plan']['native_dense']; paths.extend((binary, identity_path))
    # Rank is over E, not just Q_short. Search receipts have no replay score for
    # unshortlisted candidates; evaluating those here cannot alter selection.
    requests = []
    for i, candidate in enumerate(receipt['evaluated']):
        coordinate_path = out/f'candidate-{i}.json'; write(coordinate_path, candidate['coordinates'])
        requests.append(dict(id=f'{cell}-E{i}', batch=measured['batch'],
            export=source['input']['export'], coordinates=str(coordinate_path.resolve()),
            decompose=candidate['coordinates'] in coordinates.values()))
        paths.append(coordinate_path)
    inputs_path = out/'cells.json'; write(inputs_path, requests)
    target = ROOT/'configs/targets/sm_90_h800.json'
    paths.extend((inputs_path, target, Path(source['input']['export'])))
    paths.extend(p for base in ('include', 'lib') for p in (ROOT/base).rglob('*') if p.is_file())
    inputs = {str(p.resolve()): sha(p) for p in paths}; write(out/'inputs.json', inputs)
    command = list(map(str, [evaluator, target, ROOT, inputs_path, out/'predictions.jsonl']))
    write(out/'command.json', command)
    with (out/'evaluate.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    if any(sha(path) != digest for path, digest in inputs.items()):
        raise ValueError('DT-A0 evaluation inputs changed')
    predictions = [json.loads(line) for line in (out/'predictions.jsonl').read_text().splitlines() if line.strip()]
    evidence.update(summarize(receipt, predictions, coordinates['A0'], coordinates['F']))
    evidence.update(predictions=str(out/'predictions.jsonl'), predictions_sha256=sha(out/'predictions.jsonl'),
                    inputs_sha256=sha(out/'inputs.json'))
    write(out/'result.json', evidence)
    return evidence


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--cell', required=True)
    p.add_argument('--evaluator', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); print(json.dumps(diagnose(a.root, a.cell, a.evaluator, a.out)))
