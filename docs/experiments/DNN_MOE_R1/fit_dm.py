#!/usr/bin/env python3
"""Fit DM body costs from identity-linked, numerically gated observations."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

BODIES = frozenset(('im2col_gemm', 'depthwise', 'pool', 'global_pool_partial',
    'global_pool_reduce', 'layernorm', 'embedding_sum', 'encoder_attention',
    'dwpw_fused', 'moe_topk_dispatch', 'moe_gemm', 'moe_combine'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_nonnegative(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or \
            not np.isfinite(value) or value < 0:
        raise ValueError('invalid ' + field)
    return float(value)


def fit(train, heldout):
    """Exact NNLS faces for fixed + byte + flop costs; report heldout residuals."""
    if not train or not heldout:
        raise ValueError('each family requires training and heldout observations')
    def matrix(rows):
        x, y = [], []
        for row in rows:
            x.append([1, finite_nonnegative(row['bytes'], 'bytes'),
                      finite_nonnegative(row['flops'], 'flops')])
            ns = finite_nonnegative(row['body_ns'], 'body_ns')
            if ns == 0:
                raise ValueError('body_ns must be positive')
            y.append(ns)
        return np.asarray(x), np.asarray(y)
    x, y = matrix(train)
    hx, hy = matrix(heldout)
    scale = np.maximum(np.max(np.abs(x), axis=0), 1)
    normalized = x / scale
    candidates = []
    for size in range(1, 4):
        for columns in itertools.combinations(range(3), size):
            reduced = normalized[:, columns]
            if np.linalg.matrix_rank(reduced) != size:
                continue
            coefficient = np.linalg.lstsq(reduced, y, rcond=None)[0]
            if np.any(coefficient < 0):
                continue
            answer = np.zeros(3)
            answer[list(columns)] = coefficient
            error = float(np.sum((normalized @ answer - y) ** 2))
            candidates.append((error, size, columns, answer / scale))
    if not candidates:
        raise ValueError('nonnegative body fit failed')
    _, _, columns, answer = min(candidates, key=lambda item: item[:3])
    def residuals(rows, mx, actual):
        return [dict(case_id=row['case_id'], artifact_id=row.get('artifact_id'),
                     actual_ns=float(a), predicted_ns=float(p),
                     residual_ns=float(p-a), relative_error=float(abs(p-a)/a))
                for row, a, p in zip(rows, actual, mx @ answer)]
    tr = residuals(train, x, y)
    hr = residuals(heldout, hx, hy)
    rank = int(np.linalg.matrix_rank(normalized))
    return dict(fixed_ns=float(answer[0]), byte_ns=float(answer[1]),
                flop_ns=float(answer[2]), samples=len(train),
                median_relative_error=float(np.median([r['relative_error'] for r in tr])),
                heldout_samples=len(heldout),
                heldout_median_relative_error=float(np.median([r['relative_error'] for r in hr])),
                heldout_max_relative_error=max(r['relative_error'] for r in hr),
                feature_rank=rank, coefficients_identifiable=rank == 3,
                active_features=[('fixed', 'bytes', 'flops')[c] for c in columns],
                training_residuals=tr, heldout_residuals=hr)


def linked_json(row, name):
    path = Path(row[name]).resolve()
    if sha(path) != row[name + '_sha256']:
        raise ValueError(name + ' identity changed')
    return json.loads(path.read_text())


def validate_observation(row):
    if row['body_kind'] not in BODIES or row['section'] not in ('serving', 'serving_paged'):
        raise ValueError('unknown DM body or target section')
    family = row['family']
    if family != row['body_kind'] and not family.startswith(row['body_kind'] + '_'):
        raise ValueError('fit family does not identify its body')
    if row['split'] not in ('train', 'heldout'):
        raise ValueError('observation split must be preregistered')
    identity = linked_json(row, 'identity')
    if identity.get('schema') not in ('tilemega.dm1.identity.v1',
                                     'tilemega.dm1.native-test.identity.v1'):
        raise ValueError('unknown compiled identity schema')
    claimed = identity.pop('artifact_id')
    computed = hashlib.sha256(json.dumps(identity, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    if claimed != computed or claimed != row['artifact_id']:
        raise ValueError('compiled artifact identity differs')
    binary_hash = identity.get('so_sha256', identity.get('binary_sha256'))
    if binary_hash != sha(row['binary']):
        raise ValueError('compiled binary changed')
    gate = linked_json(row, 'numerical_gate')
    case = gate.get('cases', {}).get(row['case_id'], {})
    if gate.get('passed') is not True or case.get('passed') is not True or \
            case.get('artifact_id') != claimed or case.get('body_kind') != row['body_kind']:
        raise ValueError('observation lacks its body/case numerical gate')
    if case.get('synchronization_required'):
        rate = case.get('fresh_processes', {})
        if rate.get('passes', 0) < 50 or rate.get('passes') != rate.get('attempts'):
            raise ValueError('synchronization gate requires 50 fresh passing processes')
    work = linked_json(row, 'work')
    declared = work.get('cases', {}).get(row['case_id'], {})
    if work.get('source') != 'l_sem' or declared.get('bytes') != row['bytes'] or \
            declared.get('flops') != row['flops']:
        raise ValueError('observation work differs from L-sem derivation')
    return identity['arch']


def fit_observations(rows):
    groups, seen = {}, set()
    for row in rows:
        arch = validate_observation(row)
        key = (arch, row['section'], row['family'])
        # Repeats of one geometry are aggregated by the timing driver. A
        # geometry cannot also be a heldout sample in the same family.
        case_key = (key, row['case_id'])
        if case_key in seen:
            raise ValueError('duplicate case or training/heldout leakage')
        seen.add(case_key)
        groups.setdefault(key, {'train': [], 'heldout': []})[row['split']].append(row)
    if not groups:
        raise ValueError('empty calibration sample set')
    return {key: fit(group['train'], group['heldout']) for key, group in sorted(groups.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--target-out', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.target.resolve() == args.target_out.resolve():
        raise ValueError('calibration preserves its input target')
    samples = json.loads(args.samples.read_text())
    if samples.get('schema') != 'tilemega.dm1.calibration.samples.v1':
        raise ValueError('unknown calibration sample schema')
    result = fit_observations(samples['observations'])
    target = json.loads(args.target.read_text())
    body = target.setdefault('calibration_by_dtype', {}).setdefault('bf16', {}).setdefault('task_body', {})
    report = dict(evidence='verified', scope='identity-linked calibration fits',
                  samples_sha256=sha(args.samples), target_input_sha256=sha(args.target), families=[])
    for (arch, section, family), model in result.items():
        if arch != target['arch_tag']:
            raise ValueError('sample architecture differs from target')
        table = body.setdefault(section, {})
        if family in table:
            raise ValueError('existing body fit would be overwritten: ' + family)
        table[family] = {k: v for k, v in model.items() if not k.endswith('_residuals')}
        report['families'].append(dict(arch=arch, section=section, family=family, **model))
    args.target_out.parent.mkdir(parents=True, exist_ok=True)
    args.target_out.write_text(json.dumps(target, indent=2) + '\n')
    report['target_output_sha256'] = sha(args.target_out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
