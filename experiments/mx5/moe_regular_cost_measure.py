#!/usr/bin/env python3
"""Bind new MoE regular-stage samples to exact queries and unchanged dense drivers."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


FIELDS = {
    'gemm': ('n', 'k', 'b', 'tn', 'wn', 'depth', 'threads', 'prefetch', 'op', 'norm', 'part'),
    'attention': ('dim', 'q', 'b', 'hkv', 'length', 'b_min', 'qk_norm', 'capacity', 'threads', 'prefetch_kind'),
}


def checked(ref):
    path = Path(ref['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256']:
        raise ValueError('regular cost input changed: ' + str(path))
    return path


def record(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def validate_key(key, query, reference, source):
    kind = key['kind']
    if kind not in FIELDS:
        raise ValueError('unknown regular-stage body')
    for field in FIELDS[kind]:
        if key[field] != query['key'][field]:
            raise ValueError('regular cost/query coordinate differs: ' + field)
        if field not in ('prefetch', 'prefetch_kind') and reference['key'][field] != key[field]:
            raise ValueError('regular cost/R_next coordinate differs: ' + field)
    if reference['key']['prefetch' if kind == 'gemm' else 'prefetch_kind'] != 0:
        raise ValueError('regular R_next must use zero prefetch')
    if query['key']['source_sha256'] != source or reference['key']['source_sha256'] != source:
        raise ValueError('regular cost/reference source differs')
    if key['library'] != query['key']['library']['path'] or key['binary_sha256'] != query['key']['library']['sha256']:
        raise ValueError('regular cost binary differs from actual query')
    if key['r_next'] != reference['registers']:
        raise ValueError('regular R_next differs from actual reference query')
    if query['returncode'] != 0 or query['local_bytes'] or query['static_smem'] or query['resident'] != 1:
        raise ValueError('regular payload has rejected device resources')
    if not 0 < query['registers'] <= (227 if key['threads'] == 576 else 255):
        raise ValueError('regular payload exceeds register limit')


def validate_inputs(epoch_path, keys_path):
    epoch = json.loads(epoch_path.read_text())
    if epoch.get('kind') != 'mx5_moe_regular_source_bound_cost_epoch' or epoch.get('final') is not False:
        raise ValueError('expected a regular-stage cost epoch')
    if epoch['R_cap'] != {'t512': 248, 't576': 220}:
        raise ValueError('regular epoch changes the prompt register budget')
    if checked(epoch['driver']).resolve() != Path(__file__).resolve():
        raise ValueError('regular measurement driver differs from epoch')
    checked(epoch['packing_helper'])
    checked(epoch['compiler'])
    for helper in epoch['helpers'].values():
        checked(helper)
    for ref in epoch['inputs']:
        checked(ref)
    queries = json.loads(checked(epoch['queries']).read_text())['rows']
    references = json.loads(checked(epoch['references']).read_text())['rows']
    keys = json.loads(keys_path.read_text())
    if not keys or len({key['kind'] for key in keys}) != 1:
        raise ValueError('one nonempty body kind per fresh process is required')
    for key in keys:
        query, reference = queries[key['query_index']], references[key['reference_index']]
        validate_key(key, query, reference, epoch['source_sha256'][key['kind']])
        for row in (query, reference):
            checked(row['key']['library'])
            checked(row['key']['metadata'])
    return epoch, keys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epoch', type=Path, required=True)
    p.add_argument('--keys', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--round', type=int, choices=(1, 2, 3), required=True)
    p.add_argument('--iterations', type=int, default=128)
    a = p.parse_args()
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1' or a.out.exists() or a.iterations < 3:
        p.error('guard, fresh output and at least three iterations are required')
    epoch, keys = validate_inputs(a.epoch, a.keys)
    helper = checked(epoch['helpers'][keys[0]['kind']])
    spec = importlib.util.spec_from_file_location('bound_regular_stage_driver', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv = [str(helper), '--keys', str(a.keys), '--out', str(a.out), '--round', str(a.round),
        '--iterations', str(a.iterations)]
    module.main()
    validate_inputs(a.epoch, a.keys)
    rows = [json.loads(line) for line in a.out.read_text().splitlines()]
    if len(rows) != len(keys) or any(row['key'] != key or row['round'] != a.round for row, key in zip(rows, keys)):
        raise ValueError('regular measurement output differs from request')
    proof = dict(PID=os.getpid(), epoch=record(a.epoch), keys=record(a.keys), helper=record(helper),
        raw=record(a.out), driver=record(Path(__file__)), round=a.round, source_bound_cost_samples=True,
        raw_sample_only=True, not_model_or_default_acceptance=True)
    with a.out.with_suffix('.invocation.json').open('x') as f:
        json.dump(proof, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
