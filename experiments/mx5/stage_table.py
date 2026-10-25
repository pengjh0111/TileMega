#!/usr/bin/env python3
"""Aggregate measured MX-5 JSONL rounds without changing target defaults.

Every key includes actual threads and prefetch coordinates. Qualified rows also
need a same-key Pw=0/PF=0 resource reference, inline as r_next_reference or in
--resources JSON {"references": [{"kind": ..., "key": ..., "resources": ...,
"provenance": ...}]}. References use the same canonical key as output rows.
--diagnostic omits R_next and is explicitly unqualified. Neither mode asserts
that the supplied keys cover the complete candidate domain.
"""
import argparse
import importlib.util
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

REPO = Path(__file__).resolve().parents[2]
IMPLEMENTATIONS = {f'mx5_{kind}_v2': kind for kind in ('gemm', 'head', 'attention')}
IDENTITY = ('executor', 'implementation', 'source_sha256', 'binary_sha256', 'base_head')
RESOURCE_FIELDS = ('mtreg', 'local_bytes', 'static_smem_bytes', 'resident_ctas')
COMMON = ('threads',)
FIELDS = {
    'gemm': ('n', 'k', 'b', 'op', 'norm', 'tn', 'wn', 'depth', 'part', *COMMON, 'prefetch'),
    'head': ('n', 'k', 'b', 'tn', 'wn', 'depth', *COMMON, 'prefetch'),
    'attention': ('dim', 'q', 'hkv', 'b', 'length', 'b_min', 'qk_norm', 'capacity',
                  *COMMON, 'prefetch_kind'),
}


_probe_spec = importlib.util.spec_from_file_location('mx5_compiler_probe', Path(__file__).with_name('compiler_probe.py'))
compiler_probe = importlib.util.module_from_spec(_probe_spec)
_probe_spec.loader.exec_module(compiler_probe)


class TableError(ValueError):
    pass


def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise TableError(f'{label} must be an integer >= {minimum}')
    return value


def identity(row):
    if not isinstance(row, dict):
        raise TableError('provenance must be an object')
    result = {}
    for field in IDENTITY:
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise TableError(f'missing provenance {field}')
        if field.endswith('_sha256') and not re.fullmatch('[0-9a-f]{64}', value):
            raise TableError(f'invalid provenance {field}')
        result[field] = value
    if result['executor'] != 'mx_static_stage_bench' or result['implementation'] not in IMPLEMENTATIONS:
        raise TableError('unsupported executor/implementation')
    return result


def kind_of(row):
    result = IMPLEMENTATIONS.get(row.get('implementation'))
    if result is None:
        raise TableError('unsupported implementation')
    if 'kind' in row and row['kind'] != result:
        raise TableError('kind/implementation conflict')
    return result


def coordinate(row, name, aliases=(), default=None):
    values = [where[field] for where in (row['key'], row) for field in (name, *aliases) if field in where]
    if not values:
        if default is not None:
            return default
        raise TableError(f'missing coordinate {name}')
    for value in values:
        integer(value, f'coordinate {name}')
    if len(set(values)) != 1:
        raise TableError(f'coordinate conflict for {name}')
    return values[0]


def canonical(row, kind):
    if not isinstance(row.get('key'), dict):
        raise TableError('missing key object')
    allowed = set(FIELDS[kind]) | {'library'}
    if kind != 'attention':
        allowed |= {'pw'}
    unknown = set(row['key']) - allowed
    if unknown:
        raise TableError(f'unknown key coordinates: {sorted(unknown)}')
    metadata = None
    if kind != 'attention' and 'depth' not in row['key'] and 'depth' not in row:
        library = row['key'].get('library')
        if not isinstance(library, str):
            raise TableError('missing depth requires explicit library build metadata')
        path = Path(library).with_suffix('.json')
        if not path.is_absolute():
            path = REPO / path
        try:
            metadata = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            raise TableError(f'cannot recover depth from build metadata: {path}') from error
        for field in ('source_sha256', 'binary_sha256', 'base_head'):
            if metadata.get(field) != row.get(field):
                raise TableError(f'build metadata identity conflict for {field}')
        if metadata.get('body') != kind or metadata.get('bench') is not True or metadata.get('returncode') != 0:
            raise TableError('expected successful matching stage-bench build metadata')
        row = dict(row, depth=metadata.get('depth'))
    key = {}
    for name in FIELDS[kind]:
        key[name] = coordinate(row, name, aliases=('pw',) if name == 'prefetch' else (),
                               default=1088 if name == 'capacity' else None)
    if metadata is not None:
        for field in ('tn', 'depth', 'threads'):
            if metadata.get(field) != key[field]:
                raise TableError(f'build metadata coordinate conflict for {field}')
        width = metadata.get('pw')
        if (metadata['depth'] if width is None else width) != key['prefetch']:
            raise TableError('build metadata coordinate conflict for prefetch')
    if key['threads'] not in (512, 576) or key['b'] not in (1, 2, 4, 8, 16):
        raise TableError('invalid threads/batch coordinate')
    if kind == 'attention':
        if (key['dim'] not in (64, 128) or key['q'] not in (2, 3, 4, 8) or key['hkv'] < 1
                or key['b_min'] not in (1, 2, 4, 8) or key['prefetch_kind'] not in (0, 1, 2)
                or key['qk_norm'] not in (0, 1) or (key['qk_norm'] and key['dim'] != 128)
                or key['capacity'] != 1088 or not 1 <= key['length'] <= key['capacity']):
            raise TableError('invalid attention coordinates')
    else:
        if (key['tn'] not in (16, 32, 64, 128) or key['wn'] not in ((4, 8) if kind == 'head' else (1, 2, 4, 8))
                or not 1 <= key['depth'] <= 4 or key['n'] <= 0 or key['n'] % key['tn']
                or key['k'] <= 0 or key['k'] % 32
                or key['tn'] // 16 * 4 + key['depth'] * (key['tn'] // 16 + 1) * 4 > 160
                or (key['prefetch'] and key['prefetch'] < key['depth'])):
            raise TableError('invalid GEMM/head coordinates')
        tiles = key['n'] // key['tn']
        if key['wn'] > (tiles + 103) // 104:
            raise TableError('WN exceeds the tile geometry bound')
        if (key['depth'] - 1) * (key['tn'] // 16 + 1) + key['prefetch'] * (key['tn'] // 16) > 63:
            raise TableError('self-prefetch wait count exceeds 63')
        if kind == 'gemm' and key['part'] == 0 and tiles * (key['k'] // 32) < 104:
            raise TableError('StreamK units must cover all 104 CTAs')
        if kind == 'gemm' and (key['op'] not in (0, 1, 2) or key['norm'] not in (0, 1)
                               or key['part'] not in (0, 1) or (key['part'] == 0 and key['wn'] != 1)
                               or (key['op'] == 2 and key['tn'] < 32)):
            raise TableError('invalid GEMM operation coordinates')
    return key


def key_id(kind, key):
    return kind, tuple(key[field] for field in FIELDS[kind])


def resources(row, threads):
    if not isinstance(row, dict):
        raise TableError('resources must be an object')
    result = {field: integer(row.get(field), field) for field in RESOURCE_FIELDS}
    cap = 227 if threads == 576 else 255
    if not 1 <= result['mtreg'] <= cap or result['local_bytes']:
        raise TableError(f'resource rejection: threads={threads}, MTreg={result["mtreg"]}, stack={result["local_bytes"]}')
    if result['resident_ctas'] != 1 or result['static_smem_bytes'] != 0:
        raise TableError('stage resources require exactly one resident CTA and zero static shared memory')
    return result


def validate_measurement(row):
    provenance = identity(row)
    kind = kind_of(row)
    key = canonical(row, kind)
    res = resources(row, key['threads'])
    round_number = integer(row.get('round'), 'round', 1)
    if round_number not in (1, 2, 3):
        raise TableError('round must be 1, 2, or 3')
    for field in ('median_ns', 'p90_ns'):
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise TableError(f'{field} must be finite and positive')
    if row['p90_ns'] < row['median_ns']:
        raise TableError('p90_ns is below median_ns')
    for field in ('iterations', 'copies'):
        integer(row.get(field), field, 1)
    active = row.get('active_working_set_bytes', row.get('working_set_bytes'))
    if integer(active, 'active working_set_bytes') < 256 * 1024**2:
        raise TableError('active working set must cover at least 256 MiB')
    if row['iterations'] <= row['copies']:
        raise TableError('iterations must visit every working-set copy')
    return kind, key, provenance, res


def load_runs(paths):
    records, inputs, seen = [], [], set()
    for path in paths:
        path = Path(path).resolve()
        if path in seen:
            raise TableError(f'duplicate input file: {path}')
        seen.add(path)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        rounds = set()
        for line_number, line in enumerate(raw.decode().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TableError('measurement must be an object')
                kind, key, provenance, res = validate_measurement(row)
            except (ValueError, TypeError) as error:
                raise TableError(f'{path}:{line_number}: {error}') from error
            rounds.add(row['round'])
            records.append(dict(row=row, kind=kind, key=key, provenance=provenance, resources=res,
                                input=dict(path=str(path), sha256=digest, line=line_number)))
        if len(rounds) != 1:
            raise TableError(f'each process JSONL must contain exactly one round: {path}')
        inputs.append(dict(path=str(path), sha256=digest, round=next(iter(rounds))))
    if not records:
        raise TableError('no measured rows')
    return records, inputs


def load_references(paths):
    result, provenance = {}, []
    for path in paths:
        path = Path(path).resolve()
        raw = path.read_bytes()
        document = json.loads(raw)
        if not isinstance(document, dict) or not isinstance(document.get('references'), list):
            raise TableError('resource input must contain a references array')
        provenance.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest()))
        for reference in document['references']:
            kind, key, _, _ = validate_reference(reference)
            index = key_id(kind, key)
            if index in result:
                raise TableError('duplicate resource reference key')
            result[index] = reference
    return result, provenance


def validate_reference(reference):
    if not isinstance(reference, dict):
        raise TableError('R_next reference must be an object')
    provenance = identity(reference.get('provenance', {}))
    kind = kind_of(dict(provenance, kind=reference.get('kind')))
    key = canonical(dict(reference, **provenance), kind)
    try:
        res = compiler_probe.validate(reference) if 'compiler_probe_only' in reference else resources(reference.get('resources', {}), key['threads'])
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise TableError(str(error)) from error
    if key['prefetch_kind' if kind == 'attention' else 'prefetch'] != 0:
        raise TableError('R_next reference must use Pw=0/PF=0')
    return kind, key, provenance, res


def resolve_reference(samples, mapping):
    first = samples[0]
    expected = dict(first['key'])
    expected['prefetch_kind' if first['kind'] == 'attention' else 'prefetch'] = 0
    mapped = mapping.get(key_id(first['kind'], expected))
    inline = [s['row'].get('r_next_reference') for s in samples]
    if any(reference is not None for reference in inline) and not all(reference is not None for reference in inline):
        raise TableError('R_next reference is missing from some rounds')
    candidates = [reference for reference in [*inline, mapped] if reference is not None]
    if not candidates:
        raise TableError('missing R_next resource reference (use --diagnostic for unqualified measurements)')
    if any(reference != candidates[0] for reference in candidates[1:]):
        raise TableError('conflicting R_next resource references')
    kind, key, provenance, res = validate_reference(candidates[0])
    if kind != first['kind'] or key != expected:
        raise TableError('R_next reference coordinates do not match the exact measured body key')
    for field in ('executor', 'implementation', 'source_sha256'):
        if provenance[field] != first['provenance'][field]:
            raise TableError(f'R_next reference identity conflict for {field}')
    return res['mtreg'], candidates[0]


def aggregate(paths, resource_paths=(), diagnostic=False, card=None):
    if not diagnostic and (not isinstance(card, str) or not card.strip()):
        raise TableError('qualified rows require --card provenance')
    records, inputs = load_runs(paths)
    references, resource_inputs = load_references(resource_paths)
    groups = {}
    for record in records:
        groups.setdefault(key_id(record['kind'], record['key']), []).append(record)
    rows = []
    for _, samples in sorted(groups.items()):
        rounds = [sample['row']['round'] for sample in samples]
        if len(rounds) != len(set(rounds)):
            raise TableError(f'duplicate round for {samples[0]["kind"]} key {samples[0]["key"]}')
        if sorted(rounds) != [1, 2, 3]:
            raise TableError(f'missing round: need exactly rounds 1/2/3 for {samples[0]["key"]}')
        samples.sort(key=lambda sample: sample['row']['round'])
        first = samples[0]
        for sample in samples[1:]:
            if sample['provenance'] != first['provenance']:
                raise TableError('round provenance conflict (binary/source/executor/implementation/base_head)')
            if sample['resources'] != first['resources']:
                raise TableError('round resource conflict')
        medians = [sample['row']['median_ns'] for sample in samples]
        median = statistics.median(medians)
        row = dict(kind=first['kind'], key=first['key'], median_ns=median,
                   p90_ns=statistics.median(sample['row']['p90_ns'] for sample in samples),
                   processes=3, spread=round((max(medians) - min(medians)) / median, 4),
                   resources=first['resources'], provenance=first['provenance'],
                   rounds=[dict(round=sample['row']['round'], input=sample['input'],
                                measurement=sample['row']) for sample in samples])
        if not diagnostic:
            row['r_next'], row['r_next_reference'] = resolve_reference(samples, references)
        rows.append(row)
    probe_inputs = {}
    for row in rows:
        reference = row.get('r_next_reference', {})
        if reference.get('compiler_probe_only'):
            for descriptor in (reference['probe'][name] for name in ('binary','build_metadata','source_archive')):
                path = str(Path(descriptor['path']).resolve())
                old = probe_inputs.setdefault(path, descriptor)
                if old != descriptor:
                    raise TableError('compiler probe input changed during aggregation: ' + path)
    return dict(schema_version=2, table_kind='mx5_decode_stage_costs', qualified=not diagnostic,
                qualification_scope='measured_rows_only', full_table=False,
                coverage='provided_keys; candidate domain not enumerated',
                diagnostic_reasons=['R_next qualification disabled'] if diagnostic else [],
                provenance=dict(tool='experiments/mx5/stage_table.py', date=datetime.date.today().isoformat(),
                                compiler_probe_policy='P0 references supply unexecuted attributes; executed stage/plan gates are unchanged',
                                card=card, inputs=inputs, resource_inputs=resource_inputs, compiler_probe_inputs=list(probe_inputs.values()),
                                fresh_process_evidence='one round per input JSONL; process IDs are not independently verified',
                                method='median of three process medians and median of three process p90 values; '
                                       'spread=(max(process medians)-min(process medians))/median, rounded to 4 decimals'),
                rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--resources', nargs='+', type=Path, default=[])
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--card')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output already exists; preserve historical tables')
    try:
        document = aggregate(args.runs, args.resources, args.diagnostic, args.card)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(rows=len(document['rows']), qualified=document['qualified'], full_table=False)))


if __name__ == '__main__':
    main()
