#!/usr/bin/env python3
"""Publish source-build TVM versions from accepted baseline receipts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_versions(receipt):
    diag = receipt['configuration']['tvm_diag']
    identity = diag['cache_identity']
    source = identity['tvm_source_identity']
    if diag['tvm_version'] != identity['tvm_version'] or diag['tvm_commit'] != identity['tvm_commit']:
        raise ValueError('TVM diagnostics and compiled identity disagree')
    return {
        'tvm': identity['tvm_version'],
        'tvm_source_commit': identity['tvm_commit'],
        'apache-tvm-ffi': identity['dependency_versions']['apache-tvm-ffi'],
        'tvm_ffi_source_commit': source['ffi_commit'],
        'tvm_source_diff_sha256': source['source_diff_sha256'],
        'tvm_recipe_schema': identity['recipe']['schema'],
    }


def publish(source, destination, base=None):
    if source.resolve() == destination.resolve():
        raise ValueError('retain the original collector output; use another destination')
    with source.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    audit = []
    for index, row in enumerate(rows):
        if row['system'] != 'tvm' or row['status'] != 'pass':
            continue
        session = Path(row['cell_session'])
        guard = json.loads((session.parent.parent / 'guard_result.json').read_text())
        if guard.get('code') != 0 or json.loads(session.read_text()).get('status') != 'complete':
            raise ValueError('accepted publication requires the original successful cell guard')
        paths = [Path(value) for value in json.loads(row['source'])]
        if len(paths) != 5:
            raise ValueError('an accepted DNN1 TVM arm must have five independent receipts')
        identities = []
        for path in paths:
            receipt = json.loads(path.read_text())
            if (receipt['system'] != 'tvm' or receipt['model'] != row['model']
                    or receipt['batch'] != int(row['batch'])
                    or receipt['precision'] != row['precision']
                    or receipt['correctness_passed'] is not True):
                raise ValueError('published row and source receipt disagree: ' + str(path))
            identities.append(source_versions(receipt))
        if any(value != identities[0] for value in identities[1:]):
            raise ValueError('source-build TVM versions differ between fresh processes')
        versions = json.loads(row['versions'])
        for key, value in identities[0].items():
            if key in versions and versions[key] != value:
                raise ValueError('refuse to overwrite a conflicting version: ' + key)
        versions.update(identities[0])
        row['versions'] = json.dumps(versions, sort_keys=True, separators=(',', ':'))
        audit.append(dict(row=index, model=row['model'], batch=int(row['batch']),
                          precision=row['precision'], added_versions=identities[0],
                          source_sha256={str(path): sha(path) for path in paths}))
    merge_audit = None
    if base is not None:
        if base.resolve() == destination.resolve():
            raise ValueError('publish to an independent destination; retain the supplied base CSV')
        with base.open(newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != fields:
                raise ValueError('baseline CSV schemas differ; do not silently drop primary fields')
            existing = list(reader)
        def key(row):
            return tuple(row[name] for name in ('model', 'batch', 'system', 'arm', 'precision'))
        incoming = {key(row): row for row in rows if row['system'] == 'tvm' and row['status'] == 'pass'}
        if len(incoming) != sum(row['system'] == 'tvm' and row['status'] == 'pass' for row in rows):
            raise ValueError('duplicate accepted TVM keys in source CSV')
        bf16_cells = {(row['model'], row['batch']) for row in incoming.values()
                      if row['precision'] == 'bf16'}
        merged, removed, replaced = [], [], []
        for row in existing:
            identity = key(row)
            if (row['system'] == 'tvm' and row['precision'] == 'fp16'
                    and (row['model'], row['batch']) in bf16_cells):
                removed.append(row)
            elif identity in incoming:
                replaced.append(row)
                merged.append(incoming.pop(identity))
            else:
                merged.append(row)
        merged.extend(incoming.values())
        primary_before = [row for row in existing if row['system'] != 'tvm']
        primary_after = [row for row in merged if row['system'] != 'tvm']
        if primary_before != primary_after:
            raise ValueError('publishing TVM versions must not alter primary rows')
        merge_audit = dict(base=str(base.resolve()), base_sha256=sha(base),
                           primary_rows_unchanged=len(primary_before),
                           replaced_rows=replaced, removed_historical_fp16_rows=removed,
                           accepted_keys=[list(key(row)) for row in rows
                                          if row['system'] == 'tvm' and row['status'] == 'pass'])
        rows = merged
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    receipt = dict(evidence='verified', input=str(source.resolve()), input_sha256=sha(source),
                   output=str(destination.resolve()), output_sha256=sha(destination),
                   publisher_sha256=sha(__file__), enriched_rows=audit, merge=merge_audit,
                   scope='version metadata only; measurements and acceptance remain unchanged')
    destination.with_suffix(destination.suffix + '.audit.json').write_text(
        json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--base', type=Path, help='merge accepted TVM rows into an independent copy of this CSV')
    args = parser.parse_args()
    result = publish(args.input, args.out, args.base)
    print(json.dumps(dict(output=result['output'], enriched_rows=len(result['enriched_rows']))))


if __name__ == '__main__':
    main()
