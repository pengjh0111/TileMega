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


def publish(source, destination):
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    receipt = dict(evidence='verified', input=str(source.resolve()), input_sha256=sha(source),
                   output=str(destination.resolve()), output_sha256=sha(destination),
                   publisher_sha256=sha(__file__), enriched_rows=audit,
                   scope='version metadata only; measurements and acceptance remain unchanged')
    destination.with_suffix(destination.suffix + '.audit.json').write_text(
        json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = publish(args.input, args.out)
    print(json.dumps(dict(output=result['output'], enriched_rows=len(result['enriched_rows']))))


if __name__ == '__main__':
    main()
