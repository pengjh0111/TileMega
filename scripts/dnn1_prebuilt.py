#!/usr/bin/env python3
"""Certify a CPU-prebuilt attribution artifact before its guarded capture."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--build-receipt', type=Path, required=True)
    parser.add_argument('--batch', type=int, choices=(1, 16), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = next(p for p in Path(__file__).resolve().parents if (p / 'CLAUDE.md').is_file())
    sys.path.insert(0, str(root / 'python'))
    from tilemega.build.identity import verify
    rows = json.loads(args.build_receipt.read_text())
    matches = [row for row in rows if Path(row.get('library', '')).resolve() == args.library.resolve()]
    if len(matches) != 1:
        raise ValueError('prebuild receipt must identify exactly one artifact')
    row = matches[0]
    if row['exit_code'] != 0 or row['gpu_executed'] is not False:
        raise ValueError('successful CPU-only build evidence required')
    if (row['binary_sha256'] != sha(args.library) or
            row['identity_sha256'] != sha(str(args.library) + '.identity.json')):
        raise ValueError('prebuilt artifact changed after its build receipt')
    identity = verify(args.library)
    outcome_path = Path(str(args.library) + '.compile-result.json')
    outcome = json.loads(outcome_path.read_text())
    if (not identity['trace'] or identity['plan']['runtime_batch'] != args.batch or
            outcome['status'] != 'compiled' or outcome['inputs_unchanged'] is not True or
            outcome['generated_source_sha256'] != identity['generated_source_sha256'] or
            outcome['source_digest'] != identity['source']['source_digest']):
        raise ValueError('prebuilt attribution identity or bounded compilation evidence differs')
    result = dict(evidence='verified CPU artifact identity; no GPU or numerical claim',
        gpu_executed=False, batch=args.batch, artifact_id=identity['artifact_id'],
        library=str(args.library.resolve()), binary_sha256=sha(args.library),
        identity_sha256=sha(str(args.library) + '.identity.json'),
        source_head=identity['source']['head'], source_digest=identity['source']['source_digest'],
        compile_result_sha256=sha(outcome_path), build_receipt=str(args.build_receipt.resolve()),
        build_receipt_sha256=sha(args.build_receipt), verifier_sha256=sha(__file__))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as output:
        output.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
