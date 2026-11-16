#!/usr/bin/env python3
"""Reject corrupted store-bundle evidence using its frozen real receipts."""
import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
import dnn4_bert_current as current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    args = parser.parse_args()
    decision = current.read(args.folder/'experiment.json')
    final = current.read(args.folder/'final_result.json')
    integration = current.read(args.folder/'integration.json')
    current.validate_store_bundle_integration(args.folder, decision, final)
    original = current.read
    failures = []

    def reject(label, filename, alter):
        target = Path(filename).resolve()
        value = copy.deepcopy(original(target))
        alter(value)
        def read(path):
            return value if Path(path).resolve() == target else original(path)
        with patch.object(current, 'read', read):
            try:
                current.validate_store_bundle_integration(args.folder, decision, final)
            except ValueError:
                failures.append(label)
                return
        raise AssertionError('corrupted receipt accepted: '+label)

    reject('missing attention fixture', integration['fixture_proof'],
           lambda p: p['rows'].pop())
    reject('changed model bytes', integration['model_bit_proof'],
           lambda p: p['hashes'].__setitem__(0, '0'*64))
    reject('wrong predecessor', integration['model_bit_proof'],
           lambda p: p.update(reference=decision['gate']))
    reject('wrong ledger binary', integration['ledger_receipt'],
           lambda p: p.update(reference_sha256='0'*64))
    reject('wrong ledger card', integration['ledger_receipt'],
           lambda p: p.update(gpu=0))
    reject('wrong ordering lock', decision['unit_gates'][1],
           lambda p: p['processes'][0].update(gpu_lock='/tmp/another_gpu.lock'))
    reject('missing sanitizer process', decision['unit_gates'][2],
           lambda p: p['processes'].pop())
    reject('changed feature profile', integration['codegen_proof'],
           lambda p: p['profile'].update(TILEMEGA_F3_MATRIX_STORE='0'))
    reject('stale codegen receipt', args.folder/'integration.json',
           lambda p: p.update(codegen_proof_sha256='0'*64))
    reject('stale fixture receipt', args.folder/'integration.json',
           lambda p: p.update(fixture_proof_sha256='0'*64))
    print(json.dumps(dict(evidence='verified CPU-only', positive_receipt=True,
                         corrupted_receipts_rejected=failures)))


if __name__ == '__main__':
    main()
