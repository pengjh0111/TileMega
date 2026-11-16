#!/usr/bin/env python3
"""Audit a real frozen full-tile receipt and reject corrupted admission inputs."""
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
    parser.add_argument('--folder', required=True, type=Path)
    parser.add_argument('--batch', required=True, type=int, choices=[4, 8, 16])
    args = parser.parse_args()
    decision = current.read(args.folder/'experiment.json')
    final = current.read(args.folder/'final_result.json')
    integration = current.read(args.folder/'integration.json')
    current.validate_full_tiles_integration(args.folder, decision, final, args.batch)
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
                current.validate_full_tiles_integration(args.folder, decision, final, args.batch)
            except ValueError:
                failures.append(label)
                return
        raise AssertionError('corrupted receipt accepted: '+label)

    reject('missing component', args.folder/'integration.json',
           lambda p: p['unit_bit_proofs'].pop('order'))
    reject('changed model bytes', integration['model_bit_proof'],
           lambda p: p['hashes'].__setitem__(0, '0'*64))
    reject('wrong model predecessor', integration['model_bit_proof'],
           lambda p: p.update(reference=decision['gate']))
    reject('wrong ledger binary', integration['ledger_receipt'],
           lambda p: p.update(reference_sha256='0'*64))
    reject('wrong ledger GPU', integration['ledger_receipt'],
           lambda p: p.update(gpu=0))
    reject('wrong ordering lock', decision['unit_gates'][1],
           lambda p: p['processes'][0].update(gpu_lock='/tmp/another_gpu.lock'))
    reject('incomplete profile emission', current.read(integration['main_proof'])['codegen'],
           lambda p: p.update(invalid_settings_rejected=0))
    reject('stale integration receipt', args.folder/'integration.json',
           lambda p: p.update(main_proof_sha256='0'*64))
    if args.batch == 4:
        def change_profile(value):
            row = next(r for r in value['rows'] if r['batch'] == 4)
            row['profile']['TILEMEGA_F3_REGISTER_BUDGET'] = '3'
        reject('mismatched inherited fixture profile',
               current.read(integration['main_proof'])['on_proofs'], change_profile)
    print(json.dumps(dict(evidence='verified CPU-only',positive_receipt=True,
                         corrupted_receipts_rejected=failures)))


if __name__ == '__main__':
    main()
