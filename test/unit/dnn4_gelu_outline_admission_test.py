#!/usr/bin/env python3
"""Reject corrupted outlined GELU evidence using its frozen real receipts."""
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
    current.validate_gelu_outline_integration(args.folder, decision, final)
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
                current.validate_gelu_outline_integration(args.folder, decision, final)
            except ValueError:
                failures.append(label)
                return
        raise AssertionError('corrupted receipt accepted: '+label)

    reject('missing body fixture', integration['fixture_proof'],
           lambda p: p['rows'].pop())
    reject('changed model bytes', integration['model_bit_proof'],
           lambda p: p['hashes'].__setitem__(0, '0'*64))
    reject('wrong predecessor', integration['model_bit_proof'],
           lambda p: p.update(reference=decision['gate']))
    reject('wrong ledger binary', integration['ledger_receipt'],
           lambda p: p.update(reference_sha256='0'*64))
    reject('wrong ledger card', integration['ledger_receipt'],
           lambda p: p.update(gpu=0))
    reject('wrong body lock', decision['unit_gates'][0],
           lambda p: p['processes'][0].update(gpu_lock='/tmp/another_gpu.lock'))
    reject('missing sanitizer process', decision['unit_gates'][0],
           lambda p: p['processes'].pop())
    reject('changed feature profile', integration['codegen_proof'],
           lambda p: p['profile'].update(TILEMEGA_F3_GELU_OUTLINE='0'))
    reject('stale codegen receipt', args.folder/'integration.json',
           lambda p: p.update(codegen_proof_sha256='0'*64))
    reject('stale fixture receipt', args.folder/'integration.json',
           lambda p: p.update(fixture_proof_sha256='0'*64))
    bit = original(integration['unit_bit_proof'])
    reject('changed control bytes', bit['control'],
           lambda p: p.update(output_hash='0'*16))
    reject('missing control process', bit['control'],
           lambda p: p['processes'].pop())
    reject('wrong oracle arithmetic', integration['oracle_repair'],
           lambda p: p.update(original_scalar_mismatches=1))
    reject('incomplete default profiles', integration['defaults'],
           lambda p: p['rows'].pop())
    reject('missing committed fixture arm', integration['main_fixture'],
           lambda p: p['rows'].pop())
    reject('wrong emitted binary', integration['emission'],
           lambda p: p['rows'][-1].update(prototype=decision['reference']))
    if 'model_reuse' in integration:
        reject('wrong model reuse', integration['model_reuse'],
               lambda p: p.update(proof=integration['unit_bit_proof']))
    else:
        reject('wrong model cohort', integration['model_cohort'],
               lambda p: p.update(gate=decision['reference_gate']))
    print(json.dumps(dict(evidence='verified CPU-only', positive_receipt=True,
                         corrupted_receipts_rejected=failures)))


if __name__ == '__main__':
    main()
