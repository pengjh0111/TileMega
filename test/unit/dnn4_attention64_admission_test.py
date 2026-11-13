#!/usr/bin/env python3
"""Reject corrupted fixed-attention evidence using frozen measured receipts."""
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, required=True)
    a = p.parse_args()
    decision = current.read(a.folder/'experiment.json')
    final = current.read(a.folder/'final_result.json')
    integration = current.read(a.folder/'integration.json')
    current.validate_attention64_integration(a.folder, decision, final)
    original = current.read
    failures = []
    def reject(label, path, alter):
        target = Path(path).resolve()
        value = copy.deepcopy(original(target))
        alter(value)
        def read(path):
            return value if Path(path).resolve() == target else original(path)
        with patch.object(current, 'read', read):
            try:
                current.validate_attention64_integration(a.folder, decision, final)
            except ValueError:
                failures.append(label)
                return
        raise AssertionError('corrupted receipt accepted: '+label)
    reject('missing attention fixture', integration['fixture_proof'], lambda v: v['rows'].pop())
    reject('changed model output bytes', integration['model_bit_proof'], lambda v: v['hashes'].__setitem__(0, '0'*64))
    reject('wrong model predecessor', integration['model_bit_proof'], lambda v: v.update(reference=decision['gate']))
    reject('wrong ledger binary', integration['ledger_receipt'], lambda v: v.update(reference_sha256='0'*64))
    reject('wrong ledger card', integration['ledger_receipt'], lambda v: v.update(gpu=1))
    reject('wrong component lock', decision['unit_gates'][0], lambda v: v['processes'][0].update(gpu_lock='/tmp/other_gpu.lock'))
    reject('missing sanitizer process', decision['unit_gates'][0], lambda v: v['processes'].pop())
    reject('wrong query block', integration['codegen_proof'], lambda v: v['profile'].update(TILEMEGA_F3_ATTENTION_ROWS='128'))
    reject('disabled warpgroup math', integration['codegen_proof'], lambda v: v['profile'].update(TILEMEGA_F3_WGMMA_QK='0'))
    reject('stale codegen proof', a.folder/'integration.json', lambda v: v.update(codegen_proof_sha256='0'*64))
    reject('stale fixture proof', a.folder/'integration.json', lambda v: v.update(fixture_proof_sha256='0'*64))
    bit = original(integration['unit_bit_proof'])
    reject('changed control bytes', bit['control'], lambda v: v.update(output_hash='0'*16))
    reject('missing control process', bit['control'], lambda v: v['processes'].pop())
    reject('incomplete defaults', integration['defaults'], lambda v: v['rows'].pop())
    reject('missing committed arm', integration['main_fixture'], lambda v: v['rows'].pop())
    reject('wrong emitted binary', integration['emission'], lambda v: v['rows'][-1].update(prototype=decision['reference']))
    reject('wrong model cohort', integration['model_cohort'], lambda v: v.update(gate=decision['reference_gate']))
    reject('missing query map', integration['contract'], lambda v: v['rows'][1]['query_map_indices'].pop())
    reject('wrong layout coverage', integration['layout'], lambda v: v.update(coordinates=0))
    reject('wrong fixture scope', integration['scope_qualification'], lambda v: v.update(cohort_suite='ordering'))
    allocation = a.folder/'allocation.json'
    reject('wrong allocation card', allocation, lambda v: v.update(gpu=3))
    reject('unreported reanchor', allocation, lambda v: v.update(reanchored=True))
    reject('missing authorization', original(allocation)['authorization'], lambda v: v.update(allowed_gpu_indices=[3]))
    authorization = Path(original(allocation)['authorization'])
    active = original(authorization)
    if active['allowed_gpu_indices'] == [0]:
        transition = authorization.parent/'gpu0_transition_1794589079598058614'
        reject('wrong historical permission', transition/'authorization_before.json', lambda v: v.update(allowed_gpu_indices=[0]))
        reject('wrong historical request', transition/'authorization_before.json', lambda v: v.update(request='different request'))
        reject('wrong historical resumption', transition/'authorization_before.json', lambda v: v.update(resumed_at=0))
        reject('wrong transition card', transition/'result.json', lambda v: v.update(allowed_gpu_indices=[3]))
        reject('unknown new allocation', authorization, lambda v: v.update(gpu_restriction_request='unknown'))
        reject('timing after allocation change', authorization, lambda v: v.update(gpu_restriction_at=0))
    print(json.dumps(dict(evidence='verified CPU-only', positive_receipt=True, corrupted_receipts_rejected=failures)))

if __name__ == '__main__':
    main()
