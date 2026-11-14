#!/usr/bin/env python3
"""Reject corruptions of the completed fixed B16 prefetch admission evidence."""
import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'python')]
import dnn4_bert_repeat_prefetch as admission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    args = parser.parse_args()
    folder = args.folder
    original = admission.read
    def validate():
        return admission.validate(folder, admission.read(folder/'experiment.json'),
                                   admission.read(folder/'final_result.json'))
    validate()
    decision = original(folder/'experiment.json')
    integration = original(folder/'integration.json')
    rejected = []
    def reject(label, path, mutate):
        target = Path(path).resolve()
        value = copy.deepcopy(original(target))
        mutate(value)
        def read(path):
            return value if Path(path).resolve() == target else original(path)
        with patch.object(admission, 'read', read):
            try:
                validate()
            except ValueError:
                rejected.append(label)
                return
        raise AssertionError('corrupted receipt accepted: '+label)
    reject('wrong batch', folder/'experiment.json', lambda v: v.update(batch=4))
    reject('lowered minimum', folder/'experiment.json', lambda v: v.update(minimum_gain=0))
    reject('unaccepted pair', folder/'experiment.json', lambda v: v.update(accepted=False))
    reject('wrong final artifact', folder/'final_result.json', lambda v: v.update(library=decision['reference']))
    reject('missing gate', folder/'experiment.json', lambda v: v['unit_gates'].pop())
    reject('missing binding field', integration['binding_contract'], lambda v: v.update(checks=1379))
    reject('missing first output map', integration['binding_contract'], lambda v: v['optional_output_maps'].__setitem__(0, None))
    reject('missing main arm', integration['main_emission'], lambda v: v['rows'].pop())
    reject('changed generated source', integration['main_emission'], lambda v: v['rows'][1].update(generated_source_identical=False))
    reject('missing default profile', integration['defaults'], lambda v: v['rows'].pop())
    reject('changed default source', integration['defaults'], lambda v: v['rows'][0].update(default_explicit0_source_identical=False))
    reject('wrong reused body words', integration['body_reuse'], lambda v: v.update(encoded_words=0))
    reject('wrong reused body binary', integration['body_reuse'], lambda v: v.update(new_sha256='0'*64))
    reject('wrong map fixture words', integration['map_fixture'], lambda v: v.update(encoded_words=0))
    reject('wrong map binary', integration['map_fixture'], lambda v: v.update(prototype_sha256='0'*64))
    reject('missing sanitizer process', decision['unit_gates'][0], lambda v: v['processes'].pop())
    reject('reused ordering PID', decision['unit_gates'][2], lambda v: v['processes'][1].update(pid=v['processes'][0]['pid']))
    reject('wrong ordering binary', decision['unit_gates'][2], lambda v: v['processes'][0].update(binary_sha256='0'*64))
    reject('wrong ordering hint', integration['order_builds'][0], lambda v: v['profile'].update(TILEMEGA_F3_PREFETCH_DISTANCE='2'))
    reject('missing ordering hint flag', integration['order_builds'][0], lambda v: v['command'].remove('-DTILEMEGA_F3_PREFETCH_DISTANCE=1'))
    reject('changed ordering bytes', integration['old_order_gates'][0], lambda v: v.update(output_hash='0'*16))
    reject('missing model process', integration['model_cohort'], lambda v: v['processes'].pop())
    reject('changed model bytes', integration['model_bit_proof'], lambda v: v['hashes'].__setitem__(0, '0'*64))
    model = original(integration['model_cohort'])
    reject('relaxed FP32 cosine', model['processes'][0]['check'], lambda v: v['cases'][0]['outputs'][0].update(cosine_min=.998))
    reject('synthetic checkpoint', model['processes'][0]['check'], lambda v: v.update(synthetic_weights=True))
    reject('wrong selection winner', integration['selection'], lambda v: v.update(winner_library=decision['reference']))
    reject('missing selection candidate', integration['candidate_spec'], lambda v: v.pop())
    reject('duplicate coordinates', integration['candidate_spec'], lambda v: v[0].update(library=v[1]['library']))
    reject('changed selection order', integration['selection'], lambda v: v['measurements'][0].update(arm='B'))
    reject('wrong tie priority', integration['candidate_spec'], lambda v: v[0].update(tune_priority=3))
    reject('changed selection median', integration['selection'], lambda v: v['ranking'][0].update(median_ms=0))
    reject('changed measured gain', folder/'experiment.json', lambda v: v['pair'].update(ratio=.5))
    reject('changed measured noise', folder/'experiment.json', lambda v: v['pair'].update(threshold=0))
    reject('changed final median', folder/'final_result.json', lambda v: v['measurement'].update(candidate_ms=0))
    reject('changed final p90', folder/'final_result.json', lambda v: v['measurement'].update(p90_ms=0))
    reject('wrong allocation card', folder/'allocation.json', lambda v: v.update(gpu=1))
    reject('changed target card', folder/'allocation.json', lambda v: v.update(original_anchor_gpu=2))
    reject('unreported reanchor', folder/'allocation.json', lambda v: v.update(reanchored=True))
    allocation = original(folder/'allocation.json')
    reject('wrong permission snapshot', allocation['snapshot'], lambda v: v.update(allowed_gpu_indices=[0, 1, 2, 3]))
    reject('changed group allocation policy', allocation['snapshot'], lambda v: v.update(same_card_pair_required=True))
    reject('wrong ledger card', integration['ledger_receipt'], lambda v: v.update(gpu=0))
    reject('wrong ledger binary', integration['ledger_receipt'], lambda v: v.update(reference_sha256='0'*64))
    reject('wrong trace source', integration['trace_source_proof'], lambda v: v.update(reference=decision['reference']))
    reject('stale evidence digest', folder/'integration.json', lambda v: v['receipts'].__setitem__(next(iter(v['receipts'])), '0'*64))
    print(json.dumps(dict(evidence='verified CPU-only', positive_receipt=True,
                          corrupted_receipts_rejected=rejected)))


if __name__ == '__main__':
    main()
