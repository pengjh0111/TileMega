"""Reject corruptions of a complete fixed-distance BERT admission."""
import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'python')]
import dnn4_bert_prefetch as admission


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

    def reject(label, path, change):
        target = Path(path).resolve()
        value = copy.deepcopy(original(target))
        change(value)

        def read(path):
            return value if Path(path).resolve() == target else original(path)

        with patch.object(admission, 'read', read):
            try:
                validate()
            except ValueError:
                rejected.append(label)
                return
        raise AssertionError('corrupted receipt accepted: '+label)

    reject('wrong batch', folder/'experiment.json', lambda v: v.update(batch=16))
    reject('lowered minimum', folder/'experiment.json', lambda v: v.update(minimum_gain=0))
    reject('unaccepted pair', folder/'experiment.json', lambda v: v.update(accepted=False))
    reject('wrong final artifact', folder/'final_result.json', lambda v: v.update(library=decision['reference']))
    reject('missing component', folder/'experiment.json', lambda v: v['unit_gates'].pop())
    reject('wrong prepared artifact', integration['preparation'], lambda v: v['model'].update(library=decision['reference']))
    reject('changed prospective rule', integration['original_plan'], lambda v: v.update(minimum_gain=0))
    reject('wrong predecessor', folder/'experiment.json', lambda v: v.update(reference=decision['library']))
    reject('changed coordinates', integration['source_differential'], lambda v: v.update(coordinates_identical=False))
    reject('missing default cell', integration['defaults'], lambda v: v['rows'].pop())
    reject('empty default instruction proof', integration['defaults'], lambda v: v['rows'][0].update(encoded_words=0))
    reject('substituted default library', integration['defaults'], lambda v: v['rows'][2].update(new_library=decision['library']))
    reject('wrong reused body words', integration['body_proof'], lambda v: v.update(encoded_words=0))
    reject('wrong reused body hash', integration['body_proof'], lambda v: v.update(binary_sha256='0'*64))
    reject('wrong body reference hash', integration['body_proof'], lambda v: v.update(reference_sha256='0'*64))
    reject('wrong reused body cohort', integration['body_proof'], lambda v: v.update(reused_gate=decision['unit_gates'][1]))
    if integration.get('body_origin'):
        reject('wrong original GELU hash', integration['body_origin'], lambda v: v.update(reference_sha256='0'*64))
        reject('wrong original GELU words', integration['body_origin'], lambda v: v.update(encoded_words=0))
    reject('missing visibility build', integration['split_preparation'], lambda v: v['rows'].pop())
    reject('duplicate visibility case', integration['split_plan'], lambda v: v['cases'].__setitem__(3, v['cases'][0]))
    gate = decision['unit_gates'][1]
    reject('missing sanitizer child', gate, lambda v: v['processes'].pop())
    reject('reused child PID', gate, lambda v: v['processes'][1].update(pid=v['processes'][0]['pid']))
    reject('wrong sanitizer counts', gate, lambda v: v['processes'][6].update(name='unit-6'))
    reject('changed ordering bytes', gate, lambda v: v['processes'][0].update(output_hash='0'*16))
    reject('wrong ordering binary', gate, lambda v: v['processes'][0].update(binary_sha256='0'*64))
    reject('wrong ordering card', gate, lambda v: v.update(gpu=2))
    reject('wrong ordering hint', integration['order_builds'][0], lambda v: v['profile'].update(TILEMEGA_F3_PREFETCH_DISTANCE='2'))
    reject('wrong ordering registers', integration['order_builds'][0], lambda v: v['profile'].update(TILEMEGA_F3_REGISTER_BUDGET='3'))
    reject('missing hint flag', integration['order_builds'][0], lambda v: v['command'].remove('-DTILEMEGA_F3_PREFETCH_DISTANCE=1'))
    reject('spilling ordering build', integration['order_builds'][0], lambda v: v.update(spill_bytes=16))
    model = original(integration['model_cohort'])
    reject('missing model process', integration['model_cohort'], lambda v: v['processes'].pop())
    reject('wrong model identity', integration['model_cohort'], lambda v: v['execution'].update(artifact_id='0'*64))
    reject('reused model PID', integration['model_cohort'], lambda v: v['processes'][1].update(pid=v['processes'][0]['pid']))
    reject('relaxed FP32 cosine', model['processes'][3]['check'], lambda v: v['cases'][0]['outputs'][1].update(cosine_min=.99899))
    reject('changed per-process tensor', model['processes'][3]['check'], lambda v: v['cases'][0]['outputs'][1].update(output_sha256='0'*64))
    reject('wrong named pooler proof', integration['model_bit_proof'], lambda v: v['named_output_sha256'].update(tanh='0'*64))
    reject('unpassed byte proof', integration['model_bit_proof'], lambda v: v.update(passed=False))
    reject('wrong byte predecessor', integration['model_bit_proof'], lambda v: v.update(reference=decision['gate']))
    reject('wrong allocation card', folder/'allocation.json', lambda v: v.update(gpu=0))
    reject('changed original anchor', folder/'allocation.json', lambda v: v.update(original_anchor_gpu=1))
    reject('unreported reanchor', folder/'allocation.json', lambda v: v.update(reanchored=True))
    reject('wrong Python entry root', folder/'allocation.json', lambda v: v.update(python_entry_root='/elsewhere'))
    allocation = original(folder/'allocation.json')
    reject('wrong permission snapshot', allocation['snapshot'], lambda v: v.update(allowed_gpu_indices=[0, 2]))
    reject('timing before authorization', allocation['snapshot'], lambda v: v.update(current_gpu_restriction_at=1e18))
    reject('changed measured gain', folder/'experiment.json', lambda v: v['pair'].update(ratio=0))
    reject('changed measured noise', folder/'experiment.json', lambda v: v['pair'].update(threshold=0))
    sample = decision['pair']['measurements'][1]['measurement']
    reject('missing event sample', sample, lambda v: v['samples_ms'].pop())
    reject('wrong measured artifact', sample, lambda v: v.update(artifact_id='0'*64))
    reject('wrong measured numerical gate', sample, lambda v: v.update(numerical_gate_path=decision['reference_gate']))
    reject('changed weight provenance', sample, lambda v: v.update(random_weights=not v['random_weights']))
    reject('changed accuracy provenance', sample, lambda v: v.update(accuracy_eligible=not v['accuracy_eligible']))
    reject('wrong checkpoint bytes', sample, lambda v: v.update(source_checkpoint_sha256='0'*64))
    reject('wrong recorded clock', folder/'final_result.json', lambda v: v['measurement']['measurements'][0].update(start_sm_clock_mhz=0))
    reject('changed final median', folder/'final_result.json', lambda v: v['measurement'].update(candidate_ms=0))
    reject('changed final p90', folder/'final_result.json', lambda v: v['measurement'].update(p90_ms=0))
    reject('changed final range', folder/'final_result.json', lambda v: v['measurement'].update(process_range_ms=0))
    reject('wrong ledger card', integration['ledger_receipt'], lambda v: v.update(gpu=3))
    reject('wrong ledger binary', integration['ledger_receipt'], lambda v: v.update(reference_sha256='0'*64))
    reject('wrong trace predecessor', integration['trace_source_proof'], lambda v: v.update(reference=decision['reference']))
    reject('nontrace substitution', integration['trace_source_proof'], lambda v: v.update(library=decision['library']))
    reject('unpassed trace proof', integration['trace_source_proof'], lambda v: v.update(passed=False))
    reject('stale receipt digest', folder/'integration.json', lambda v: v['receipts'].__setitem__(next(iter(v['receipts'])), '0'*64))
    reject('missing receipt digest', folder/'integration.json', lambda v: v['receipts'].pop(integration['model_cohort']))
    print(json.dumps(dict(evidence='verified CPU-only', positive_receipt=True,
                          corrupted_receipts_rejected=rejected)))


if __name__ == '__main__':
    main()
