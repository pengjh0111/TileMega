#!/usr/bin/env python3
"""Reject incomplete or substituted FFN1-only admission evidence."""
import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
import dnn4_bert_current as current
import dnn4_bert_ffn1_isolation as isolation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    folder = parser.parse_args().folder
    original = isolation.read
    decision = original(folder/'experiment.json')
    final = original(folder/'final_result.json')
    integration = original(folder/'integration.json')
    isolation.validate(folder, decision, final)
    rejected = []

    def reject(label, path, alter):
        target = Path(path).resolve()
        value = copy.deepcopy(original(target))
        alter(value)

        def read(path):
            return value if Path(path).resolve() == target else original(path)

        with patch.object(isolation, 'read', read), patch.object(current, 'read', read):
            try:
                isolation.validate(folder, read(folder/'experiment.json'),
                                   read(folder/'final_result.json'))
            except ValueError:
                rejected.append(label)
                return
        raise AssertionError('corrupted receipt accepted: '+label)

    reject('unaccepted comparison', folder/'experiment.json', lambda v: v.update(accepted=False))
    reject('lowered gain gate', folder/'experiment.json', lambda v: v.update(minimum_gain=0))
    reject('different final binary', folder/'final_result.json', lambda v: v.update(library=decision['reference']))
    reject('different final gate', folder/'final_result.json', lambda v: v.update(gate=decision['reference_gate']))
    reject('changed pair ratio', folder/'experiment.json', lambda v: v['pair'].update(ratio=0))
    reject('changed pair noise', folder/'experiment.json', lambda v: v['pair'].update(threshold=0))
    reject('missing component gate', folder/'experiment.json', lambda v: v['unit_gates'].pop())
    reject('changed original gain', integration['original_plan'], lambda v: v.update(minimum_gain=0))
    reject('wrong register budget', integration['preparation'], lambda v: v['profile'].update(TILEMEGA_F3_REGISTER_BUDGET='2'))
    reject('wrong defaults', integration['defaults'], lambda v: v.update(encoded_words=0))
    reject('unpassed defaults', integration['default_proof'], lambda v: v.update(passed=False))
    reject('missing component', integration['components'], lambda v: v['rows'].pop())
    reject('wrong reuse instructions', integration['components'], lambda v: v['rows'][0].update(encoded_words=1))
    reject('wrong reused binary', integration['components'], lambda v: v['rows'][0].update(reference_sha256='0'*64))
    reject('wrong fixture profile', integration['components'], lambda v: v['rows'][0]['profile'].update(TILEMEGA_F3_REGISTER_BUDGET='2'))
    reject('changed fixture selector', integration['components'], lambda v: v['rows'][0]['fixture_defines'].pop('TILEMEGA_F3_FFN2_STATS96'))
    reject('wrong component headers', integration['components'], lambda v: v['rows'][0]['command'].remove('-I/root/TileMega-dnn3/include'))
    for index, path in enumerate(decision['unit_gates']):
        reject(f'incomplete historical cohort {index}', path, lambda v: v['processes'].pop())
    model = original(integration['model_cohort'])
    reject('missing model process', integration['model_cohort'], lambda v: v['processes'].pop())
    reject('wrong model artifact', integration['model_cohort'], lambda v: v['execution'].update(artifact_id='0'*64))
    reject('wrong model card', integration['model_cohort'], lambda v: v.update(gpu=2))
    reject('model byte nondeterminism', model['processes'][0]['check'], lambda v: v['cases'][0]['outputs'][1].update(output_sha256='0'*64))
    reject('model precision failure', model['processes'][0]['check'], lambda v: v['cases'][0]['outputs'][0].update(cosine_min=.998))
    allocation = original(folder/'allocation.json')
    reject('wrong allocation', folder/'allocation.json', lambda v: v.update(gpu=0))
    reject('changed fixed anchor', folder/'allocation.json', lambda v: v.update(original_anchor_gpu=1))
    reject('unreported reanchor', folder/'allocation.json', lambda v: v.update(reanchored=True))
    reject('wrong Python root', folder/'allocation.json', lambda v: v.update(python_entry_root='/elsewhere'))
    reject('wrong historical permission', allocation['snapshot'], lambda v: v.update(allowed_gpu_indices=[2]))
    reject('timing before permission', allocation['snapshot'], lambda v: v.update(current_gpu_restriction_at=1e18))
    sample = decision['pair']['measurements'][1]['measurement']
    reject('wrong timing artifact', sample, lambda v: v.update(artifact_id='0'*64))
    reject('wrong timing gate', sample, lambda v: v.update(numerical_gate_path=decision['reference_gate']))
    reject('missing event sample', sample, lambda v: v['samples_ms'].pop())
    reject('changed checkpoint', sample, lambda v: v.update(source_checkpoint_sha256='0'*64))
    reject('changed timing provenance', sample, lambda v: v.update(random_weights=not v['random_weights']))
    reject('changed final median', folder/'final_result.json', lambda v: v['measurement'].update(candidate_ms=0))
    reject('changed final p90', folder/'final_result.json', lambda v: v['measurement'].update(p90_ms=0))
    reject('changed final range', folder/'final_result.json', lambda v: v['measurement'].update(process_range_ms=0))
    reject('wrong final clock', folder/'final_result.json', lambda v: v['measurement']['measurements'][0].update(start_sm_clock_mhz=0))
    reject('wrong ledger card', integration['ledger_receipt'], lambda v: v.update(gpu=3))
    reject('wrong ledger binary', integration['ledger_receipt'], lambda v: v.update(reference_sha256='0'*64))
    reject('wrong trace source', integration['trace_source_proof'], lambda v: v.update(library=decision['library']))
    reject('FFN2 coordinate change', decision['library']+'.plan.json', lambda v: v['native_forward']['stages'][5].update(split_k=2))
    reject('missing sealed receipt', folder/'integration.json', lambda v: v['receipts'].pop(integration['components']))
    reject('stale sealed receipt', folder/'integration.json', lambda v: v['receipts'].__setitem__(integration['components'], '0'*64))
    print(json.dumps(dict(evidence='verified CPU-only', positive_receipt=True,
                          corrupted_receipts_rejected=rejected)))


if __name__ == '__main__':
    main()
