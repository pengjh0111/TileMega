#!/usr/bin/env python3
"""Reject corrupted PF1 receipts without executing GPU work."""
import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
import dnn4_bert_prefetch_current as validator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    folder = parser.parse_args().folder
    read = validator.read
    dp, fp = folder / 'experiment.json', folder / 'final_result.json'
    decision, final = read(dp), read(fp)
    integration = read(folder / 'integration.json')
    validator.validate(folder, decision, final)
    failures = []

    def reject(label, *changes):
        altered = {}
        for path, change in changes:
            path = Path(path).resolve()
            altered.setdefault(path, copy.deepcopy(read(path)))
            change(altered[path])
        def mocked(path):
            return altered.get(Path(path).resolve(), read(path))
        with patch.object(validator, 'read', mocked):
            try:
                validator.validate(folder, mocked(dp), mocked(fp))
            except ValueError:
                failures.append(label)
                return
        raise AssertionError('corrupted receipt accepted: ' + label)

    def case(label, path, change):
        reject(label, (path, change))

    case('missing frozen receipts', folder / 'integration.json', lambda v: v['receipts'].clear())
    case('changed receipt digest', folder / 'integration.json',
         lambda v: v['receipts'].update({integration['preparation']: '0' * 64}))
    case('different predecessor', folder / 'integration.json', lambda v: v.update(parent_final=str(fp)))
    case('missing driver sources', integration['driver_sources'], lambda v: v.clear())
    case('wrong batch', dp, lambda v: v.update(batch=16))
    case('unaccepted comparison', dp, lambda v: v.update(accepted=False))
    case('lowered one-percent floor', dp, lambda v: v.update(minimum_gain=.005))
    case('different reference', dp, lambda v: v.update(reference=v['library']))
    case('different final library', fp, lambda v: v.update(library=decision['reference']))
    case('different final gate', fp, lambda v: v.update(gate=decision['reference_gate']))
    case('missing component gate', fp, lambda v: v['unit_gates'].pop())
    plan = integration['original_plan']
    case('changed original gain floor', plan, lambda v: v.update(minimum_gain=0))
    case('changed original reference', plan, lambda v: v['reference'].update(library=decision['library']))
    case('different prepared profile', integration['preparation'],
         lambda v: v['profile'].update(TILEMEGA_F3_REGISTER_BUDGET='2'))
    case('unprepared model', integration['preparation'], lambda v: v.update(status='failed'))
    case('different prepared artifact', integration['preparation'], lambda v: v['model'].update(artifact_id='0' * 64))
    case('lost source differential', integration['source_differential'], lambda v: v.update(passed=False))
    case('changed coordinates proof', integration['source_differential'], lambda v: v.update(coordinates_identical=False))
    proof = integration['body_map_identity']
    case('lost complete word proof', proof, lambda v: v.update(passed=False))
    case('missing inherited map', proof, lambda v: v['rows'].pop())
    case('different body word count', proof, lambda v: v['rows'][0].update(encoded_words=1))
    case('different executed body digest', proof, lambda v: v['rows'][0].update(reference_sha256='0' * 64))
    case('different candidate map digest', proof, lambda v: v['rows'][1].update(binary_sha256='0' * 64))
    case('wrong inherited gate', dp, lambda v: v['unit_gates'].__setitem__(0, v['unit_gates'][2]))
    case('missing order build', folder / 'integration.json', lambda v: v['order_builds'].pop())
    order = decision['unit_gates'][2]
    case('missing sanitizer process', order, lambda v: v['processes'].pop())
    case('duplicate order PID', order, lambda v: v['processes'][1].update(pid=v['processes'][0]['pid']))
    case('changed order output', order, lambda v: v['processes'][0].update(output_hash='0' * 16))
    case('changed order binary', order, lambda v: v['processes'][0].update(binary_sha256='0' * 64))
    case('changed order lock', order, lambda v: v['processes'][0].update(gpu_lock='/tmp/wrong.lock'))
    case('failed sanitizer', order, lambda v: v['processes'][0].update(exit_code=1))
    build = integration['order_builds'][0]
    case('wrong order profile', build, lambda v: v['profile'].update(TILEMEGA_F3_PREFETCH_DISTANCE='2'))
    case('spilled order kernel', build, lambda v: v.update(spill_bytes=8))
    case('missing actual PF1 flag', build,
         lambda v: v['command'].remove('-DTILEMEGA_F3_PREFETCH_DISTANCE=1'))
    model = integration['model_cohort']
    case('missing model process', model, lambda v: v['processes'].pop())
    case('duplicate model PID', model, lambda v: v['processes'][1].update(pid=v['processes'][0]['pid']))
    case('failed model', model, lambda v: v.update(passes='4/5'))
    case('changed model identity', model, lambda v: v['execution'].update(artifact_id='0' * 64))
    case('lost named outputs', integration['model_bit_proof'], lambda v: v.update(passed=False))
    case('wrong model predecessor', integration['model_bit_proof'], lambda v: v.update(reference=decision['gate']))
    case('unreported reanchor', folder / 'allocation.json', lambda v: v.update(reanchored=True))
    case('different actual allocation', folder / 'allocation.json', lambda v: v.update(gpu=3))
    case('different pair allocation', folder / 'allocation.json', lambda v: v.update(pair_gpu=3))
    case('wrong smoke card', dp, lambda v: v['smoke'].update(gpu=3))
    case('missing preferred card', integration['routing_preference'], lambda v: v.update(preferred_gpu_indices=[3]))
    case('timing before preference', integration['routing_preference'], lambda v: v.update(effective_at=1e20))
    timing = decision['pair']['measurements'][0]['measurement']
    case('wrong timing checkpoint', timing, lambda v: v.update(source_checkpoint_sha256='0' * 64))
    case('wrong timing inputs', timing, lambda v: v.update(input_sha256='0' * 64))
    case('wrong timing artifact', timing, lambda v: v.update(artifact_id='0' * 64))
    case('wrong timing numerical gate', timing, lambda v: v.update(numerical_gate_path=decision['gate']))
    for field, value, label in (('ratio', .5, 'fabricated pair ratio'), ('threshold', .005, 'lowered pair noise')):
        change = lambda v, k=field, x=value: v['pair'].update({k: x})
        reject(label, (dp, change), (fp, change))
    case('fabricated final median', fp, lambda v: v['measurement'].update(candidate_ms=.5))
    case('fabricated final p90', fp, lambda v: v['measurement'].update(p90_ms=.5))
    case('fabricated final range', fp, lambda v: v['measurement'].update(process_range_ms=0))
    ledger = integration['ledger_receipt']
    case('wrong ledger binary', ledger, lambda v: v.update(reference_sha256='0' * 64))
    case('wrong ledger GPU', ledger, lambda v: v.update(gpu=3))
    case('boundary ledger substituted', ledger, lambda v: v.update(trace_level=3))
    case('wrong trace predecessor', integration['trace_source_proof'], lambda v: v.update(reference=decision['reference']))
    print(json.dumps(dict(positive_receipt=True, gpu_executed=False,
        corrupted_receipts_rejected=failures)), flush=True)


if __name__ == '__main__':
    main()
