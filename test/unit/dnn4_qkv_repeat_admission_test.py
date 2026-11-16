#!/usr/bin/env python3
"""Reject corrupted repeated-QKV TMA receipts without executing GPU work."""
import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
import dnn4_bert_qkv_repeat as validator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    folder = parser.parse_args().folder
    read = validator.read
    decision_path, final_path = folder / 'experiment.json', folder / 'final_result.json'
    decision, final = read(decision_path), read(final_path)
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
                validator.validate(folder, mocked(decision_path), mocked(final_path))
            except ValueError:
                failures.append(label)
                return
        raise AssertionError('corrupted receipt accepted: ' + label)

    def case(label, path, change):
        reject(label, (path, change))

    case('missing frozen receipts', folder / 'integration.json', lambda v: v['receipts'].clear())
    case('changed receipt digest', folder / 'integration.json',
         lambda v: v['receipts'].update({integration['units']: '0' * 64}))
    case('wrong batch', decision_path, lambda v: v.update(batch=16))
    case('unaccepted comparison', decision_path, lambda v: v.update(accepted=False))
    case('lowered gain floor', decision_path, lambda v: v.update(minimum_gain=0))
    case('different final library', final_path, lambda v: v.update(library=decision['reference']))
    case('different final gate', final_path, lambda v: v.update(gate=decision['reference_gate']))
    case('missing component gate', final_path, lambda v: v['unit_gates'].pop())
    plan = integration['original_plan']
    case('changed original gain floor', plan, lambda v: v.update(minimum_gain=0))
    case('changed original reference', plan, lambda v: v['reference'].update(library=decision['library']))
    case('different preparation profile', plan, lambda v: v['profile'].update(TILEMEGA_F3_REGISTER_BUDGET='2'))
    case('unprepared model', integration['preparation'], lambda v: v.update(status='failed'))
    case('different prepared artifact', integration['preparation'], lambda v: v['model'].update(artifact_id='0' * 64))
    case('wrong binding count', integration['binding_contract'], lambda v: v.update(checks=0))
    case('first optional map present', integration['binding_contract'], lambda v: v['optional_output_maps'].__setitem__(0, 1))
    case('repeated optional map absent', integration['binding_contract'], lambda v: v['optional_output_maps'].__setitem__(1, None))
    case('missing optional map', integration['binding_contract'], lambda v: v['optional_output_maps'].pop())
    case('missing ordering reuse', integration['units'], lambda v: v.update(ordering_all_reusable=False))
    case('missing unit build', integration['units'], lambda v: v['rows'].pop())
    case('missing committed fixture', integration['fixtures'], lambda v: v['rows'].pop())
    case('changed committed digest', integration['fixtures'], lambda v: v['rows'][1].update(committed_build_sha256='0'*64))
    case('changed executed digest', integration['fixtures'], lambda v: v['rows'][1].update(executed_build_sha256='0'*64))
    case('lost old B16 parser proof', integration['fixtures'], lambda v: v.update(old_b16_words=0))
    body = decision['unit_gates'][0]
    case('missing sanitizer process', body, lambda v: v['processes'].pop())
    case('duplicate body PID', body, lambda v: v['processes'][1].update(pid=v['processes'][0]['pid']))
    case('changed body output', body, lambda v: v['processes'][0].update(output_hash='0' * 16))
    case('changed body binary', body, lambda v: v['processes'][0].update(binary_sha256='0' * 64))
    case('changed body lock', body, lambda v: v['processes'][0].update(gpu_lock='/tmp/wrong.lock'))
    case('failed sanitizer', body, lambda v: v['processes'][0].update(exit_code=1))
    units = read(integration['units'])
    case('wrong body profile', units['rows'][1]['build'], lambda v: v['profile'].update(TILEMEGA_F3_WGMMA_QK='0'))
    case('spilled body', units['rows'][1]['build'], lambda v: v.update(spill_bytes=8))
    case('missing historical identity', integration['units'], lambda v: v['rows'][2].update(passed=False))
    case('wrong executed ordering ELF', integration['units'], lambda v: v['rows'][2].update(executed_sha256='0'*64))
    case('wrong historical gate', integration['units'], lambda v: v['rows'][2].update(reference_gate=decision['unit_gates'][0]))
    mapped = decision['unit_gates'][1]
    case('missing actual map process', mapped, lambda v: v['processes'].pop())
    case('different map output', mapped, lambda v: v['processes'][0].update(output_hash='0'*16))
    case('different map binary', mapped, lambda v: v['processes'][0].update(binary_sha256='0'*64))
    case('missing control process', integration['control'], lambda v: v['processes'].pop())
    case('changed control binary', integration['control'], lambda v: v['processes'][0].update(binary_sha256='0' * 64))
    model = integration['model_cohort']
    case('missing model process', model, lambda v: v['processes'].pop())
    case('duplicate model PID', model, lambda v: v['processes'][1].update(pid=v['processes'][0]['pid']))
    case('failed model', model, lambda v: v.update(passes='4/5'))
    case('changed model identity', model, lambda v: v['execution'].update(artifact_id='0' * 64))
    case('lost named outputs', integration['model_bit_proof'], lambda v: v.update(bitwise_equal=False))
    case('wrong model predecessor', integration['model_bit_proof'], lambda v: v.update(reference=decision['gate']))
    case('unreported reanchor', folder / 'allocation.json', lambda v: v.update(reanchored=True))
    case('different actual allocation', folder / 'allocation.json', lambda v: v.update(gpu=3))
    case('missing authorized card', integration['routing_preference'], lambda v: v.update(preferred_gpu_indices=[3]))
    case('timing before preference', integration['routing_preference'], lambda v: v.update(effective_at=1e20))
    timing = pair_raw = decision['pair']['measurements'][0]['measurement']
    case('wrong timing checkpoint', timing, lambda v: v.update(source_checkpoint_sha256='0' * 64))
    case('wrong timing inputs', timing, lambda v: v.update(input_sha256='0' * 64))
    case('wrong timing artifact', timing, lambda v: v.update(artifact_id='0' * 64))
    case('wrong timing numerical gate', timing, lambda v: v.update(numerical_gate_path=decision['gate']))
    change = lambda v: v['pair'].update(ratio=.5)
    reject('fabricated pair ratio', (decision_path, change), (final_path, change))
    change = lambda v: v['pair'].update(threshold=.005)
    reject('lowered pair noise', (decision_path, change), (final_path, change))
    case('fabricated final median', final_path, lambda v: v['measurement'].update(candidate_ms=.5))
    case('fabricated final p90', final_path, lambda v: v['measurement'].update(p90_ms=.5))
    case('fabricated final range', final_path, lambda v: v['measurement'].update(process_range_ms=0))
    case('wrong ledger binary', integration['ledger_receipt'], lambda v: v.update(reference_sha256='0' * 64))
    case('wrong ledger GPU', integration['ledger_receipt'], lambda v: v.update(gpu=3))
    case('boundary ledger substituted', integration['ledger_receipt'], lambda v: v.update(trace_level=3))
    case('wrong trace predecessor', integration['trace_source_proof'], lambda v: v.update(reference=decision['reference']))
    print(json.dumps(dict(positive_receipt=True, gpu_executed=False,
        corrupted_receipts_rejected=failures)), flush=True)


if __name__ == '__main__':
    main()
