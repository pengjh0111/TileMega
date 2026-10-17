#!/usr/bin/env python3
"""Validate a completed native numeric job before its dependent timing jobs."""
import argparse
import json
from pathlib import Path

from collect_r16_candidates import load_driver, read, read_states
from r16_measurement_validation import (NATIVE_LLM_SYSTEMS, completed_guard, file_sha,
                                         validate_native_repetitions)
from summarize_r16 import record_directory
from r16_runtime_contract import validate_receipt as validate_runtime_receipt


def validate_payload(path, candidate, pair, stage, expected_uuid=None):
    path = Path(path)
    value = read(path)
    directory = path.parent.parent
    validate_runtime_receipt(candidate,value,directory.parent)
    validate_native_repetitions(path, value)
    label = candidate.get('native_system_label', candidate['system'])
    if label not in NATIVE_LLM_SYSTEMS or value['system'] != label:
        raise ValueError('numeric receipt is not the declared native system')
    if value.get('evidence') != 'verified' or value.get('valid') is False or (path.parent/'invalid_measurements.json').exists():
        raise ValueError('numeric measurement is invalid')
    count = 16 if stage == 'smoke' else 1024
    batch = candidate['batch']
    if (value['batch'] != batch or value['max_tokens'] != count or
            value['prompt_length'] != 64 or value['gpu_index'] != candidate['gpu'] or
            value['model_config_sha256'] != candidate['config_sha256'] or
            value['prompt_ids_sha256'] != file_sha(candidate['prompt_ids'])):
        raise ValueError('numeric workload differs from frozen candidate')
    rows = value['generation_runs']
    if len(rows) != 8:
        raise ValueError('numeric gate requires one warmup and three repeats for both lengths')
    for length in (1, count):
        group = [r for r in rows if r['N'] == length]
        if len(group) != 4 or {r['run'] for r in group} != {0, 1, 2, 3}:
            raise ValueError('numeric repetitions are incomplete')
        for row in group:
            if row['warmup'] is not (row['run'] == 0):
                raise ValueError('numeric warmup order differs')
            tokens = read(path.parent/row['tokens_file'])
            if (len(tokens) != batch or any(len(r) != length or
                    any(type(t) is not int or t < 0 for t in r) for r in tokens)):
                raise ValueError('numeric token shape or IDs differ')
    evidence = read(directory/'candidate-evidence.json')
    artifacts = {p: i['artifact_id'] for p, i in pair.items()}
    if evidence['artifact_ids'] != artifacts:
        raise ValueError('numeric gate certified another binary pair')
    identities = value['configuration']['identities']
    if identities != pair:
        raise ValueError('numeric measured binary identities differ')
    c1, c2 = read(directory/'c1.json'), read(directory/'c2.json')
    positions = batch * (min(count, 256) if candidate.get('moe') else count)
    if c1.get('pass') is not True or c1.get('positions') != positions:
        raise ValueError('teacher-forced gate failed or used the wrong length')
    modes = {('L1', 0), ('L2', 0),
             ('L2' if pair['decode']['plan']['pg'] == 'pages' else 'L1', 1)}
    checks = c2.get('checks', [])
    if (c2.get('pass') is not True or c2.get('same_binary') is not True or
            len(checks) != 3 or
            {(r['mode'], int(r['decode_loop'])) for r in checks if r.get('pass') is True} != modes):
        raise ValueError('same-binary executor gate failed')
    actual = read(directory/'actual-execution.json')
    calls = actual['calls']
    if (actual['driver_sha256'] != candidate['script_sha256'] or len(calls) != 11 or
            any(r['device_loop_used'] != r['requested_device_loop'] for r in calls) or
            sum(r['tokens'] == 1 for r in calls) != 4 or
            sum(r['tokens'] == count for r in calls) != 7 or
            any(r['executor'] != candidate['mode'] or r['device_loop_used'] != bool(candidate['loop'] and r['tokens'] > 1) for r in calls[:8]) or
            {(r['executor'], int(r['requested_device_loop'])) for r in calls if r['tokens'] == count} != modes):
        raise ValueError('actual executor receipt differs')
    if expected_uuid is not None:
        identity = value['physical_device_identity']
        receipt = Path(value['physical_device_identity_file'])
        if (file_sha(receipt) != value['physical_device_identity_sha256'] or
                read(receipt) != identity or
                identity['cuda_uuid'] != expected_uuid.removeprefix('GPU-') or
                identity['nvml_uuid'].removeprefix('GPU-') != identity['cuda_uuid'] or
                identity['gpu_index'] != candidate['gpu']):
            raise ValueError('numeric physical-device proof differs')
    return dict(passed=True,stage=stage,pid=value['pid'],batch=batch,tokens=count,
                c1_positions=positions,artifact_ids=artifacts,
                repeat_tokens_identical=True,
                physical_device_identity=value.get('physical_device_identity'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--scheduler-root',type=Path,action='append')
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--job',required=True)
    parser.add_argument('--stage',choices=('smoke','longcheck'),required=True)
    parser.add_argument('--expected-uuid')
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    candidate = read(args.candidate)
    state, cards = read_states(args.root,args.scheduler_root)
    record = state[args.job]
    if cards[args.job] != candidate['gpu']:
        raise ValueError('numeric job was scheduled on another GPU')
    directory = record_directory(args.root,args.job,record)
    guard = completed_guard(directory,record)
    pair = load_driver(candidate).check_pair(candidate)
    result = validate_payload(directory/f'measurements/B{candidate["batch"]}/measurements.json',
                              candidate,pair,args.stage,args.expected_uuid)
    result.update(job=args.job,guard=guard,candidate=str(args.candidate.resolve()),
                  candidate_sha256=file_sha(args.candidate),validator_sha256=file_sha(__file__),
                  validator_dependencies_sha256={name:file_sha(Path(__file__).with_name(name)) for name in
                    ('collect_r16_candidates.py','r16_measurement_validation.py','summarize_r16.py')})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,job=args.job,stage=args.stage)))


if __name__ == '__main__':
    main()
