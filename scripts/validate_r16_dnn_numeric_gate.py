#!/usr/bin/env python3
"""Require a true full-graph numerical result before DNN timing."""
import argparse
import json
from pathlib import Path

from r16_measurement_validation import completed_guard, file_sha, native_identity


def validate_payload(gate, config, artifact_id, checker_sha256, checkpoint_sha256, bridge_sha256):
    expected = dict(artifact_id=artifact_id, checker_sha256=checker_sha256,
        source_checkpoint_sha256=checkpoint_sha256, bridge_sha256=bridge_sha256,
        archive_sha256=config['export_files']['exported_program.pt2'],
        input_tensors_sha256=config['export_files']['inputs_b1.safetensors'], batch=1)
    if gate.get('passed') is not True or gate.get('evidence') != 'verified':
        raise ValueError('full-graph numerical result did not pass')
    for key, value in expected.items():
        if gate.get(key) != value:
            raise ValueError('numerical result differs from frozen candidate: ' + key)
    if {row['mode'] for row in gate['cases']} != {1, 2}:
        raise ValueError('same-binary L1/L2 numerical cases are absent')
    return dict(passed=True, artifact_id=artifact_id, batch=1, modes=[1, 2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--scheduler-root', type=Path)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--expected-uuid', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.candidate.read_text())
    scheduler=args.scheduler_root or args.root/'runs/r16/scheduler'
    state = json.loads((scheduler / f'gpu{config["gpu"]}/state.json').read_text())
    event = state[args.job]
    attempt = Path(event['out'])
    guard = completed_guard(attempt, event)
    identity = native_identity(Path(config['directory']) / 'plan.so')
    manifest = json.loads((Path(config['export']) / 'manifest.json').read_text())
    checkpoint = file_sha(Path(config['export']) / 'checkpoint/model.safetensors')
    if checkpoint != manifest['checkpoint']['sha256']:
        raise ValueError('source checkpoint differs from its export manifest')
    path = attempt / 'full-model-gate.json'
    result = validate_payload(json.loads(path.read_text()), config, identity['artifact_id'],
        file_sha(Path(config['source']) / 'python/tilemega/dnn/check_generated.py'),
        checkpoint, file_sha(config['bridge']))
    device_path = attempt / 'physical-device-identity.json'
    device = json.loads(device_path.read_text())
    command = json.loads((attempt / 'physical-device-command.json').read_text())
    if (device['probe_exit_code'] != 0 or command['actual_returncode'] != 0 or
            device['cuda_uuid'] != args.expected_uuid.removeprefix('GPU-') or
            device['cuda_uuid'] != device['nvml_uuid'].removeprefix('GPU-') or
            device['gpu_index'] != config['gpu']):
        raise ValueError('numerical result belongs to another physical device')
    result.update(job=args.job, candidate=str(args.candidate),
        candidate_sha256=file_sha(args.candidate), gate=str(path), gate_sha256=file_sha(path),
        physical_device_identity=device, physical_device_identity_sha256=file_sha(device_path),
        guard=guard, validator_sha256=file_sha(__file__))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=True, job=args.job)))


if __name__ == '__main__':
    main()
