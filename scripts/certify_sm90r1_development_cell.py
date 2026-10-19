#!/usr/bin/env python3
"""Bind an existing full-length same-artifact numerical run to a final cell."""
import argparse
import json
from pathlib import Path

from r16_measurement_validation import completed_guard, file_sha
from run_sm90r1_final_cell import absolute, validate_numeric, validate_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--scheduler-state', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    identities = validate_pair(config)
    record = json.loads(args.scheduler_state.read_text())[args.job]
    raw = Path(record['out'])
    completed_guard(raw, record)
    directory = raw / 'measurements'
    batch = config['batch']
    measured = json.loads((directory / f'B{batch}/measurements.json').read_text())
    metadata = measured['configuration']
    expected = {phase: value['artifact_id'] for phase, value in identities.items()}
    actual = {phase: value['artifact_id'] for phase, value in metadata['identities'].items()}
    if (actual != expected or measured['batch'] != batch or measured['max_tokens'] != 1024 or
            measured['model_config_sha256'] != config['model_config_sha256'] or
            measured['prompt_ids_sha256'] != config['prompt_ids_sha256'] or
            metadata['runtime_manifest_sha256'] != config['runtime']['manifest_sha256']):
        raise ValueError('development proof belongs to another artifact, workload or runtime')
    command = json.loads((raw / 'physical-device-command.json').read_text())['command']
    if ('--check-executors' not in command or '--hf-check' not in command or
            '--prefill-mode' in command):
        raise ValueError('proof must check both phase binaries across all executors')
    checkpoint = config.get('checkpoint')
    if checkpoint:
        manifest = json.loads((absolute(checkpoint['path']) / 'random_checkpoint.json').read_text())
        if measured['checkpoint']['checkpoint_sha256'] != manifest['checkpoint_sha256']:
            raise ValueError('development proof used another checkpoint')
    runtime = json.loads((absolute(config['runtime']['root']) / 'manifest.json').read_text())
    c1 = json.loads((directory / 'c1.json').read_text())
    if c1['checker_sha256'] != runtime['files']['tilemega/serving/hf_check.py']:
        raise ValueError('development proof used another C-1 checker')
    reference = json.loads((directory / 'reference_tokens.json').read_text())
    for check in json.loads((directory / 'c2.json').read_text())['checks']:
        path = directory / f"tokens_{check['mode']}_loop{check['decode_loop']}.json"
        if json.loads(path.read_text()) != reference:
            raise ValueError('C-2 token files differ from the claimed pass')
    config_sha = file_sha(args.config)
    binding = dict(status='passed', config_sha256=config_sha, artifact_ids=expected,
                   correctness_job=args.job, correctness_gpu=measured['gpu_index'],
                   guard_log_sha256=file_sha(raw / 'guard.jsonl'),
                   certifier_sha256=file_sha(__file__),
                   provenance='reused full-length same-artifact development proof',
                   source_measurement_sha256=file_sha(directory / f'B{batch}/measurements.json'),
                   command_sha256=file_sha(raw / 'physical-device-command.json'))
    for name in ('c1', 'c2', 'reference_tokens'):
        path = directory / (name + '.json')
        binding[name] = dict(path=str(path), sha256=file_sha(path))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise ValueError('numerical certificate already exists')
    temporary = args.out.with_suffix('.pending')
    temporary.write_text(json.dumps(binding, indent=2) + '\n')
    validate_numeric(temporary, config_sha, identities, batch,
                     256 if config['model'] == 'qwen3_30b_a3b' else 1024)
    temporary.replace(args.out)


if __name__ == '__main__':
    main()
