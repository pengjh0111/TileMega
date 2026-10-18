#!/usr/bin/env python3
"""Run one immutable SM90R1 LLM cell inside the shared GPU guard."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from r16_measurement_validation import file_sha, native_identity


ROOT = Path(__file__).resolve().parents[1]


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def validate_pair(config):
    identities = {}
    for phase in ('prefill', 'decode'):
        binding = config['libraries'][phase]
        binary = absolute(binding['path'])
        identity = native_identity(binary)
        if (identity['artifact_id'] != binding['artifact_id'] or
                identity['binary_sha256'] != binding['sha256'] or
                file_sha(str(binary) + '.identity.json') != binding['identity_sha256']):
            raise ValueError('selected artifact binding changed: ' + phase)
        plan = identity['plan']
        if identity['nvcc_arch'] != 'sm_90a' or plan['phase'] != phase:
            raise ValueError('wrong architecture or serving phase')
        if not plan['batch_lo'] <= config['batch'] <= plan['batch_hi']:
            raise ValueError('batch is outside the compiled interval')
        if plan['capacity'] < 1088 or plan['seq'] != (64 if phase == 'prefill' else 1):
            raise ValueError('compiled workload differs from 64/1024')
        identities[phase] = identity
    pre, dec = (identities[p]['plan'] for p in ('prefill', 'decode'))
    if pre['model'] != dec['model'] or len(pre['gemms']) != len(dec['gemms']):
        raise ValueError('prefill and decode are different full models')
    recipes = []
    for plan in (pre, dec):
        recipes.append({b['name']: (b['recipe'], b['dtype'], b['constant'])
                        for b in plan['buffers'] if 'recipe' in b})
    conflicts = [name for name in recipes[0].keys() & recipes[1].keys()
                 if recipes[0][name] != recipes[1][name]]
    if conflicts:
        raise ValueError('phase weight recipes disagree: ' + ', '.join(conflicts))
    return identities


def validate_numeric(path, config_sha, identities, batch, positions):
    binding = json.loads(path.read_text())
    expected = {phase: value['artifact_id'] for phase, value in identities.items()}
    if (binding['status'] != 'passed' or binding['config_sha256'] != config_sha or
            binding['artifact_ids'] != expected):
        raise ValueError('numerical receipt belongs to another configuration')
    for name in ('c1', 'c2', 'reference_tokens'):
        if file_sha(binding[name]['path']) != binding[name]['sha256']:
            raise ValueError('numerical evidence changed: ' + name)
    c1 = json.loads(Path(binding['c1']['path']).read_text())
    c2 = json.loads(Path(binding['c2']['path']).read_text())
    if (not c1['pass'] or c1['batch'] != batch or c1['max_new_tokens'] != positions or
            c1['positions'] != batch * positions or not c2['pass'] or
            not c2['same_binary'] or not all(row['pass'] for row in c2['checks'])):
        raise ValueError('final C-1/C-2 requirements are not satisfied')
    modes = {(row['mode'], int(row['decode_loop'])) for row in c2['checks']}
    if not {('L1', 0), ('L2', 0), ('L1', 1)} <= modes:
        raise ValueError('C-2 executor coverage is incomplete')
    tokens = json.loads(Path(binding['reference_tokens']['path']).read_text())
    if len(tokens) != batch or any(len(row) != 1024 for row in tokens):
        raise ValueError('C-2 did not cover a full 1024-token request')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=('correctness', 'measure'), required=True)
    parser.add_argument('--round-index', type=int, default=0)
    parser.add_argument('--numeric-receipt', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    from tilemega.serving.baseline_common import require_guard
    require_guard()
    config = json.loads(args.config.read_text())
    config_sha = file_sha(args.config)
    if file_sha(__file__) != config['driver_sha256']:
        raise ValueError('selected final driver changed')
    identities = validate_pair(config)
    runtime = absolute(config['runtime']['root'])
    manifest = runtime / 'manifest.json'
    if file_sha(manifest) != config['runtime']['manifest_sha256']:
        raise ValueError('selected Python runtime manifest changed')
    if file_sha(absolute(config['model_dir']) / 'config.json') != config['model_config_sha256']:
        raise ValueError('model configuration changed')
    if file_sha(absolute(config['prompt_ids'])) != config['prompt_ids_sha256']:
        raise ValueError('frozen prompt IDs changed')
    moe = config['model'] == 'qwen3_30b_a3b'
    positions = 256 if moe else 1024
    if args.stage == 'measure':
        if args.numeric_receipt is None:
            parser.error('measure requires --numeric-receipt')
        validate_numeric(args.numeric_receipt, config_sha, identities, config['batch'], positions)
    args.out.mkdir(parents=True, exist_ok=False)
    evidence = dict(config=str(args.config.resolve()), config_sha256=config_sha,
                    driver_sha256=file_sha(__file__), stage=args.stage,
                    round_index=args.round_index,
                    artifact_ids={p: v['artifact_id'] for p, v in identities.items()},
                    runtime_manifest_sha256=file_sha(manifest),
                    timing_requested=args.stage == 'measure',
                    acceptance='pending completed guard and independent collection')
    (args.out / 'selected-config.json').write_text(json.dumps(evidence, indent=2) + '\n')
    python = config['python']
    command = [python, '-m', 'tilemega.serving.r16_run',
               '--prefill', str(absolute(config['libraries']['prefill']['path'])),
               '--decode', str(absolute(config['libraries']['decode']['path'])),
               '--prompt-ids', str(absolute(config['prompt_ids'])),
               '--batch', str(config['batch']), '--max-tokens', '1024', '--warmup', '1',
               '--repeats', '3' if args.stage == 'measure' else '1',
               '--mode', config['execution']['decode'],
               '--decode-loop', str(config['execution']['loop']),
               '--round-index', str(args.round_index), '--out', str(args.out)]
    if args.stage == 'correctness':
        command += ['--hf-check', '--hf-positions', str(positions), '--check-executors']
    else:
        command += ['--prefill-mode', config['execution']['prefill']]
    environment = dict(os.environ, PYTHONPATH=str(runtime / 'python'),
                       TILEMEGA_RUNTIME_MANIFEST=str(manifest), PYTHONNOUSERSITE='1')
    if moe:
        environment['TILEMEGA_RANDOM_CHECKPOINT'] = str(absolute(config['checkpoint']['path']))
    else:
        command = [python, '-m', 'tilemega.serving.random_checkpoint',
                   '--config', str(absolute(config['model_dir'])),
                   '--out', str(args.out.parent / 'checkpoint'), '--seed', '0',
                   '--max-gib', '32', '--', *command]
    return subprocess.run(command, cwd=ROOT, env=environment).returncode


if __name__ == '__main__':
    raise SystemExit(main())
