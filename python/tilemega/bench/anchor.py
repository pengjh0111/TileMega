"""Run one rotated, guarded comparison round with one process per arm."""
import argparse
import json
import os
import subprocess
from pathlib import Path

from .arms import check_binaries, sha256
from .report import metrics


def command(arm, cell, out, policy):
    root = Path(arm['root'])
    env = dict(os.environ, **arm.get('env', {}), PYTHONPATH=str(root / 'python'))
    workload = cell['workload']
    common = ['--model', cell['model_path'], '--prompt-ids', cell['prompt_ids'],
              '--batch', str(workload['batch']), '--out', str(out),
              '--policy', str(policy), '--warmup', '1', '--repeats', '1']
    if arm['kind'] == 'tm':
        cmd = [arm['python'], '-m', 'tilemega.serving.measure', *common,
               '--prefill-so', arm['prefill'], '--decode-so', arm['decode'],
               '--mode', arm.get('mode', 'L1'),
               '--decode-loop', str(arm.get('decode_loop', 0)),
               '--prefill-mode', arm.get('prefill_mode', 'L1'),
               '--step-events', str(arm.get('step_events', 1)),
               '--max-new-tokens', str(workload['osl'])]
    else:
        libs = sorted((Path(arm['python']).parent.parent / 'lib').glob(
            'python*/site-packages/nvidia/cu*/lib'))
        env['LD_LIBRARY_PATH'] = ':'.join(map(str, libs)) + ':' + env.get('LD_LIBRARY_PATH', '')
        cmd = [arm['python'], str(root / 'python/tilemega/serving/vllm_baseline.py'),
               *common, '--max-tokens', str(workload['osl'])]
    return cmd, env


def run_round(data, cell_name, rnd, out):
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise RuntimeError('measurement rounds must run through scheduler and GPU guard')
    cell = data['cells'][cell_name]
    arms = cell['arms']
    offset = rnd % len(arms)
    arms = arms[offset:] + arms[:offset]
    if list(data['cells']).index(cell_name) % 2:
        arms = list(reversed(arms))
    result = {'cell': cell_name, 'round': rnd, 'workload': cell['workload'],
              'order': [arm['label'] for arm in arms], 'arms': {},
              'prompt_sha256': sha256(cell['prompt_ids']),
              'step_bytes': cell.get('step_bytes'), 'dram_gbps': cell.get('dram_gbps')}
    failed = False
    out.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        check_binaries(arm)
        folder = out / arm['label']
        folder.mkdir(parents=True, exist_ok=True)
        cmd, env = command(arm, cell, folder, data['measurement_policy'])
        (folder / 'command.json').write_text(json.dumps({
            'command': cmd, 'env': arm.get('env', {}), 'kind': arm['kind'],
            'binaries': arm.get('binaries', {})},
            indent=2) + '\n')
        with (folder / 'stdout.txt').open('w') as stdout, (folder / 'stderr.txt').open('w') as stderr:
            code = subprocess.run(cmd, cwd=arm['root'], env=env,
                                  stdout=stdout, stderr=stderr).returncode
        record = {'exit_code': code, 'out': str(folder)}
        path = (folder / ('B' + str(cell['workload']['batch']))
                if arm['kind'] == 'vllm' else folder) / 'measurements.json'
        if code == 0:
            try:
                measured = json.loads(path.read_text())
                if arm['kind'] == 'tm' and arm.get('decode_loop') in (1, '1', True) and not measured.get('decode_loop_used'):
                    raise ValueError('requested loop was not used')
                record.update(measured, summary=metrics(measured, cell['workload'],
                    step_bytes=cell.get('step_bytes'), dram_gbps=cell.get('dram_gbps')))
            except (OSError, ValueError, KeyError) as error:
                record.update(exit_code=3, error=str(error))
        result['arms'][arm['label']] = record
        failed = failed or record['exit_code'] != 0
        if code == 75:
            result['invalidated'] = True
            break
    (out / 'round.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'cell': cell_name, 'round': rnd,
                      'arms': {k: v['exit_code'] for k, v in result['arms'].items()}}), flush=True)
    return 75 if result.get('invalidated') else (3 if failed else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms', type=Path, required=True)
    parser.add_argument('--cell', required=True)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run_round(json.loads(args.arms.read_text()), args.cell, args.round, args.out))


if __name__ == '__main__':
    main()
