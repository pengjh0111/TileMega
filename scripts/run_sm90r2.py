#!/usr/bin/env python3
"""Run one guarded dense-decode development or correctness job."""
import argparse
import os
from pathlib import Path
import subprocess
import time

from sm90d1_common import ROOT, PYTHON, read, sha, write

RUN = ROOT / 'runs/sm90r2'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--stage', choices=('measure', 'correctness'), required=True)
    p.add_argument('--round', type=int, default=0)
    p.add_argument('--positions', type=int, default=256)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if os.environ.get('TILEMEGA_DEVICE_INDEX') == '6':
        raise ValueError('GPU6 is excluded')
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise ValueError('GPU jobs require the scheduler and guard')
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / 'selected-config.json').exists():
        raise FileExistsError('retain the prior job receipt; choose a new attempt')
    c = read(a.config)
    reference = ROOT / 'runs/sm90d1/build/reference2' / f"{c['model']}-B{c['batch']}"
    prefill = reference / 'prefill.so'
    decode = (reference if a.tag == 'reference' else RUN / 'build' / a.tag / reference.name) / 'decode.so'
    for library in (prefill, decode):
        identity = read(str(library) + '.identity.json')
        if sha(library) != identity['binary_sha256']:
            raise ValueError(f'artifact identity changed: {library}')
        if identity['trace']:
            raise ValueError('instrumented libraries cannot enter this harness')
    command = [PYTHON, '-m', 'tilemega.serving.r16_run',
               '--prefill', str(prefill), '--decode', str(decode),
               '--prompt-ids', str(ROOT / c['prompt_ids']), '--batch', str(c['batch']),
               '--max-tokens', '1024', '--warmup', '1',
               '--repeats', '3' if a.stage == 'measure' else '1',
               '--mode', 'L1', '--decode-loop', '1', '--prefill-mode', 'L1',
               '--round-index', str(a.round), '--out', str(a.out / 'measurements')]
    if a.stage == 'correctness':
        command += ['--check-step-loop']
        if a.round == 0:
            command += ['--hf-check', '--hf-positions', str(a.positions)]
    command = [PYTHON, '-m', 'tilemega.serving.random_checkpoint',
               '--config', str(ROOT / c['model_dir']), '--out', str(a.out / 'checkpoint'),
               '--seed', '0', '--max-gib', '32', '--', *command]
    evidence = dict(config=str(a.config.resolve()), config_sha256=sha(a.config),
                    decode_sha256=sha(decode), prefill_sha256=sha(prefill), command=command,
                    stage=a.stage, fresh_process=a.round, started=time.time())
    write(a.out / 'selected-config.json', evidence)
    code = subprocess.call(command, cwd=ROOT)
    evidence.update(ended=time.time(), exit_code=code)
    write(a.out / 'execution.json', evidence)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
