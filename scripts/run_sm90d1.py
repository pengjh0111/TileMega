#!/usr/bin/env python3
"""Run one budgeted diagnostic job inside the existing guarded scheduler."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from sm90d1_common import ROOT, RUN, PYTHON, budgeted, read, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path)
    p.add_argument('--tag', default='reference')
    p.add_argument('--stage', choices=('measure', 'overhead', 'correctness', 'trace', 'primitive', 'resources'), required=True)
    p.add_argument('--round', type=int, default=0)
    p.add_argument('--decode', type=Path)
    p.add_argument('--mode', choices=('L1', 'L2'))
    p.add_argument('--loop', type=int, choices=(0, 1))
    p.add_argument('--timeout', type=int, default=900)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    if a.stage == 'resources':
        return budgeted([PYTHON, str(ROOT / 'scripts/resources_sm90d1.py'),
                         '--out', str(a.out / 'resources.json')], a.timeout, 'resource-queries')
    if a.stage == 'primitive':
        binary = RUN / ('memory-primitives' if a.tag == 'memory' else 'primitives')
        write(a.out / 'artifact.json', dict(binary=str(binary), sha256=sha(binary)))
        with (a.out / 'primitives.csv').open('w') as stream:
            command = ['bash', '-c', 'exec "$1" > "$2"', 'sm90d1',
                       str(RUN / ('memory-primitives' if a.tag == 'memory' else 'primitives')),
                       str(a.out / 'primitives.csv')]
            return budgeted(command, a.timeout, f'primitive-{a.round}')
    c = read(a.config)
    base = RUN / 'build/reference2' / f"{c['model']}-B{c['batch']}"
    prefill = base / 'prefill.so'
    decode = a.decode or RUN / 'build' / ('reference2' if a.tag == 'reference' else a.tag) / base.name / 'decode.so'
    for path in (prefill, decode):
        identity = read(str(path) + '.identity.json')
        if sha(path) != identity['binary_sha256']:
            raise ValueError('binary differs from its bound identity')
        if a.stage == 'measure' and identity['trace']:
            raise ValueError('instrumented binary must never enter TPOT')
    mode = a.mode or c['execution']['decode']
    loop = c['execution']['loop'] if a.loop is None else a.loop
    if a.tag.startswith('PG'):
        mode, loop = 'L1', 0
    if a.tag.startswith('EV'):
        mode, loop = 'L2', 0
    evidence = dict(config=str(a.config.resolve()), config_sha256=sha(a.config),
                    binary_sha256=sha(decode), prefill_sha256=sha(prefill), stage=a.stage,
                    round=a.round, mode=mode, loop=loop, performance_eligible=a.stage == 'measure')
    write(a.out / 'selected-config.json', evidence)
    if a.stage == 'trace':
        command = [PYTHON, str(ROOT / 'scripts/trace_sm90d1.py'),
                   '--prefill', str(prefill), '--decode', str(decode),
                   '--batch', str(c['batch']), '--mode', mode, '--loop', str(loop),
                   '--out', str(a.out / 'trace')]
    else:
        command = [PYTHON, '-m', 'tilemega.serving.r16_run',
                   '--prefill', str(prefill), '--decode', str(decode),
                   '--prompt-ids', str(ROOT / c['prompt_ids']), '--batch', str(c['batch']),
                   '--max-tokens', '1024', '--warmup', '1',
                   '--repeats', '3' if a.stage in ('measure', 'overhead') else '1',
                   '--mode', mode, '--decode-loop', str(loop), '--prefill-mode', 'L1',
                   '--round-index', str(a.round), '--out', str(a.out / 'measurements')]
        if a.stage == 'correctness':
            command += ['--hf-check', '--hf-positions', '256', '--check-executors']
    # The checkpoint wrapper expands its environment path without shell evaluation.
    if a.stage == 'trace':
        code = ('import os,subprocess,sys; '
                'raise SystemExit(subprocess.call([os.environ["TILEMEGA_RANDOM_CHECKPOINT"] '
                'if x=="{checkpoint}" else x for x in sys.argv[1:]]))')
        command = [PYTHON, '-c', code, *command]
    command = [PYTHON, '-m', 'tilemega.serving.random_checkpoint',
               '--config', str(ROOT / c['model_dir']), '--out', str(a.out / 'checkpoint'),
               '--seed', '0', '--max-gib', '32', '--', *command]
    return budgeted(command, a.timeout, f"{base.name}-{a.tag}-{a.stage}-{a.round}")


if __name__ == '__main__':
    raise SystemExit(main())
