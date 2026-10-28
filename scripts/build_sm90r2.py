#!/usr/bin/env python3
"""Build SM90R2 development programs from repository-owned inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess

from sm90d1_common import ROOT, read, sha, write
from sm90r4_inputs import extract
RUN = ROOT / os.environ.get('TILEMEGA_BUILD_RUN', 'runs/sm90r2')


def build(config, compiler, tag, changes, phase):
    config = Path(config).resolve()
    c = read(config)
    directory = RUN / 'build' / tag / f"{c['model']}-B{c['batch']}"
    directory.mkdir(parents=True, exist_ok=True)
    graph = directory / f'{phase}.export.json'
    extract(c['model'], phase, graph)
    binary = directory / f'{phase}.so'
    receipt = directory / f'{phase}.build.json'
    if receipt.exists():
        previous = read(receipt)
        if previous['status'] == 'completed' and binary.is_file():
            return previous
        raise ValueError('retain failed build evidence; choose another tag')
    options = list(c['compile_options'][phase])
    for key, value in changes.items():
        if key in options:
            options[options.index(key) + 1] = str(value)
        else:
            options += [key, str(value)]
    command = [str(Path(compiler).resolve()), 'compile', str(graph), str(binary), *options]
    command = ['/usr/bin/python3', str(ROOT / 'scripts/run_sm90r1_build.py'),
               '--receipt', str(receipt), '--lock', str(directory / f'{phase}.lock'),
               '--timeout', '5400', '--max-gib', '48', '--', *command]
    with (directory / f'{phase}.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print(json.dumps(dict(config=str(config), phase=phase, status='failed', log=str(log.name))), flush=True)
        return read(receipt)
    identity = read(str(binary) + '.identity.json')
    binding = dict(config=str(config), config_sha256=sha(config), tag=tag,
                   phase=phase, binary=str(binary), binary_sha256=sha(binary),
                   artifact_id=identity['artifact_id'], head=identity['source']['head'],
                   changes=changes, performance_eligible=not identity['trace'])
    write(directory / f'{phase}.binding.json', binding)
    print(json.dumps(binding), flush=True)
    return binding


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--jobs', type=int, default=1)
    p.add_argument('--phase', choices=('prefill', 'decode', 'both'), default='both')
    p.add_argument('--changes', default='{}')
    p.add_argument('configs', nargs='+')
    a = p.parse_args()
    tasks = [(c, phase) for c in a.configs for phase in
             (('prefill', 'decode') if a.phase == 'both' else (a.phase,))]
    development = {'qwen3_0_6b_b1', 'qwen3_0_6b_b16', 'llama3_2_1b_b1',
                   'llama3_1_8b_b1', 'llama3_1_8b_b16'}
    tasks.sort(key=lambda task: Path(task[0]).stem not in development)
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futures = [pool.submit(build, c, a.compiler, a.tag, json.loads(a.changes), phase)
                   for c, phase in tasks]
        for future in futures:
            future.result()


if __name__ == '__main__':
    main()
