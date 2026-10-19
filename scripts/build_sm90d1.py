#!/usr/bin/env python3
"""Rebuild the recorded SM90R1 coordinates using repository-owned inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import lzma
import os
from pathlib import Path
import subprocess

from sm90d1_common import ROOT, RUN, read, sha, write


def build(config, compiler, tag, changes, phase):
    config = Path(config).resolve()
    c = read(config)
    manifest = read(ROOT / 'configs/serving/sm90d1/inputs/manifest.json')
    packed = ROOT / f"configs/serving/sm90d1/inputs/{c['model']}/{phase}.export.json.xz"
    expected = manifest[str(packed.relative_to(ROOT))]
    if sha(packed) != expected['compressed_sha256']:
        raise ValueError('compressed build input changed')
    directory = RUN / 'build' / tag / f"{c['model']}-B{c['batch']}"
    directory.mkdir(parents=True, exist_ok=True)
    graph = directory / f'{phase}.export.json'
    graph.write_bytes(lzma.decompress(packed.read_bytes()))
    if sha(graph) != expected['raw_sha256']:
        raise ValueError('exported-program byte identity changed')
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
    p.add_argument('--tag', default='reference')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--phase', choices=('prefill', 'decode', 'both'), default='both')
    p.add_argument('--changes', default='{}')
    p.add_argument('configs', nargs='+')
    a = p.parse_args()
    tasks = [(c, phase) for c in a.configs for phase in
             (('prefill', 'decode') if a.phase == 'both' else (a.phase,))]
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futures = [pool.submit(build, c, a.compiler, a.tag, json.loads(a.changes), phase)
                   for c, phase in tasks]
        for future in futures:
            future.result()


if __name__ == '__main__':
    main()
