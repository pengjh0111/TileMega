"""Compile a CUDA artifact once per source, dependency closure and toolchain.

The compiler supplies its exact nvcc command; this module does no code generation
or scheduling. Dependency discovery uses that command's preprocessing options.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from tilemega.cache import atomic_json, file_sha, key, locked, record_outputs, valid_record


def split_command(command: list[str]):
    if '-o' not in command:
        raise ValueError('artifact command needs -o OUTPUT')
    output = Path(command[command.index('-o') + 1]).resolve()
    sources, options = [], []
    index = 1
    while index < len(command):
        token = command[index]
        if token in ('-o', '-x', '-L', '-l', '-Xlinker', '-cudart'):
            index += 2
            continue
        if token in ('-shared', '-c', '-Xptxas=-v') or token.startswith(('-L', '-l', '-Xcompiler=', '-Xlinker=')):
            index += 1
            continue
        if token in ('-I', '-D', '-U', '-isystem', '-include', '--include-path', '--pre-include', '-arch', '--gpu-architecture'):
            options.extend(command[index:index + 2])
            index += 2
            continue
        if not token.startswith('-') and Path(token).suffix in ('.cu', '.cpp', '.cc', '.c'):
            sources.append(Path(token).resolve())
        else:
            options.append(token)
        index += 1
    if not sources:
        raise ValueError('artifact command contains no source')
    return output, sources, options


def identity(command: list[str]) -> tuple[str, dict]:
    output, sources, options = split_command(command)
    nvcc = str(Path(shutil.which(command[0]) or command[0]).resolve())
    version = subprocess.check_output([nvcc, '--version'], text=True)
    dependencies = set()
    for source in sources:
        listing = subprocess.check_output([nvcc, *options, '-x', 'cu', '-M', '-MT',
                                           'tilemega_artifact', str(source)], text=True)
        listing = listing.replace('\\\n', ' ')
        if ':' not in listing:
            raise ValueError('nvcc -M did not return a dependency rule')
        dependencies.update(Path(p).resolve() for p in shlex.split(listing.split(':', 1)[1]))
    # Temporary candidate names are not semantic inputs. The contents and their
    # order remain in the key; all other dependency paths retain their identity.
    names = {p: f'input:{i}' for i, p in enumerate(sources)}
    closure = {names.get(p, str(p)): file_sha(p) for p in sorted(dependencies)}
    normalized = [str(x) for x in command]
    normalized[0] = nvcc
    normalized[normalized.index('-o') + 1] = '<output>'
    for i, token in enumerate(normalized):
        if not token.startswith('-') and Path(token).resolve() in names:
            normalized[i] = names[Path(token).resolve()]
    inputs = dict(nvcc=nvcc, nvcc_version=version, argv=normalized,
                  generated_sources=[file_sha(p) for p in sources], dependencies=closure)
    return key(inputs), inputs


def compile_artifact(command: list[str], cache: Path, log: Path) -> dict:
    output, _, _ = split_command(command)
    start = time.monotonic()
    digest, inputs = identity(command)
    directory = cache / digest
    marker = directory / 'record.json'
    directory.mkdir(parents=True, exist_ok=True)
    with locked(directory / '.lock'):
        hit = valid_record(marker)
        if not hit:
            temporary = directory / f'pending-{os.getpid()}.so'
            argv = list(command)
            argv[argv.index('-o') + 1] = str(temporary)
            try:
                with (directory / 'ptxas.txt').open('w') as stream:
                    subprocess.run(argv, check=True, stdout=stream, stderr=subprocess.STDOUT)
                os.replace(temporary, directory / 'artifact.so')
                record_outputs(marker, [directory / 'artifact.so', directory / 'ptxas.txt'], inputs=inputs)
            finally:
                temporary.unlink(missing_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output != directory / 'artifact.so':
            shutil.copy2(directory / 'artifact.so', output)
        shutil.copy2(directory / 'ptxas.txt', log)
    result = dict(key=digest, hit=hit, artifact=str(directory / 'artifact.so'),
                  output=str(output), seconds=time.monotonic() - start, inputs=inputs)
    atomic_json(Path(str(output) + '.cache.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--command', required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compile_artifact(shlex.split(args.command), args.cache, args.log)))


if __name__ == '__main__':
    main()
