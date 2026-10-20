#!/usr/bin/env python3
"""Freeze source and relink the compile command against an existing host build."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    root, build, out = [x.resolve() for x in (a.root, a.build, a.out)]
    out.mkdir(parents=True, exist_ok=False)
    source = out / 'source'
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    subprocess.run(['git', '-C', str(root), 'worktree', 'add', '--detach', str(source), head], check=True)
    prefixes = ['include', 'lib', 'tools', 'python', 'cmake', 'CMakeLists.txt']
    patch = subprocess.check_output(['git', '-C', str(root), 'diff', '--binary', 'HEAD', '--', *prefixes])
    (out / 'source.patch').write_bytes(patch)
    if patch:
        subprocess.run(['git', '-C', str(source), 'apply', '-'], input=patch, check=True)
    untracked = subprocess.check_output(['git', '-C', str(root), 'ls-files', '--others',
                                       '--exclude-standard', '--', *prefixes], text=True).splitlines()
    for name in untracked:
        if Path(name).suffix not in ('.h', '.hpp', '.cuh', '.cu', '.cpp', '.py', '.cmake'):
            continue
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, destination)
    cutlass = source / 'third_party/cutlass'
    if cutlass.is_dir():
        cutlass.rmdir()
    cutlass.symlink_to(root / 'third_party/cutlass', target_is_directory=True)
    commands = subprocess.check_output(['ninja', '-C', str(build), '-t', 'commands', 'tilemega'], text=True).splitlines()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    replacements, rebuilt = {}, []
    with (out / 'compile.log').open('w') as log:
        # Both entry points embed the source root used by CUDA and identity.
        # Reusing the native entry object would compile from the live tree.
        for name in ('compile.cpp', 'native_dense.cpp'):
            lines = [line for line in commands if ' -c ' + str(root / 'tools/commands' / name) in line]
            if not lines and name == 'native_dense.cpp':
                continue
            compile_line, = lines
            command = shlex.split(compile_line)
            original_object = command[command.index('-o') + 1]
            obj = out / (name + '.o')
            replacements[original_object] = str(obj)
            command[command.index('-o') + 1] = str(obj)
            if '-MF' in command:
                command[command.index('-MF') + 1] = str(out / (name + '.d'))
            command[command.index('-c') + 1] = str(source / 'tools/commands' / name)
            command = [x.replace(str(root / 'include'), str(source / 'include'))
                       if x.startswith('-I') else x for x in command]
            command = [x.replace(str(root), str(source)) if x.startswith('-DTILEMEGA_SOURCE_DIR=') else x
                       for x in command]
            subprocess.run(command, cwd=build, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            rebuilt.append(command)
    link_line, = [line for line in commands if ' -o tools/tilemega ' in line]
    segments = [s for s in link_line.split(' && ') if s.strip() != ':']
    if len(segments) != 1:
        raise ValueError('unexpected linker recipe; refuse to run shell fragments')
    link = shlex.split(segments[0])
    link = [replacements.get(x, x) for x in link]
    compiler = out / 'tilemega'
    link[link.index('-o') + 1] = str(compiler)
    with (out / 'link.log').open('w') as log:
        subprocess.run(link, cwd=build, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    objects = {str((build / x).resolve()): sha(build / x) for x in link
               if x.endswith(('.o', '.a')) and x not in replacements.values() and (build / x).is_file()}
    (out / 'receipt.json').write_text(json.dumps(dict(
        source=str(source), head=head, patch_sha256=sha(out / 'source.patch'),
        compiler=str(compiler), compiler_sha256=sha(compiler), compile_commands=rebuilt,
        link_command=link, reused_host_objects=objects,
        scope='compiler entry translation units rebuilt; host core reused; CUDA uses frozen headers'),
        indent=2) + '\n')
    print(compiler, flush=True)


if __name__ == '__main__':
    main()
