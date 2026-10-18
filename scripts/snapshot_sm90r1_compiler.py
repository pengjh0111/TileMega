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
    compile_line, = [line for line in commands if ' -c ' + str(root / 'tools/commands/compile.cpp') in line]
    command = shlex.split(compile_line)
    original_object = command[command.index('-o') + 1]
    obj = out / 'compile.cpp.o'
    command[command.index('-o') + 1] = str(obj)
    if '-MF' in command:
        command[command.index('-MF') + 1] = str(out / 'compile.cpp.d')
    command[command.index('-c') + 1] = str(source / 'tools/commands/compile.cpp')
    command = [x.replace(str(root / 'include'), str(source / 'include'))
               if x.startswith('-I') else x for x in command]
    command = [x.replace(str(root), str(source)) if x.startswith('-DTILEMEGA_SOURCE_DIR=') else x
               for x in command]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    with (out / 'compile.log').open('w') as log:
        subprocess.run(command, cwd=build, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    link_line, = [line for line in commands if ' -o tools/tilemega ' in line]
    segments = [s for s in link_line.split(' && ') if s.strip() != ':']
    if len(segments) != 1:
        raise ValueError('unexpected linker recipe; refuse to run shell fragments')
    link = shlex.split(segments[0])
    link = [str(obj) if x == original_object else x for x in link]
    compiler = out / 'tilemega'
    link[link.index('-o') + 1] = str(compiler)
    with (out / 'link.log').open('w') as log:
        subprocess.run(link, cwd=build, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    objects = {str((build / x).resolve()): sha(build / x) for x in link
               if x.endswith(('.o', '.a')) and x != str(obj) and (build / x).is_file()}
    (out / 'receipt.json').write_text(json.dumps(dict(
        source=str(source), head=head, patch_sha256=sha(out / 'source.patch'),
        compiler=str(compiler), compiler_sha256=sha(compiler), compile_command=command,
        link_command=link, reused_host_objects=objects,
        scope='compile translation unit rebuilt; unchanged host objects reused; CUDA uses frozen headers'),
        indent=2) + '\n')
    print(compiler, flush=True)


if __name__ == '__main__':
    main()
