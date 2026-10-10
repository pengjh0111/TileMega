#!/usr/bin/env python3
"""Freeze a TaskBody fixture before submitting correctness-only GPU jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--implementations', nargs='+', required=True)
    parser.add_argument('--scope', required=True)
    parser.add_argument('--define', action='append', default=[])
    args = parser.parse_args()
    definitions={}
    for item in args.define:
        name,separator,value=item.partition('=')
        if not separator or not re.fullmatch(r'DM_TEST_[A-Z0-9_]+',name) or not re.fullmatch(r'[0-9]+',value) or name in definitions:
            parser.error('--define requires distinct DM_TEST_NAME=INTEGER entries')
        definitions[name]=value
    repo = Path(__file__).resolve().parents[3]
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'framework').mkdir()
    for name in ['capture_macros_dm.py', 'identity_dm.py']:
        shutil.copy2(Path(__file__).with_name(name), root / 'framework' / name)
    shutil.copy2(Path(__file__).with_name('check_task_body.py'), root / 'build.py')
    shutil.copy2(args.fixture, root / 'fixture.cu')
    shutil.copytree(repo / 'include', root / 'include')
    for name in ['include', 'tools/util/include']:
        shutil.copytree(repo / 'third_party/cutlass' / name, root / 'third_party/cutlass' / name)
    inputs = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in root.rglob('*') if path.is_file()}
    preparation = dict(evidence='stated', source_head=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        diff_sha256=hashlib.sha256(subprocess.check_output(['git', 'diff'], cwd=repo)).hexdigest(),
        scope=args.scope, implementations=args.implementations, definitions=definitions, inputs=inputs)
    (root / 'preparation.json').write_text(json.dumps(preparation, indent=2) + '\n')
    steps = []
    def step(name, command, after, priority, timeout):
        steps.append(dict(name=name, command=['flock', '/root/r14_work/gpu.lock',
            'timeout', str(timeout), *command], cwd=str(repo), after=after,
            gpu=False, timeout_s=timeout+600, priority=priority))
    for arch in [89, 80, 90, 100, 120]:
        step(f'build_sm_{arch}', ['python3', str(root/'build.py'), '--root', str(root),
             '--arch', str(arch)], [], 0 if arch==89 else 200, 1800)
    for process in range(50):
        step(f'check_{process:02}', [str(root/'task-sm_89')], ['build_sm_89'], 10, 600)
    for tool in ['memcheck', 'racecheck']:
        step(tool, ['/usr/local/cuda/bin/compute-sanitizer', '--tool', tool,
             '--error-exitcode', '86', str(root/'task-sm_89')], ['build_sm_89'], 2, 1800)
    (root/'queue').mkdir()
    (root/'queue/queue_task.json').write_text(json.dumps(steps, indent=2)+'\n')
    print(json.dumps(dict(root=str(root), steps=len(steps), snapshot_files=len(inputs))), flush=True)


if __name__ == '__main__':
    main()
