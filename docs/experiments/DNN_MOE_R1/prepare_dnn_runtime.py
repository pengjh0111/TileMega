#!/usr/bin/env python3
"""Freeze native DNN correctness inputs and generate a lock-serialized queue."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--support-from', type=Path, required=True)
    parser.add_argument('--processes', type=int, default=50)
    parser.add_argument('--architectures', type=int, nargs='+', default=[89, 80, 90, 100, 120])
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    root = args.out.resolve()
    if args.processes <= 0 or 89 not in args.architectures:
        raise ValueError('require positive fresh-process count and sm_89 execution artifact')
    root.mkdir(parents=True, exist_ok=False)
    framework = root / 'framework'
    framework.mkdir()
    for name in ['capture_macros_dm.py', 'identity_dm.py']:
        shutil.copy2(Path(__file__).with_name(name), framework / name)
    shutil.copytree(repo / 'include', root / 'include')
    for name in ['include', 'tools/util/include']:
        shutil.copytree(repo / 'third_party/cutlass' / name, root / 'third_party/cutlass' / name)
    shutil.copy2(repo / 'test/unit/dm_dnn_runtime_test.cu', root / 'dm_dnn_runtime_test.cu')
    shutil.copy2(Path(__file__).with_name('check_dnn_runtime.py'), root / 'build.py')
    previous = args.support_from.resolve()
    support = json.loads((previous / 'support.json').read_text())
    for row in support:
        assert sha(Path(row['object'])) == row['sha256']
        original = Path(row['command'][row['command'].index('-c') + 1])
        relative = original.parts[original.parts.index('lib'):]
        if sha(original) != sha(repo.joinpath(*relative)):
            raise ValueError('support source changed; rebuild support before preparing: ' + str(original))
    for name in ['support.json', 'support_sources.json']:
        shutil.copy2(previous / name, root / name)
    inputs = {str(path): sha(path) for path in root.rglob('*') if path.is_file()}
    preparation = dict(evidence='stated', source_head=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        diff_sha256=hashlib.sha256(subprocess.check_output(['git', 'diff'], cwd=repo)).hexdigest(),
        scope='Native DNN forward ABI/executor correctness; no model gate or timing claim', inputs=inputs)
    (root / 'preparation.json').write_text(json.dumps(preparation, indent=2) + '\n')
    steps = []
    lock = ['flock', '/root/r14_work/gpu.lock']

    def step(name, command, dependencies, priority, timeout):
        steps.append(dict(name=name, command=lock + ['timeout', str(timeout)] + command,
            cwd=str(repo), gpu=False, timeout_s=timeout + 600,
            after=dependencies, priority=priority))

    configs = [(0, pg, split) for pg in [0, 1] for split in [1, 5]] + [(1, pg, 1) for pg in [0, 1]]
    for embedding, paged, split in configs:
        tag = f'emb{embedding}-pg{paged}-split{split}'
        for arch in args.architectures:
            step(f'build_{tag}_sm_{arch}', ['python3', str(root / 'build.py'), '--root', str(root),
                '--arch', str(arch), '--embedding', str(embedding), '--paged', str(paged),
                '--split', str(split)], [], 0 if arch == 89 else 200, 1800)
        build = f'build_{tag}_sm_89'
        binary = str(root / (tag + '-sm_89'))
        for tool in ['memcheck', 'racecheck']:
            step(f'{tool}_{tag}', ['/usr/local/cuda/bin/compute-sanitizer', '--tool', tool,
                '--error-exitcode', '86', binary], [build], 2 if tool == 'memcheck' else 3, 300)
        for process in range(args.processes):
            step(f'check_{tag}_{process:02}', [binary], [build], 10, 120)
    (root / 'queue').mkdir()
    (root / 'queue/queue_runtime.json').write_text(json.dumps(steps, indent=2) + '\n')
    print(json.dumps(dict(root=str(root), steps=len(steps), snapshot_files=len(inputs))), flush=True)


if __name__ == '__main__':
    main()
