#!/usr/bin/env python3
"""Freeze CG-generated CUDA and its runtime before correctness queue submission."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'framework').mkdir()
    for name in ['capture_macros_dm.py', 'identity_dm.py']:
        shutil.copy2(Path(__file__).with_name(name), root/'framework'/name)
    shutil.copy2(Path(__file__).with_name('check_generated_dnn.py'), root/'check.py')
    shutil.copy2(args.source, root/'generated.cu')
    for name in ['dnn_semantic_lifting_test.cpp', 'dnn_epilogue_semantics_test.cpp']:
        (root/'test/unit').mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo/'test/unit'/name, root/'test/unit'/name)
    for name in ['lib/Analysis/TaskStorage.cpp', 'lib/Frontend/DnnStorage.cpp',
                 'lib/Frontend/DnnSemanticLifting.cpp']:
        (root/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo/name, root/name)
    for name in ['tilemega/__init__.py', 'tilemega/serving/__init__.py', 'tilemega/serving/plan.py']:
        (root/'python'/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo/'python'/name, root/'python'/name)
    shutil.copytree(repo/'include', root/'include')
    for name in ['include', 'tools/util/include']:
        shutil.copytree(repo/'third_party/cutlass'/name, root/'third_party/cutlass'/name)
    support = ['lib/Target/TargetSpec.cpp', 'lib/Support/Json.cpp',
        'lib/Codegen/RuntimeTaskGraph.cpp', 'lib/Solver/PlanMaterialize.cpp',
        'lib/Dialect/CouplingGraph/PlacementPlan.cpp',
        'lib/Solver/BalancedPlacement.cpp', 'lib/Solver/ListScheduler.cpp']
    for name in support:
        (root/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo/name, root/name)
    inputs = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob('*') if path.is_file()}
    preparation = dict(source_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo,
        text=True).strip(), diff_sha256=hashlib.sha256(subprocess.check_output(
            ['git', 'diff', 'HEAD'], cwd=repo)).hexdigest(), support=support, inputs=inputs,
        codegen_test_sha256=hashlib.sha256((repo/'test/unit/dnn_semantic_lifting_test.cpp').read_bytes()).hexdigest())
    (root/'preparation.json').write_text(json.dumps(preparation, indent=2)+'\n')
    steps = []
    def step(name, command, after, priority, seconds):
        steps.append(dict(name=name, command=['flock', '/root/r14_work/gpu.lock',
            'timeout', str(seconds), *command], cwd=str(repo), gpu=False,
            after=after, priority=priority, timeout_s=seconds+600))
    for arch in [89, 80, 90, 100, 120]:
        step(f'build_sm_{arch}', ['python3', str(root/'check.py'), '--root', str(root),
            '--arch', str(arch)], [], 0 if arch==89 else 200, 1800)
    command = ['/root/dm1_work/venv-gpu/bin/python', str(root/'check.py'), '--root', str(root)]
    for process in range(50):
        step(f'check_{process:02}', command, ['build_sm_89' if process==0 else 'check_00'], 10, 300)
    for tool in ['memcheck', 'racecheck']:
        step(tool, ['/usr/local/cuda/bin/compute-sanitizer', '--tool', tool,
            '--target-processes', 'all', '--error-exitcode', '86', *command], ['build_sm_89'], 2, 1800)
    (root/'queue').mkdir()
    (root/'queue/queue_generated.json').write_text(json.dumps(steps, indent=2)+'\n')
    print(json.dumps(dict(root=str(root), steps=len(steps))), flush=True)
