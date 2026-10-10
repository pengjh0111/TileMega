#!/usr/bin/env python3
"""Freeze a dataset-level G-DNN check of an already identified library."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('library', 'export', 'bridge', 'data', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--model', choices=('resnet18', 'mbv1', 'mbv2', 'bert', 'nafnet'), required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--nafnet-weights', type=Path)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--mask', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    root = args.out.resolve()
    library = args.library.resolve()
    identity_path = Path(str(library) + '.identity.json')
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    identity = json.loads(identity_path.read_text())
    if identity['binary_sha256'] != sha(library):
        raise ValueError('library identity mismatch')
    root.mkdir(parents=True, exist_ok=False)
    (root / 'queue').mkdir()
    shutil.copytree(repo / 'python/tilemega', root / 'python/tilemega',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(args.bridge, root / 'bridge.json')
    inputs = {str(path): sha(path) for path in root.rglob('*') if path.is_file()}
    inputs.update({str(path): sha(path) for path in (library, identity_path,
        args.export.resolve() / 'checkpoint/model.safetensors')})
    checkpoint_files = []
    if args.checkpoint:
        checkpoint_files.extend(args.checkpoint.rglob('*') if args.checkpoint.is_dir()
                                else [args.checkpoint])
    if args.nafnet_weights:
        checkpoint_files.append(args.nafnet_weights)
    for path in checkpoint_files:
        if path.is_file():
            inputs[str(path.resolve())] = sha(path)
    preparation = dict(source_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        cwd=repo, text=True).strip(), inputs=inputs, artifact_id=identity['artifact_id'],
        scope='G-DNN dataset correctness; no performance measurements')
    (root / 'preparation.json').write_text(json.dumps(preparation, indent=2) + '\n')
    command = ['env', 'PYTHONPATH=' + str(root / 'python'),
        '/root/dm1_work/venv-gpu/bin/python', '-m', 'tilemega.dnn.check',
        '--library', str(library), '--export', str(args.export.resolve()),
        '--bridge', str(root / 'bridge.json'), '--data', str(args.data.resolve()),
        '--model', args.model, '--batch', str(args.batch), '--out', str(root / 'correctness.json')]
    if args.checkpoint:
        command += ['--checkpoint', str(args.checkpoint.resolve())]
    if args.nafnet_weights:
        command += ['--nafnet-weights', str(args.nafnet_weights.resolve())]
    if args.mask:
        command += ['--mask']
    step = dict(name='dataset', command=['flock', '/root/r14_work/gpu.lock',
        'timeout', '7200', *command], cwd=str(repo), gpu=False, after=[], priority=0,
        timeout_s=14400)
    (root / 'queue/queue_dataset.json').write_text(json.dumps([step], indent=2) + '\n')
    print(json.dumps(dict(root=str(root), artifact_id=identity['artifact_id'])), flush=True)


if __name__ == '__main__':
    main()
