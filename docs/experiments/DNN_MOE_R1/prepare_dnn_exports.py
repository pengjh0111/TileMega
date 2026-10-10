#!/usr/bin/env python3
"""Prepare real-weight DNN exports with frozen code and official local assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(repo / 'python', root / 'python', ignore=shutil.ignore_patterns('__pycache__'))
    checkpoint = dict(resnet18='/root/.cache/tilemega-dm/torch/hub/checkpoints/resnet18-f37072fd.pth',
        mbv2='/root/.cache/tilemega-dm/torch/hub/checkpoints/mobilenet_v2-7ebf99e0.pth',
        mbv1='/root/models/mobilenetv1_100', bert='/root/models/bert_base_uncased')
    files = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
             for name in [checkpoint['resnet18'], checkpoint['mbv2'],
                 checkpoint['mbv1']+'/model.safetensors', checkpoint['bert']+'/model.safetensors',
                 '/root/models/nafnet/NAFNet-SIDD-width32.pth']}
    preparation = dict(evidence='stated', source_head=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        assets=files, inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*.py')}, scope='Official BF16 upstream exports and original-FQN safetensors; no model correctness or timing claim')
    (root / 'preparation.json').write_text(json.dumps(preparation, indent=2)+'\n')
    steps = []
    for label in ['resnet18', 'mbv1', 'mbv2', 'bert', 'bert_masked', 'nafnet']:
        model = label.replace('_masked', '')
        command = ['env', 'PYTHONPATH='+str(root/'python'), 'HF_HUB_OFFLINE=1',
            '/root/dm1_work/venv/bin/python', '-m', 'tilemega.dnn.export',
            '--model', model, '--out', str(root/'exports'), '--fixtures', str(root/'fixtures')]
        if model in checkpoint:
            command += ['--checkpoint', checkpoint[model]]
        if label.endswith('_masked'):
            command += ['--attention-mask']
        steps.append(dict(name='export_'+label, command=command, cwd=str(repo), gpu=False,
                          timeout_s=1800, priority=0))
    (root/'queue').mkdir()
    (root/'queue/queue_exports.json').write_text(json.dumps(steps, indent=2)+'\n')
    print(json.dumps(dict(root=str(root), steps=len(steps))), flush=True)


if __name__ == '__main__':
    main()
