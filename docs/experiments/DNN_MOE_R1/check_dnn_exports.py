#!/usr/bin/env python3
"""Verify original-FQN checkpoints against both real-weight exported archives."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import modeling_outputs  # Registers HF output pytrees before archive load.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    preparation = json.loads((root/'preparation.json').read_text())
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for path, expected in {**preparation['assets'], **preparation['inputs']}.items():
        if sha(path) != expected:
            raise ValueError('real export input changed: '+path)
    models = []
    for name in ['resnet18', 'mbv1', 'mbv2', 'bert', 'bert_masked', 'nafnet']:
        directory = root/'exports'/name
        manifest = json.loads((directory/'manifest.json').read_text())
        prior = json.loads((args.fixtures/(name+'_ops.json')).read_text())
        current = json.loads((root/'fixtures'/(name+'_ops.json')).read_text())
        assert manifest['accuracy_eligible'] and current['weights']=='pretrained'
        assert current['before']==prior['before'] and current['core_aten']==prior['core_aten']
        checkpoint = manifest['checkpoint']
        assert sha(checkpoint['path']) == checkpoint['sha256']
        tensors = load_file(checkpoint['path'])
        assert set(tensors)==set(checkpoint['tensors'])
        archives = []
        for label in ['exported_program.pt2', 'core_aten.pt2']:
            path = directory/label
            assert sha(path)==manifest['artifacts'][label]
            program = torch.export.load(path)
            assert set(program.state_dict)<=set(tensors)
            for fqn, tensor in program.state_dict.items():
                assert tensor.dtype==tensors[fqn].dtype and torch.equal(tensor, tensors[fqn]), fqn
            archives.append(dict(path=str(path), sha256=sha(path),
                                 state_tensors=len(program.state_dict)))
        models.append(dict(model=name, checkpoint=checkpoint, archives=archives,
                           operator_inventories_unchanged=True, all_archive_state_tensors_equal=True))
    receipt = dict(evidence='verified', passed=True, scope=preparation['scope'],
        preparation=preparation, models=models, model_correctness_gate=False)
    (root/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(models=len(models), archives=12, passed=True)), flush=True)


if __name__ == '__main__':
    main()
