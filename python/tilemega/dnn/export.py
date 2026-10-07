"""DN-1: export upstream architectures without altering their forward code."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch

MODELS = ('resnet18', 'mbv1', 'mbv2', 'bert', 'nafnet')
DEFAULT_NAFNET = Path('/root/models/nafnet/NAFNet-SIDD-width32.pth')


def versions():
    return {name: importlib.metadata.version(name) for name in
            ('torch', 'torchvision', 'timm', 'transformers')}


def model(name, structure_only=False, nafnet_weights=DEFAULT_NAFNET):
    if name in ('resnet18', 'mbv2'):
        import torchvision.models as tv
        if name == 'resnet18':
            module = tv.resnet18(weights=None if structure_only else
                                 tv.ResNet18_Weights.IMAGENET1K_V1)
            source = 'torchvision/resnet18/IMAGENET1K_V1'
        else:
            module = tv.mobilenet_v2(weights=None if structure_only else
                                     tv.MobileNet_V2_Weights.IMAGENET1K_V2)
            source = 'torchvision/mobilenet_v2/IMAGENET1K_V2'
    elif name == 'mbv1':
        import timm
        source = 'mobilenetv1_100.ra4_e3600_r224_in1k'
        module = timm.create_model(source, pretrained=not structure_only)
    elif name == 'bert':
        from transformers import BertConfig, BertModel
        source = 'google-bert/bert-base-uncased'
        if structure_only:
            config = BertConfig()
            config._attn_implementation = 'sdpa'
            module = BertModel(config)
        else:
            module = BertModel.from_pretrained(source, attn_implementation='sdpa')
    elif name == 'nafnet':
        from .models import nafnet
        module = nafnet()
        source = str(nafnet_weights)
        if not structure_only:
            if not Path(nafnet_weights).is_file():
                raise FileNotFoundError('DN-1 requires the official NAFNet-SIDD-width32 '
                                        'checkpoint at ' + str(nafnet_weights))
            weights = torch.load(nafnet_weights, map_location='cpu', weights_only=True)
            module.load_state_dict(weights.get('params', weights), strict=True)
    else:
        raise ValueError(name)
    return module.eval().to(dtype=torch.bfloat16, device='cpu'), source


def inventory(program):
    counts = Counter(str(node.target) for node in program.graph.nodes
                     if node.op == 'call_function')
    return dict(sorted(counts.items()))


def conv_geometries(program):
    rows = []
    for node in program.graph.nodes:
        if node.op != 'call_function' or str(node.target) not in (
                'aten.conv2d.default', 'aten.convolution.default'):
            continue
        args = node.args
        x = args[0].meta['val']
        w = args[1].meta['val']
        y = node.meta['val']
        if str(node.target) == 'aten.convolution.default':
            stride, padding, dilation, groups = args[3], args[4], args[5], args[8]
        else:
            stride = args[3] if len(args) > 3 else [1, 1]
            padding = args[4] if len(args) > 4 else [0, 0]
            dilation = args[5] if len(args) > 5 else [1, 1]
            groups = args[6] if len(args) > 6 else 1
        rows.append(dict(node=node.name, input_shape=[str(v) for v in x.shape],
                         output_shape=[str(v) for v in y.shape],
                         weight_shape=[str(v) for v in w.shape], stride=list(stride),
                         padding=list(padding), dilation=list(dilation), groups=int(groups),
                         rows_per_batch=int(y.shape[2]) * int(y.shape[3]),
                         n=int(w.shape[0]), k=int(w.shape[1]) * int(w.shape[2]) * int(w.shape[3])))
    return rows


def export(name, out, fixtures, structure_only=False, mask=False,
           nafnet_weights=DEFAULT_NAFNET):
    torch.manual_seed(20261007)
    torch.set_num_threads(4)
    module, source = model(name, structure_only, nafnet_weights)
    batch = torch.export.Dim('batch', min=1, max=64)
    # Export specializes dimensions of size one, so the example must use B=2.
    if name == 'bert':
        kwargs = dict(input_ids=torch.zeros((2, 128), dtype=torch.int64),
                      token_type_ids=torch.zeros((2, 128), dtype=torch.int64))
        if mask:
            kwargs['attention_mask'] = torch.ones((2, 128), dtype=torch.int64)
        shapes = {key: {0: batch} for key in kwargs}
        with torch.no_grad():
            program = torch.export.export(module, (), kwargs=kwargs,
                                          dynamic_shapes=shapes, strict=False)
    else:
        side = 256 if name == 'nafnet' else 224
        inputs = (torch.zeros((2, 3, side, side), dtype=torch.bfloat16),)
        with torch.no_grad():
            program = torch.export.export(module, inputs, dynamic_shapes=({0: batch},),
                                          strict=False)
    core = program.run_decompositions()
    label = name + ('_masked' if mask else '')
    out = Path(out) / label
    out.mkdir(parents=True, exist_ok=True)
    torch.export.save(program, out / 'exported_program.pt2')
    torch.export.save(core, out / 'core_aten.pt2')
    fixture = dict(model=label, upstream=source, versions=versions(),
                   input_dtype='bf16 images / int64 ids', weight_dtype='bf16',
                   dynamic_batch=[1, 64], seq=128 if name == 'bert' else 1,
                   before=inventory(program), core_aten=inventory(core),
                   conv_geometries=conv_geometries(program),
                   weights='architecture initialization' if structure_only else 'pretrained',
                   evidence='verified')
    fixtures = Path(fixtures)
    fixtures.mkdir(parents=True, exist_ok=True)
    (fixtures / (label + '_ops.json')).write_text(json.dumps(fixture, indent=2) + '\n')
    manifest = dict(fixture, purpose='operator coverage' if structure_only else 'forward',
                    accuracy_eligible=not structure_only,
                    artifacts={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (out / 'exported_program.pt2', out / 'core_aten.pt2')})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(model=label, before=sum(fixture['before'].values()),
                          core_aten=sum(fixture['core_aten'].values()),
                          accuracy_eligible=not structure_only)), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS + ('all',), required=True)
    parser.add_argument('--out', type=Path, default=Path('runs/dm1-exports'))
    parser.add_argument('--fixtures', type=Path, default=Path('test/fixtures/dnn'))
    parser.add_argument('--structure-only', action='store_true',
                        help='Phase 0 inventories; ineligible for correctness/performance')
    parser.add_argument('--attention-mask', action='store_true')
    parser.add_argument('--nafnet-weights', type=Path, default=DEFAULT_NAFNET)
    args = parser.parse_args()
    for name in MODELS if args.model == 'all' else (args.model,):
        export(name, args.out, args.fixtures, args.structure_only,
               args.attention_mask and name == 'bert', args.nafnet_weights)


if __name__ == '__main__':
    main()
