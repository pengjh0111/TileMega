#!/usr/bin/env python3
"""Small upstream PyTorch graphs for generated DNN correctness checks."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn


class DenseResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(32, 32)
        self.first, self.last = nn.Linear(32, 64), nn.Linear(64, 32)

    def forward(self, x):
        x = self.projection(x.flatten(1))
        return self.last(torch.nn.functional.gelu(self.first(x))) + x


class ConvPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 32, 3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(32, 16)

    def forward(self, x):
        return self.head(self.pool(self.norm(self.conv(x)).relu()).flatten(1))


class DepthwisePointwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.depthwise = nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False)
        self.norm = nn.BatchNorm2d(16)
        self.pointwise = nn.Conv2d(16, 32, 1)

    def forward(self, x):
        return self.pointwise(self.norm(self.depthwise(x)).relu())


class PixelShuffle(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 64, 1)
        self.shuffle = nn.PixelShuffle(2)
        self.head = nn.Conv2d(16, 16, 1)

    def forward(self, x):
        return self.head(self.shuffle(self.conv(x)))


def models():
    from tilemega.dnn.models import nafnet
    from transformers import BertConfig, BertModel
    # Load the exact upstream NAFBlock registered by the existing NAFNet adapter.
    nafnet()
    from tilemega.dnn.models.nafnet_arch import NAFBlock
    config = BertConfig(hidden_size=256, intermediate_size=512, num_hidden_layers=1,
                        num_attention_heads=4, vocab_size=128, max_position_embeddings=192)
    config._attn_implementation = 'sdpa'
    return [('dense_residual', DenseResidual(), (2,32,1,1)),
            ('conv_pool', ConvPool(), (2,16,9,11)),
            ('depthwise_pointwise', DepthwisePointwise(), (2,16,9,11)),
            ('pixel_shuffle', PixelShuffle(), (2,16,9,11)),
            ('naf_block', NAFBlock(16), (2,16,9,11)),
            ('encoder_masked', BertModel(config), None)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--labels', help='export only these comma-separated pipeline names')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20261011)
    from tilemega.export_bridge import serialize
    entries = models()
    if args.labels:
        selected = set(args.labels.split(','))
        if not selected <= {name for name, _, _ in entries}:
            parser.error('unknown pipeline name')
        entries = [entry for entry in entries if entry[0] in selected]
    for name, module, shape in entries:
        out = args.out / name
        out.mkdir(parents=True, exist_ok=False)
        module = module.eval().bfloat16()
        batch = torch.export.Dim('batch', min=1, max=16)
        if shape is None:
            values = dict(input_ids=torch.zeros((2,128), dtype=torch.int64),
                          token_type_ids=torch.zeros((2,128), dtype=torch.int64),
                          attention_mask=torch.ones((2,128), dtype=torch.int64))
            program = torch.export.export(module, (), kwargs=values,
                dynamic_shapes={key:{0:batch} for key in values}, strict=False)
        else:
            program = torch.export.export(module, (torch.zeros(shape, dtype=torch.bfloat16),),
                dynamic_shapes=({0:batch},), strict=False)
        torch.export.save(program, out / 'exported_program.pt2')
        (out / 'bridge.json').write_text(json.dumps(serialize(program)) + '\n')
        (out / 'manifest.json').write_text(json.dumps(dict(model=name,
            source='small independent upstream PyTorch graph', seed=20261011)) + '\n')
        print('Exported ' + name, flush=True)


if __name__ == '__main__':
    main()
