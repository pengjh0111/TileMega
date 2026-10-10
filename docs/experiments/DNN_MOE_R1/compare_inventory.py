#!/usr/bin/env python3
"""Compare observed counts to the unchanged DM-1 Appendix C assertions."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BEFORE = {
    'resnet18': dict(conv2d=20, batch_norm=20, relu_=17, add_=8,
                     max_pool2d=1, adaptive_avg_pool2d=1, flatten=1, linear=1),
    'mbv2': dict(conv2d=52, batch_norm=52, hardtanh_=35, add=10,
                 dropout=1, adaptive_avg_pool2d=1, flatten=1, linear=1),
    'mbv1': dict(conv2d=27, batch_norm=27, hardtanh_=27,
                 adaptive_avg_pool2d=1, flatten=1, linear=1),
    'nafnet': dict(conv2d=226, adaptive_avg_pool2d=36, chunk=72, pixel_shuffle=4,
                   pad=1, mean=144, sub=144, pow=72, sqrt=72, div=72,
                   mul=252, add=221, view=144, getitem=144),
    'bert': dict(embedding=3, layer_norm=25, linear=73, gelu=12,
                 scaled_dot_product_attention=12, tanh=1, dropout=25,
                 transpose=48, view=36, reshape=12, add=28),
}
CORE = {
    'resnet18': dict(convolution=20, _native_batch_norm_legit_no_training=20,
                     getitem=21, relu=17, add=8, max_pool2d_with_indices=1,
                     mean=1, addmm=1),
    'mbv2': dict(convolution=52, _native_batch_norm_legit_no_training=52,
                 hardtanh=35, add=10, mean=1, addmm=1),
    'mbv1': dict(convolution=27, _native_batch_norm_legit_no_training=27,
                 hardtanh=27, mean=1, addmm=1),
    'nafnet': dict(convolution=226, mean=180, split_with_sizes=72,
                   constant_pad_nd=1, permute=4, clone=8, view=152,
                   sub=144, pow=72, sqrt=72, div=72, mul=252, add=221, getitem=144),
    'bert': dict(addmm=73, bmm=24, _softmax=12, native_layer_norm=25,
                 gelu=12, tanh=1, permute=157, view=264),
}


def canonical(target):
    if target == '<built-in function getitem>':
        return 'getitem'
    return target.split('.')[1] if target.startswith('aten.') else target


def main():
    rows = []
    for model in BEFORE:
        fixture = json.loads((ROOT / f'test/fixtures/dnn/{model}_ops.json').read_text())
        for spelling, expected in [('before', BEFORE[model]), ('core_aten', CORE[model])]:
            counts = {}
            for target, count in fixture[spelling].items():
                name = canonical(target)
                counts[name] = counts.get(name, 0) + count
            differences = {key: dict(stated=want, observed=counts.get(key, 0))
                           for key, want in expected.items() if counts.get(key, 0) != want}
            rows.append(dict(model=model, spelling=spelling, evidence='verified',
                             differences=differences, observed=counts))
    output = HERE / 'results/T2_inventory_comparison.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(dict(models=5, spellings=10,
                          different_counts=sum(len(r['differences']) for r in rows))))


if __name__ == '__main__':
    main()
