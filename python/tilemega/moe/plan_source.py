"""Check that native MoE aliases and packed weights implement the selected pin."""
from __future__ import annotations

import re

from .cost_provenance import Files, need

CLASSES = ('qkv', 'o', 'router', 'gate_up', 'down', 'head')


def validate_geometry(source, pin, batch, prompt, *, routing_weights_fp32=True):
    config = pin['mx_moe_v2']
    threads = config['threads']
    need(batch in (1, 2, 4, 8, 16) and threads in (512, 576) and type(prompt) is bool
         and type(routing_weights_fp32) is bool,
         'unsupported native MoE geometry request')

    def constant(name, expected):
        found = re.findall(r'\b' + name + r'\s*=\s*([^,;]+)', source)
        need(found == [str(expected)], 'actual Plan constant differs: ' + name)

    for name, value in dict(kLayers=24, kHidden=2048, kIntermediate=768, kCapacity=1088,
            kBatch=batch, kThreads=threads, kGrid=104, kPastLo=0 if prompt else 64,
            kPastHi=63 if prompt else 1086, kQ=8, kHeadDim=128).items():
        constant(name, value)
    constant('kPrompt', 'true' if prompt else 'false')
    routing = 'true' if routing_weights_fp32 else 'false'
    constant('kRoutingFp32', routing)
    for stage, name in (('qkv', 'kQkvStreamK'), ('o', 'kOStreamK'), ('router', 'kRouterStreamK')):
        constant(name, 'true' if config[stage]['part'] == 'kStreamK' else 'false')

    def alias(name, expected):
        found = re.findall(r'\busing\s+' + name + r'\s*=\s*([^;]+);', source)
        need(len(found) == 1 and re.sub(r'\s+', '', found[0]) == expected,
             'actual native body alias differs: ' + name)

    alias('Binding', 'mx::MxMoeBindingV2<kLayers>')
    alias('Dispatch', 'MxMoeDispatchV2<kBatch,kHidden,kIntermediate,%d,%d,kRoutingFp32,kThreads>' %
          (config['gate_up']['tn'], config['down']['tn']))
    alias('Attn', 'MxAttentionV2<128,8,%d,true,%d,kCapacity>' %
          (config['b_min'], config['attn_prefetch']))
    alias('AttnShape', 'MxStaticAttentionShapeV2<kBatch,4,kThreads>')
    shapes = dict(qkv=(5120, 2048), o=(2048, 4096), router=(128, 2048),
        gate_up=(1536, 2048), down=(2048, 768), head=(151936, 2048))
    for stage, name in zip(CLASSES, ('Qkv', 'O', 'Router', 'GateUp', 'Down', 'Head')):
        c = config[stage]
        tn, wn, depth, part, prefetch = (c[key] for key in ('tn', 'wn', 'd', 'part', 'prefetch'))
        need(part in ('kStreamK', 'kTileRange'), 'unsupported partition in native pin')
        if stage in ('qkv', 'o'):
            body = 'MxGemm<%d,%d,%d,MxOpV2::%s,%s,MxPart::%s,%d>' % (
                tn, wn, depth, 'kStore' if stage == 'qkv' else 'kResidual',
                'true' if stage == 'qkv' else 'false', part, prefetch)
        elif stage == 'router':
            need(wn == 1, 'router wave groups differ from the native domain')
            body = 'MxRouterGemmV2<%d,%d,MxPart::%s,Dispatch,%d>' % (tn, depth, part, prefetch)
        elif stage in ('gate_up', 'down'):
            need(part == 'kTileRange', 'grouped native body must use TileRange')
            body = 'MxGroupedGemm<%d,%d,%d,MxOpV2::%s,%s,Chain,%s,%d>' % (
                tn, wn, depth, 'kSwiGLU' if stage == 'gate_up' else 'kStore',
                'true' if stage == 'gate_up' else 'false',
                'MxNoUnitCompletionV2' if stage == 'gate_up' else
                'MxMoeDownCompletionV2<%s>' % routing, prefetch)
        else:
            need(part == 'kTileRange', 'native head must use TileRange')
            body = 'MxHead<%d,%d,%d,%d>' % (tn, wn, depth, prefetch)
        alias(name, body)
        alias(name + 'Shape', 'MxStaticShapeV2<%d,%d,kBatch,kThreads>' % shapes[stage])
        if stage == 'down':
            alias('DownLast', body)
    return dict(batch=batch, threads=threads, prompt=prompt, classes=6,
        routing_weights_fp32=routing_weights_fp32,
        attention_b_min=config['b_min'], attention_prefetch=config['attn_prefetch'],
        not_resource_or_numerical_or_model_acceptance=True)


def validate_manifest(manifest, pin):
    config = pin['mx_moe_v2']
    need(manifest['classes'] == {name: config[name] for name in CLASSES},
         'manifest classes differ from the selected pin')
    for name in ('threads', 'b_min', 'attn_prefetch'):
        need(manifest[name] == config[name], 'manifest geometry differs: ' + name)
    expected = {'lm_head.weight.dn': 'head'}
    for layer in range(24):
        prefix = 'model.layers.%d.' % layer
        expected.update({
            'l%d.qkv.weight.dn' % layer: 'qkv', prefix + 'self_attn.o_proj.weight': 'o',
            prefix + 'mlp.gate.weight': 'router', prefix + 'mlp.experts.gate_up.packed': 'gate_up',
            prefix + 'mlp.experts.down.packed': 'down'})
    rows = [row for row in manifest['buffers'] if row.get('recipe', {}).get('kind') == 'tile_pages']
    need(len(rows) == 121 and {row['name'] for row in rows} == set(expected),
         'native packed-weight set differs from all24 layers and head')
    for row in rows:
        recipe = row['recipe']
        need(row['role'] == 'external' and row['dtype'] == 'bf16'
             and recipe['layout'] == 'xcore_mma' and recipe['tile_k'] == 32
             and recipe['tile_n'] == config[expected[row['name']]]['tn'],
             'packed-weight tile does not implement the selected body: ' + row['name'])
    return dict(packed_weight_buffers=121, all_selected_weight_tile_sizes_match=True,
        not_resource_or_numerical_or_model_acceptance=True)


def validate_body_snapshot(snapshot, closure):
    files = Files()
    files.walk(closure)
    names = set()
    for family in closure['families']:
        for header in family['headers']:
            path = files.check(header['adopted'])
            name = 'include/' + str(path).split('/include/', 1)[1]
            need(header['adopted']['sha256'] == header['frozen']['sha256']
                 and snapshot['source_files'].get(name) == header['frozen']['sha256'],
                 'compiled body header differs from the measured cost source: ' + name)
            names.add(name)
    need(bool(names), 'no measured body dependencies were checked')
    files.unchanged()
    return dict(compiled_measured_header_matches=len(names),
        not_resource_or_numerical_or_model_acceptance=True)
