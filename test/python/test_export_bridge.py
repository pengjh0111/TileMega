"""CI-1 transcription and constant-evaluation contracts."""
import base64
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import unittest

import torch
from tilemega.export_bridge import serialize
from tilemega.export_constants import argument


class Call(torch.nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, *inputs):
        return self.function(*inputs)


def export(function, inputs):
    return torch.export.export(Call(function), inputs, strict=False)


def literal(encoded):
    kind, value = encoded['t'], encoded['v']
    if kind == 'list':
        return [literal(item) for item in value]
    if kind == 'node':
        return ('node', value)
    return value


class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_recursive_types_and_nonfinite_json(self):
        values = [17, 0.5, True, 'tanh', None, [False, [2, 3]],
                  torch.bfloat16, torch.device('cpu'), torch.strided,
                  torch.channels_last, -math.inf, math.inf, math.nan]
        encoded = argument(values)
        self.assertEqual([v['t'] for v in encoded['v']],
                         ['int', 'float', 'bool', 'str', 'none', 'list',
                          'dtype', 'device', 'layout', 'memory_format',
                          'float', 'float', 'float'])
        self.assertEqual(literal(encoded)[:6], [17, .5, True, 'tanh', None, [False, [2, 3]]])
        self.assertEqual(literal(encoded)[-3:], ['-inf', '+inf', 'nan'])
        self.assertEqual(json.loads(json.dumps(encoded, allow_nan=False)), encoded)

    def test_operator_parameters_survive_json_roundtrip(self):
        x = torch.zeros(2, 4, 8, 8)
        w = torch.zeros(8, 4, 3, 3)
        q = torch.zeros(2, 2, 8, 4)
        cases = [
            ('conv2d.default', lambda a, b: torch.ops.aten.conv2d.default(a, b, None, [2, 1], [1, 2], [1, 1], 1), (x, w), {3: [2, 1], 4: [1, 2]}, {}),
            ('convolution.default', lambda a, b: torch.ops.aten.convolution.default(a, b, None, [2, 1], [1, 2], [1, 1], False, [0, 0], 1), (x, w), {3: [2, 1], 6: False, 7: [0, 0]}, {}),
            ('max_pool2d.default', lambda a: torch.ops.aten.max_pool2d.default(a, [3, 2], [2, 1], [1, 0], [1, 1], False), (x,), {1: [3, 2], 2: [2, 1], 3: [1, 0]}, {}),
            ('max_pool2d_with_indices.default', lambda a: torch.ops.aten.max_pool2d_with_indices.default(a, [3, 2], [2, 1], [1, 0], [1, 1], True), (x,), {1: [3, 2], 5: True}, {}),
            ('gelu.default', lambda a: torch.ops.aten.gelu.default(a, approximate='tanh'), (x,), {}, {'approximate': 'tanh'}),
            ('scaled_dot_product_attention.default', lambda a: torch.ops.aten.scaled_dot_product_attention.default(a, a, a, None, 0., False, scale=.5), (q,), {}, {'scale': .5}),
            ('hardtanh.default', lambda a: torch.ops.aten.hardtanh.default(a, 0., 6.), (x,), {1: 0., 2: 6.}, {}),
            ('layer_norm.default', lambda a: torch.ops.aten.layer_norm.default(a, [4, 8, 8], None, None, 1.7e-5), (x,), {1: [4, 8, 8], 4: 1.7e-5}, {}),
            ('pixel_shuffle.default', lambda a: torch.ops.aten.pixel_shuffle.default(a, 2), (x,), {1: 2}, {}),
            ('constant_pad_nd.default', lambda a: torch.ops.aten.constant_pad_nd.default(a, [1, 2, 3, 4], 0.), (x,), {1: [1, 2, 3, 4], 2: 0.}, {}),
            ('mean.dim', lambda a: torch.ops.aten.mean.dim(a, [2, 3], True, dtype=torch.float32), (x,), {1: [2, 3], 2: True}, {'dtype': 'torch.float32'}),
        ]
        for target, function, inputs, expected_args, expected_kwargs in cases:
            with self.subTest(target=target):
                document = json.loads(json.dumps(serialize(export(function, inputs)), allow_nan=False))
                node = next(n for n in document['nodes'] if n['target'] == 'aten.' + target)
                args = literal(node['args'])
                for index, value in expected_args.items():
                    self.assertEqual(args[index], value)
                for name, value in expected_kwargs.items():
                    self.assertEqual(literal(node['kwargs'][name]), value)

    def test_static_constants_exclude_tensor_values(self):
        class Constants(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(8))
                self.register_buffer('buffer', torch.zeros(8))

            def forward(self, x):
                return x + self.weight + self.buffer + torch.arange(8)
        program = torch.export.export(Constants(), (torch.ones(8),), strict=False)
        document = serialize(program)
        nodes = document['nodes']
        arange = next(n for n in nodes if n['target'] == 'aten.arange.default')
        self.assertEqual(struct.unpack('<8q', base64.b64decode(arange['constant']['data_base64'])), tuple(range(8)))
        for node in nodes:
            if node['op'] == 'placeholder' or node['target'] == 'aten.add.Tensor':
                self.assertNotIn('constant', node)

    def test_lifted_tensor_constant_is_preserved(self):
        program = export(lambda x: x + torch.tensor([2, 3]),
                         (torch.zeros(2, dtype=torch.int64),))
        before = {name: tensor.clone() for name, tensor in program.constants.items()}
        document = serialize(program)
        source = next(spec for spec in program.graph_signature.input_specs
                      if spec.kind.name == 'CONSTANT_TENSOR')
        node = next(n for n in document['nodes'] if n['name'] == source.arg.name)
        self.assertEqual(struct.unpack('<2q', base64.b64decode(node['constant']['data_base64'])), (2, 3))
        for name, value in before.items():
            self.assertTrue(torch.equal(value, program.constants[name]))

    def test_shape_fragment_binding_and_value_mask(self):
        class ShapeMask(torch.nn.Module):
            def forward(self, ids):
                mask = (torch.arange(128).view(1, 1, 1, 128) >= 0)
                return ids, mask.expand(ids.shape[0], 1, 128, 128)
        program = torch.export.export(ShapeMask(), (torch.ones(2, 128, dtype=torch.int64),),
            dynamic_shapes=({0: torch.export.Dim('batch', min=1, max=64)},), strict=False)
        symbol = str(next(iter(program.range_constraints)))
        unbound = next(n for n in serialize(program)['nodes'] if n['target'] == 'aten.expand.default')
        self.assertNotIn('constant', unbound)
        self.assertEqual(unbound['shape_constant']['symbols'], [symbol])
        self.assertTrue(any(n['target'] == 'aten.sym_size.int' for n in unbound['shape_constant']['fragment']))
        for batch in (1, 8, 32, 64):
            mask = next(n for n in serialize(program, {symbol: batch})['nodes'] if n['target'] == 'aten.expand.default')
            self.assertEqual(mask['constant']['shape'], [batch, 1, 128, 128])
            self.assertTrue(mask['constant']['all_true'])
        for bindings in ({symbol: 65}, {'unknown_symbol': 8}, {symbol: True}):
            with self.assertRaises(ValueError):
                serialize(program, bindings)
        class ValueMask(torch.nn.Module):
            def forward(self, x):
                return x.bool().view(x.shape[0], 1, 1, 128).expand(-1, 1, 128, 128)
        value = export(lambda x: ValueMask()(x), (torch.ones(2, 128),))
        node = next(n for n in serialize(value)['nodes'] if n['target'] == 'aten.expand.default')
        self.assertNotIn('constant', node)
        self.assertNotIn('shape_constant', node)

    def test_constant_element_limit(self):
        program = export(lambda x: (x, torch.ones(1024, 1024, dtype=torch.bool),
                                   torch.ones(1025, 1025, dtype=torch.bool),
                                   torch.ones((1 << 20) + 1, 0, dtype=torch.bool)), (torch.ones(1),))
        nodes = [n for n in serialize(program)['nodes'] if n['target'] == 'aten.ones.default']
        self.assertTrue(nodes[0]['constant']['all_true'])
        self.assertEqual(nodes[0]['constant']['elements'], 1 << 20)
        self.assertNotIn('constant', nodes[1])
        self.assertEqual(nodes[2]['constant']['elements'], 0)

    def test_legacy_fields_are_exact(self):
        reference = Path(os.environ.get('TILEMEGA_REFERENCE_ROOT', '/root/dm1_work/reference')) / 'python/tilemega/export_bridge.py'
        if not reference.exists():
            self.skipTest('requires the sealed baseline checkout')
        spec = importlib.util.spec_from_file_location('dm1_legacy_bridge', reference)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        program = export(lambda x: torch.nn.functional.gelu(x + 1e-6), (torch.ones(2, 8),))
        document = serialize(program)
        extra = {'args', 'kwargs', 'constant', 'shape_constant', 'constant_evaluation'}
        for node in document['nodes']:
            for key in extra:
                node.pop(key, None)
        self.assertEqual(json.dumps(document, indent=2), json.dumps(old.serialize(program), indent=2))


if __name__ == '__main__':
    unittest.main()
