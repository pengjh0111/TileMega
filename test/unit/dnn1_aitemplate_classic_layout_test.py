"""CPU value oracles for identity pooling and NCHW classifier flattening."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]


def function(path, name):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class Tensor:
    def __init__(self, data):
        self.data, self._attrs = data, {}

    def shape(self):
        return [SimpleNamespace(_attrs={'values': [size]}) for size in self.data.shape]


class Ops:
    @staticmethod
    def permute():
        return lambda value, axes: Tensor(value.data.permute(axes).contiguous())

    @staticmethod
    def reshape():
        return lambda value, shape: Tensor(value.data.reshape(shape))


class ClassicLayoutTest(unittest.TestCase):
    def setUp(self):
        self.identity = function(ROOT / 'python/tilemega/dnn/aitemplate_aten.py',
                                 'identity_adaptive_avg_pool')
        path = ROOT / 'python/tilemega/dnn/aitemplate_layout.py'
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        self.layout = dict(Tensor=Tensor, ops=Ops, AIT_CONVERTERS={}, torch=torch)
        exec(compile(tree, str(path), 'exec'), self.layout)

    def test_actual_VGG_and_Alexnet_spatial_pool_is_same_tensor_and_bits(self):
        for channels, size in [(512, 7), (256, 6)]:
            data = ((torch.arange(2*channels*size*size) % 251) - 125).to(torch.bfloat16)
            data = data.reshape(2, channels, size, size)
            value = self.layout['nhwc'](Tensor(data))
            def unexpected(*_):
                raise AssertionError('identity must not emit an avg_pool kernel')
            convert = self.layout['wrap']('aten._adaptive_avg_pool2d.default',
                                          self.identity(unexpected))
            actual = convert('aten._adaptive_avg_pool2d.default',
                             (value, [size, size]), {}, 'pool')
            self.assertIs(actual, value)
            logical = self.layout['nchw'](actual).data
            expected = torch.nn.functional.adaptive_avg_pool2d(data, (size, size))
            self.assertTrue(torch.equal(logical.view(torch.uint16), expected.view(torch.uint16)))

    def test_nonunit_spatial_classifier_flatten_keeps_NCHW_value_order(self):
        for channels, size in [(512, 7), (256, 6)]:
            original = ((torch.arange(2*channels*size*size) % 251)-125).to(torch.bfloat16)
            original = original.reshape(2, channels, size, size)
            physical = self.layout['nhwc'](Tensor(original))
            flatten = self.layout['wrap']('aten.view.default',
                lambda target, args, kwargs, name: Tensor(args[0].data.reshape(args[1])))
            actual = flatten('aten.view.default', (physical, [2, -1]), {}, 'flatten').data
            expected = original.flatten(1)
            self.assertTrue(torch.equal(actual.view(torch.uint16), expected.view(torch.uint16)))
            self.assertFalse(torch.equal(physical.data.flatten(1), expected))

    def test_identity_handles_rectangles_None_and_preserves_dtype(self):
        value = Tensor(torch.randn(2, 3, 5, 4, dtype=torch.bfloat16))
        convert = self.identity(None)
        self.assertIs(convert(None, (value, (3, None)), {}, 'pool'), value)
        self.assertEqual(value.data.dtype, torch.bfloat16)

    def test_nonidentity_delegates_without_changing_arguments(self):
        calls = []
        value = Tensor(torch.zeros(1, 7, 7, 4, dtype=torch.bfloat16))
        args = (value, (1, 1))
        fallback = lambda *values: calls.append(values) or 'original'
        self.assertEqual(self.identity(fallback)('target', args, {}, 'pool'), 'original')
        self.assertEqual(calls, [('target', args, {}, 'pool')])


if __name__ == '__main__':
    unittest.main()
