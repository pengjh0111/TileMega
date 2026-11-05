"""CPU fake-Model ABI checks; no tensor allocation or CUDA initialization."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import dnn1_aitemplate as runner


class OutputsTest(unittest.TestCase):
    def allocate(self, *, dtype=6, shape=(1,1000), names=None):
        engine = SimpleNamespace(get_output_name_to_index_map=lambda: names or {'output_0':0},
            get_output_dtype=lambda _:dtype, get_output_maximum_shape=lambda _:list(shape))
        calls = []
        torch = SimpleNamespace(empty=lambda shape,**kwargs:calls.append((shape,kwargs)) or 'buffer')
        result = runner.allocate_outputs(engine,
            [dict(name='output_0',shape=[1,1000],dtype='bfloat16')], torch,
            lambda name:{'bfloat16':6}[name], lambda name:'torch.'+name)
        return result,calls

    def test_official_integer_BF16_enum_uses_IR_string_and_matches_names_shapes(self):
        result,calls = self.allocate()
        self.assertEqual(result,{'output_0':'buffer'})
        self.assertEqual(calls,[([1,1000],dict(device='cuda',dtype='torch.bfloat16'))])

    def test_dtype_mismatch_cannot_be_silently_cast(self):
        with self.assertRaisesRegex(ValueError,'dtype differs'):
            self.allocate(dtype=1)

    def test_shape_or_output_name_mismatch_fails_before_allocation(self):
        with self.assertRaisesRegex(ValueError,'shape differs'):
            self.allocate(shape=(16,1000))
        with self.assertRaisesRegex(ValueError,'names differ'):
            self.allocate(names={'unexpected':0})


if __name__ == '__main__':
    unittest.main()
