"""Actual typed chain order, tails and spatial residual classes are distinct."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('chain_probe', ROOT/'scripts/dnn1_chain_apply_probe.py')
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def step(kind, vector=0):
    return f'Sm90ForwardEpilogueStep<Sm90ForwardOp::k{kind},{vector},codegen::DmActivation::kRelu,codegen::DmGatePair::kSwiGLU,16,0,0>'


def source(body, vector1='kAbsent'):
    return f'''struct Bind_0 {{
 kVector0=1u,kVector1={vector1},kVector2=kAbsent,kVector3=kAbsent;
}};
struct Shape_0 {{
 kResidualStride=64,kOutputWidth=14,kOutputHeight=10,kChannels=64,kOutChannels=64;
}};
using Body_0={body};
'''


class ChainInventory(unittest.TestCase):
    def inventory(self, text):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/'plan.so.cu'
            path.write_text(text)
            return probe.inventory([path])

    def test_dense_tails_and_residual_null_mode_are_explicit(self):
        chain = 'Sm90ForwardEpilogueChain<64,64,0,1,0,0,0,'+step('Bias')+','+step('Residual',1)+'>'
        body = 'Sm90ForwardGemm<64,64,64,2,1,Sm90ForwardAMode::kDense,70,64,64,48,'+chain+'> '
        unscaled = self.inventory(source(body))
        scaled = self.inventory(source(body, '2u'))
        self.assertEqual({r['key']['live_rows'] for r in unscaled['keys']}, {6,64})
        self.assertEqual({r['key']['live_columns'] for r in unscaled['keys']}, {48})
        self.assertEqual({r['key']['vector_mask'] for r in unscaled['keys']}, {1})
        self.assertEqual({r['key']['vector_mask'] for r in scaled['keys']}, {3})
        self.assertTrue(all(r['key']['chain']=='bias0+residual1map0' for r in scaled['keys']))

    def test_spatial_tail_preserves_residual_row_mapping(self):
        chain = 'Sm90ForwardEpilogueChain<64,64,0,1,0,0,0,'+step('Bias')+','+step('Residual',1)+'>'
        result = self.inventory(source('Sm90ForwardDwPw<8,8,64,64,1,2,'+chain+'>'))
        classes = {(r['key']['live_rows'],r['key']['tile_width'],r['key']['residual_image_width']) for r in result['keys']}
        self.assertEqual(classes, {(12,6,14),(16,8,14),(48,6,14),(64,8,14)})

    def test_collective_gate_and_norm_are_excluded(self):
        for kind in ('GatePair','DeferredLayerNorm'):
            chain = 'Sm90ForwardEpilogueChain<64,64,0,1,0,0,0,'+step('Bias')+','+step(kind)+'>'
            result = self.inventory(source('Sm90ForwardGemm<64,64,64,2,1,Sm90ForwardAMode::kDense,64,64,64,64,'+chain+'>'))
            self.assertEqual(result['keys'], [])
            self.assertEqual(result['excluded_chain_instances'], {'k'+kind:1})


if __name__=='__main__':
    unittest.main()
