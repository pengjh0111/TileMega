import unittest
from tilemega.dnn.debug_buffers import materialized_buffers
from tilemega.dnn.diagnose import reference_aliases


class DebugBuffersTest(unittest.TestCase):
    def fixture(self):
        source = '''constexpr GemmDesc kGemms[] = {
  {64, 32, 0u, 1u, kNoOperand, 2u, 0.0f, 0u, kNoOperand, kNoOperand, kNoOperand,
   {}, {0u, {}, 1u, {{static_cast<DmSideOutputKind>(0u), 6u, 4294967295u, 0u}}}},
};
constexpr StageDesc kStages[] = {
  {TaskKind::kGemm, 0u, 64u, 32u, 0u, {0u, 1u, 2u}},
  {TaskKind::kLayerNorm, 0u, 64u, 32u, 0u, {2u, 1u, 1u, 4u, kNoOperand}},
};'''
        manifest = dict(buffers=[dict(name=name, role='external' if i < 2 else 'internal')
            for i, name in enumerate(('input', 'weight', 'conv', 'folded_norm', 'norm', 'dead', 'stats'))])
        return source, manifest

    def test_dead_folded_norms_are_not_stage_outputs(self):
        result = materialized_buffers(*self.fixture())
        self.assertEqual(set(result), {'input', 'weight', 'conv', 'norm', 'stats'})
        self.assertEqual(result['conv']['writers'], [0])
        self.assertEqual(result['norm']['writers'], [1])
        self.assertEqual(result['stats']['writers'], [0])

    def test_unknown_contract_and_changed_output_index_fail_closed(self):
        source, manifest = self.fixture()
        for text in (source.replace('kLayerNorm', 'kUnknown'), source.replace('4u, kNoOperand', '90u, kNoOperand'),
                     source.replace('kStages[]', 'other[]')):
            with self.assertRaises(ValueError):
                materialized_buffers(text, manifest)

    def test_fused_parameter_scale_and_residual_use_final_output(self):
        nodes = [dict(name='input', target='input', inputs=[]),
                 dict(name='conv', target='aten.conv2d.default', inputs=['input']),
                 dict(name='scale', target='aten.mul.Tensor', inputs=['conv', 'beta']),
                 dict(name='residual', target='aten.add.Tensor', inputs=['input', 'scale'])]
        live = {'input', 'conv'}
        captured = {node['name'] for node in nodes}
        actual = reference_aliases(nodes, live, captured, {'beta'})
        self.assertEqual(actual, {'input': 'input', 'conv': 'residual'})
        # An activation/activation multiplication is not a folded weight scale.
        changed = [*nodes[:2], dict(nodes[2], inputs=['conv', 'activation'])]
        self.assertEqual(reference_aliases(changed, live, captured, {'beta'})['conv'], 'conv')


if __name__ == '__main__':
    unittest.main()
