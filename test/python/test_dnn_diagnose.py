import copy
import unittest

from tilemega.build.identity import digest
from tilemega.dnn.diagnose import check_binding, reference_aliases, summarize


class DnnDiagnose(unittest.TestCase):
    def test_pixel_shuffle_residual_belongs_to_the_upsample_producer(self):
        def node(name, target, inputs=(), args=()):
            return dict(name=name,target=target,inputs=list(inputs),args=dict(t='list',v=list(args)))
        nodes=[node('skip_conv','aten.conv2d.default'),
               node('skip','aten.add.Tensor',['skip_conv','older_skip']),
               node('up_conv','aten.conv2d.default'),
               node('shuffle','aten.pixel_shuffle.default',['up_conv'],
                    [{},dict(t='int',v=2)]),
               node('sum','aten.add.Tensor',['shuffle','skip'])]
        self.assertEqual(reference_aliases(nodes,{'skip_conv','up_conv'},
                         {n['name'] for n in nodes}),
                         {'skip_conv':'skip','up_conv':'sum'})

    def test_simple_gate_maps_to_the_fused_half_width_output(self):
        def node(name, target, inputs=(), args=()):
            return dict(name=name,target=target,inputs=list(inputs),args=dict(t='list',v=list(args)))
        integer=lambda value:dict(t='int',v=value)
        nodes=[node('conv','aten.conv2d.default'),
               node('halves','aten.chunk.default',['conv'],[{},integer(2),integer(1)]),
               node('left','<built-in function getitem>',['halves'],[{},integer(0)]),
               node('right','<built-in function getitem>',['halves'],[{},integer(1)]),
               node('gate','aten.mul.Tensor',['left','right'])]
        self.assertEqual(reference_aliases(nodes,{'conv'},{'conv','gate'}),{'conv':'gate'})
        for change in ('same_half','batch_split','other_source','unrelated_multiply'):
            invalid=copy.deepcopy(nodes)
            if change=='same_half':invalid[3]['args']['v'][1]=integer(0)
            if change=='batch_split':invalid[1]['args']['v'][2]=integer(0)
            if change=='other_source':invalid[3]['inputs']=['different_chunk']
            if change=='unrelated_multiply':invalid[4]['inputs']=['conv','right']
            with self.subTest(change=change):
                self.assertEqual(reference_aliases(invalid,{'conv'},{'conv','gate'}),{'conv':'conv'})

    def fixture(self):
        bridge = dict(nodes=[dict(name='a', target='conv'), dict(name='b', target='gate')])
        def sample(batch, tm, bf16):
            return dict(batch=batch, tm_fp32_mse=tm, bf16_fp32_mse=bf16,
                        tm_bf16_mse=.01, tm_bf16_max_error=.2)
        data = dict(buffers=[dict(buffer='out_b', reference='b', per_batch=[
            sample(0, .1, .1), sample(1, 10, .1)]),
            dict(buffer='out_a', reference='a', per_batch=[sample(0, .1, .1), sample(1, .1, .1)])])
        return data, bridge

    def test_second_image_divergence_is_not_hidden_by_first_image(self):
        report = summarize(*self.fixture())
        self.assertIsNone(report['samples'][0]['first_observed_excess'])
        first = report['samples'][1]['first_observed_excess']
        self.assertEqual((first['reference'], first['operator'], first['graph_order']), ('b', 'gate', 1))
        self.assertAlmostEqual(first['excess_mse'], 9.85)

    def test_zero_bf16_error_never_hides_positive_tm_error(self):
        data, bridge = self.fixture()
        for sample in data['buffers'][0]['per_batch']:
            sample['bf16_fp32_mse'] = 0
        report = summarize(data, bridge)
        self.assertIsNotNone(report['samples'][0]['first_observed_excess'])
        self.assertIsNone(report['samples'][0]['first_observed_excess']['error_ratio'])

    def test_invalid_or_ambiguous_measurements_are_rejected(self):
        for change in ('negative', 'nonfinite', 'duplicate', 'unknown', 'shape', 'empty'):
            data, bridge = copy.deepcopy(self.fixture())
            if change == 'negative': data['buffers'][0]['per_batch'][0]['tm_fp32_mse'] = -1
            if change == 'nonfinite': data['buffers'][0]['per_batch'][0]['tm_fp32_mse'] = float('nan')
            if change == 'duplicate': data['buffers'].append(copy.deepcopy(data['buffers'][0]))
            if change == 'unknown': data['buffers'][0]['reference'] = 'missing'
            if change == 'shape': data['buffers'][0]['shape_mismatch'] = True
            if change == 'empty': data['buffers'] = []
            with self.subTest(change=change), self.assertRaises(ValueError):
                summarize(data, bridge)

    def test_first_divergence_cannot_be_joined_to_another_artifact_or_graph(self):
        data, bridge = self.fixture()
        identity = dict(artifact_id='artifact', binary_sha256='binary')
        data.update(identity, bridge_digest=digest(bridge))
        check_binding(data, bridge, identity)
        for key in ('artifact_id', 'binary_sha256', 'bridge_digest'):
            wrong = dict(data, **{key: 'changed'})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'not bound'):
                check_binding(wrong, bridge, identity)


if __name__ == '__main__':
    unittest.main()
