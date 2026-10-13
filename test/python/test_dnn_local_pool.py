import unittest

import torch

from tilemega.dnn.local_pool_probe import PoolProbe


class LocalPoolTest(unittest.TestCase):
    def probe(self):
        return PoolProbe([dict(name='pool', target='aten.adaptive_avg_pool2d.default',
                               inputs=['fused_gate'], shape=['B', '3', '1', '1'])])

    def test_compares_with_native_input_not_original_oracle(self):
        p = self.probe()
        value = (torch.arange(48).reshape(2, 3, 4, 2) + 100).to(torch.bfloat16)
        expected = value.float().mean((-2, -1), keepdim=True).bfloat16().float()
        # Native buffer descriptor order need not follow graph order.
        p.observe('pool', expected)
        p.observe('fused_gate', value)
        rows = p.report()['operators'][0]['per_batch']
        self.assertTrue(all(r['rounded_disagreements'] == 0 for r in rows))
        self.assertGreater(rows[1]['mse_against_each_input_image'][0], 0)

    def test_detects_cross_image_addressing(self):
        p = self.probe()
        value = torch.stack((torch.ones(3, 2, 2), torch.full((3, 2, 2), 8.)))
        p.observe('fused_gate', value)
        p.observe('pool', value.mean((-2, -1), keepdim=True).flip(0))
        rows = p.report()['operators'][0]['per_batch']
        self.assertEqual([r['mse'] for r in rows], [49., 49.])
        self.assertEqual(rows[1]['mse_against_each_input_image'][0], 0)

    def test_unwritten_input_is_not_invented(self):
        p = self.probe()
        p.observe('pool', torch.ones(2, 3, 1, 1))
        self.assertFalse(p.report()['operators'][0]['observed'])

    def test_only_global_pool_is_admitted(self):
        p = PoolProbe([dict(name='pool', target='aten.adaptive_avg_pool2d.default',
                           inputs=['x'], shape=['B', '3', '2', '2'])])
        self.assertEqual(p.report()['operators'], [])


if __name__ == '__main__':
    unittest.main()
