import unittest
import torch
from tilemega.dnn.local_replay import local_replay


class LocalReplayTest(unittest.TestCase):
    def module(self):
        class Chain(torch.nn.Module):
            def forward(self, x):
                a=x*2
                b=a+1
                return b*3
        return torch.fx.symbolic_trace(Chain())

    def test_upstream_error_is_removed_from_the_next_operator(self):
        module=self.module();x=torch.ones(2,3)
        actual={'mul':torch.full_like(x,100), 'add':torch.full_like(x,101),
                'mul_1':torch.full_like(x,304)}
        report=local_replay(module,(x,),module,(x,),actual,'mul_1')
        rows={r['reference']:r['per_batch'] for r in report['boundaries']}
        self.assertEqual(rows['mul'][0]['tm_fp32_mse'],98**2)
        self.assertEqual(rows['add'][0]['tm_fp32_mse'],0)
        self.assertEqual(rows['mul_1'][1]['tm_fp32_mse'],1)

    def test_endpoint_stops_before_later_work(self):
        module=self.module();x=torch.ones(2,3)
        report=local_replay(module,(x,),module,(x,),{'mul':x*2},'mul')
        self.assertEqual(len(report['boundaries']),1)

    def test_shape_mismatch_and_unobserved_endpoint_are_rejected(self):
        module=self.module();x=torch.ones(2,3)
        with self.assertRaisesRegex(ValueError,'shape differs'):
            local_replay(module,(x,),module,(x,),{'mul':torch.ones(1,3)},'mul')
        with self.assertRaisesRegex(ValueError,'not a materialized'):
            local_replay(module,(x,),module,(x,),{},'add')


if __name__=='__main__':
    unittest.main()
