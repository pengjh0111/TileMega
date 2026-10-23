import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from refine_sm90r2_joint_cost import JointPrices, PooledPrices, fit_identifiable, physical_family


class IdentifiableFitTest(unittest.TestCase):
    def test_collinear_and_unobserved_costs_remain_at_prior(self):
        matrix = np.array([[1.,1.,0.],[2.,2.,0.],[3.,3.,0.]])
        prior = np.array([.2,.4,.7])
        result,audit = fit_identifiable(lambda x:matrix@x,np.array([1.,2.,3.]),
            np.ones(3),prior,np.zeros(3),np.ones(3))
        self.assertTrue(result.success,result.message)
        self.assertEqual(audit['rank'],1)
        self.assertEqual(audit['zero_sensitivity_parameters'],[2])
        np.testing.assert_allclose(result.x,[.4,.6,.7],atol=2e-6)
        self.assertAlmostEqual(result.x[0]-result.x[1],prior[0]-prior[1],places=8)

    def test_physical_bounds_survive_reduced_basis(self):
        result,audit = fit_identifiable(lambda x:np.array([x.sum()]),np.array([10.]),
            np.ones(1),np.zeros(2),np.zeros(2),np.array([.2,1.]))
        self.assertTrue(result.success,result.message)
        np.testing.assert_allclose(result.x,[.2,.2],atol=2e-6)
        self.assertLess(audit['maximum_bound_violation'],1e-7)

    def test_no_information_does_not_change_prices(self):
        prior = np.array([.4,.8])
        result,audit = fit_identifiable(lambda x:np.array([3.]),np.array([7.]),
            np.ones(1),prior,np.zeros(2),np.ones(2))
        self.assertEqual(audit['rank'],0)
        self.assertEqual(result.nfev,0)
        np.testing.assert_array_equal(result.x,prior)

    def test_pooling_shares_shapes_and_stages_without_cell_keys(self):
        parameters = [('gemm_mma_sync_tn8_32_wk4','memory_scale',1),
            ('gemm_mma_sync_tn72_plus_wk1','memory_scale',1),
            ('gemm_mma_sync_tn8_32_wk4:qkv_low','issue_ns',1000),
            ('gemm_mma_sync_tn72_plus_wk1:down_low','issue_ns',1000),
            ('gemm_mma_sync_tn72_plus_wk1:down_high','issue_ns',1000)]
        prices = SimpleNamespace(parameters=parameters,prior=np.array([.8,1.2,.2,.4,.8]),
            lower=np.zeros(5),upper=np.full(5,4.),predict=lambda x:x)
        pooled = PooledPrices(prices)
        self.assertEqual(len(pooled.groups),3)
        np.testing.assert_allclose(pooled.expand(pooled.prior),[1.,1.,.3,.3,.8])
        self.assertEqual(physical_family('attention_simt_d64_q4_norm0'),'native_attention_simt')
        self.assertNotEqual(physical_family('attention_simt_d64_q4_norm0'),
            physical_family('attention_mma_sync_d64_q4_norm0'))


class L2PriceTest(unittest.TestCase):
    def make_prices(self,l2_service,l2_efficiency):
        key='gemm_mma_sync_tn8_32_wk4';protocol=key+':qkv_low'
        stage=dict(key=key,protocol_key=protocol,kind='qkv',waves=1,barrier_ns=5.,startup_ns=15.,
            initialization_ns=2.,read_startup_ns=3.,flow_ns=80.,compute_ns=40.,
            epilogue_ns=10.,dram_ns=60.,stream_ns=80.,activation_ns=20.,
            overlap_limit_ns=43.,preissue_service_ns=10.,late_service_ns=5.,
            l2_service_ns=l2_service,issue_count=1.,issue_service_ns=0.,receive_count=1.)
        fits={key:dict(coefficients=dict(startup_ns=10.,memory_scale=1.,activation_scale=1.,mma_scale=1.)),
            protocol:dict(coefficients=dict(preissue_efficiency=.5,tail_efficiency=.4,
                issue_ns=1.,issue_service_scale=0.,receive_ns=2.,l2_efficiency=l2_efficiency))}
        for op in ['store','residual','swiglu','argmax']:
            fits['native_epilogue_'+op]=dict(coefficients=dict(scale=1.))
        return JointPrices([dict(terms=[dict(stage=stage,weight=1.)])],dict(body_fits=fits))

    def test_nonzero_l2_prior_uses_native_overlap_equation(self):
        prices=self.make_prices(20.,.5)
        # 110 ns body + 3 ns protocol - (5 early + 2 late + 10 L2).
        self.assertAlmostEqual(prices.predict(prices.prior)[0],96.)
        self.assertFalse(any(name=='l2_efficiency' for _,name,_ in prices.parameters))

    def test_l2_credit_cannot_exceed_receiver_overlap_limit(self):
        prices=self.make_prices(200.,1.)
        self.assertAlmostEqual(prices.predict(prices.prior)[0],70.)

    def attention_prices(self,flow,early,waves):
        key='attention_mma_sync_d128_q2_norm1';protocol=key+':attention_high'
        stage=dict(key=key,protocol_key=protocol,kind='attention',waves=waves,
            barrier_ns=0.,startup_ns=0.,initialization_ns=0.,read_startup_ns=0.,
            flow_ns=flow,compute_ns=flow,epilogue_ns=0.,dram_ns=0.,stream_ns=0.,
            activation_ns=0.,overlap_limit_ns=0.,preissue_service_ns=early,
            late_service_ns=0.,l2_service_ns=0.,issue_count=0.,issue_service_ns=0.,receive_count=1.)
        fits={key:dict(coefficients=dict(startup_ns=0.,body_scale=1.)),
            protocol:dict(coefficients=dict(preissue_efficiency=1.,tail_efficiency=0.,
                issue_ns=0.,issue_service_scale=0.,receive_ns=0.,l2_efficiency=0.))}
        for op in ['store','residual','swiglu','argmax']:
            fits['native_epilogue_'+op]=dict(coefficients=dict(scale=1.))
        return JointPrices([dict(terms=[dict(stage=stage,weight=1.)])],dict(body_fits=fits))

    def test_attention_prefix_does_not_discount_later_waves(self):
        prices=self.attention_prices(flow=1600.,early=800.,waves=16)
        self.assertAlmostEqual(prices.predict(prices.prior)[0],1500.)

    def test_attention_body_refit_does_not_multiply_transferred_bytes(self):
        prices=self.attention_prices(flow=1600.,early=32.,waves=16)
        index=next(i for i,(_,name,_) in enumerate(prices.parameters) if name=='body_scale')
        x=prices.prior.copy();x[index]=2.
        self.assertAlmostEqual(prices.predict(x)[0],3200.-32.)


if __name__=='__main__':
    unittest.main()
