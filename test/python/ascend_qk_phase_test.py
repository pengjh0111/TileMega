import unittest
from tilemega.bench.as2_precision import REPLAY,STREAMING_FP32,validate_plan_precision
from tilemega.bench.as2_body_profile import logical_bytes
from tilemega.bench.as2_native_prices import task_overheads


class QkPhaseContracts(unittest.TestCase):
    def stage(self):
        s=[0]*33;s[17]=13;s[19]=8;s[23:28]=[8,128,2,1088,2];s[31]=13
        return s

    def test_precision_and_rejected_shapes(self):
        s=self.stage();self.assertEqual(validate_plan_precision(REPLAY,[s]),1)
        with self.assertRaises(ValueError):validate_plan_precision(STREAMING_FP32,[s])
        for i,v in ((19,16),(23,4),(24,64),(25,3),(26,128),(27,4)):
            bad=list(s);bad[i]=v
            with self.assertRaises(ValueError):validate_plan_precision(REPLAY,[bad])

    def test_traffic_and_composite_price_preserve_old_contract(self):
        s=self.stage();old=list(s);old[31]=9;m=[0]*15;m[8]=1;m[9]=1
        for past in (95,319,799):
            self.assertEqual(logical_bytes(s,m,0,past),logical_bytes(old,m,0,past))
        cost=dict(dispatcher_aic_ns=10,dispatcher_aiv_ns=20,cross_worker_ready_ns=30)
        self.assertEqual(task_overheads(s,dict(internal_mix_coordination_included=True),cost),(10,0))
        with self.assertRaises(ValueError):task_overheads(s,dict(internal_mix_coordination_included=False),cost)


if __name__=='__main__':unittest.main()
