import unittest
from tilemega.bench.as2_request import gemm_variant_geometry
from tilemega.bench.as2_body_profile import logical_bytes
from tilemega.bench.as2_native_prices import task_overheads


class BatchedResidualContracts(unittest.TestCase):
    def stage(self):
        s=[0]*33
        s[16]=512;s[17:23]=[0,1,8,16,1024,2048];s[31]=12
        return s

    def test_geometry_and_rejected_recipes(self):
        s=self.stage()
        for k in (2048,3072):
            s[22]=k;self.assertEqual(gemm_variant_geometry(s),1)
        for i,v in ((18,2),(19,7),(20,1),(21,2048),(22,4096),(23,3),
                    (16,(1<<64)-1),(2,(1<<64)-1),(3,(1<<64)-1)):
            bad=list(s);bad[i]=v
            with self.assertRaises(ValueError):gemm_variant_geometry(bad)

    def test_traffic_and_completion_contract_unchanged(self):
        s=self.stage();old=list(s);old[31]=1
        m=[0]*15;m[8]=16;m[9]=1
        self.assertEqual(logical_bytes(s,m,0,95),logical_bytes(old,m,0,95))
        costs=dict(dispatcher_aic_ns=10,dispatcher_aiv_ns=20,cross_worker_ready_ns=30)
        for stage in (s,old):
            self.assertEqual(task_overheads(stage,dict(internal_mix_coordination_included=False),costs),(30,60))


if __name__=='__main__':unittest.main()
