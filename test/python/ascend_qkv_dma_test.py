import unittest
from tilemega.bench.as2_request import gemm_variant_geometry
from tilemega.bench.as2_native_prices import task_overheads


class QkvDmaContracts(unittest.TestCase):
    def stage(self):
        s=[0]*33
        s[17:24]=[0,0,32,1,4096,1024,0]
        s[31]=11
        return s

    def test_geometry_and_invalid_payload_consumers(self):
        s=self.stage()
        self.assertEqual(gemm_variant_geometry(s),1)
        for index,value in ((18,1),(19,31),(20,16),(21,2048),(22,4096),(23,2),(0,(1<<64)-1)):
            changed=list(s);changed[index]=value
            with self.assertRaises(ValueError):gemm_variant_geometry(changed)

    def test_legacy_prices_cannot_price_dma_policy(self):
        cost=dict(dispatcher_aic_ns=10,dispatcher_aiv_ns=20,cross_worker_ready_ns=30)
        profile=dict(internal_mix_coordination_included=False)
        with self.assertRaisesRegex(ValueError,'measured dispatcher policy'):
            task_overheads(self.stage(),profile,cost)
        cost.update(dispatcher_dma_aic_ns=7,
            dma_dispatch_policy='first_acquire_32_task_immutable_POD_DMA_DSB_publication_v1')
        self.assertEqual(task_overheads(self.stage(),profile,cost),(7,0))
        control=self.stage();control[31]=1
        self.assertEqual(task_overheads(control,profile,cost),(10,0))
        for value in (None,0,-1,float('nan')):
            cost['dispatcher_dma_aic_ns']=value
            with self.assertRaises(ValueError):task_overheads(self.stage(),profile,cost)


if __name__=='__main__':unittest.main()
