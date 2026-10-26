"""CPU-only accounting checks; fixture costs are not hardware calibration."""
import unittest
from tilemega.bench.as2_native_prices import task_overheads,tile_body_prices
from tilemega.bench.as2_body_profile import logical_bytes


class AscendPriceAccounting(unittest.TestCase):
    def setUp(self):
        self.cost=dict(dispatcher_aic_ns=10.,dispatcher_aiv_ns=20.,cross_worker_ready_ns=3.)

    def stage(self,kind,variant=0,epilogue=0):
        fields=[0]*33;fields[17]=kind;fields[18]=epilogue;fields[31]=variant
        return fields

    def test_owner_and_participant_service_spans(self):
        for variant in (4,5,6):
            self.assertEqual(task_overheads(self.stage(13,variant),dict(internal_mix_coordination_included=True),self.cost),(10.,0.))
        self.assertEqual(task_overheads(self.stage(0,1,1),{},self.cost),(30.,6.))
        self.assertEqual(task_overheads(self.stage(0),{},self.cost),(10.,0.))
        self.assertEqual(task_overheads(self.stage(13,3),{},self.cost),(20.,0.))
        self.assertEqual(task_overheads(self.stage(1,3),{},self.cost),(20.,0.))

    def test_wrong_span_rejected_before_pricing(self):
        for stage,flag in [(self.stage(13,6),False),(self.stage(13,3),True),(self.stage(0,1,1),True)]:
            with self.assertRaises(ValueError):task_overheads(stage,dict(internal_mix_coordination_included=flag),self.cost)

    def test_split_and_combine_share_one_physical_traffic_budget(self):
        partial=self.stage(0,1);partial[20:26]=[64,2048,6144,3,2048,1]
        combine=partial.copy();combine[17]=6;combine[18]=1
        self.assertEqual(logical_bytes(partial,[],0,0),598016)
        self.assertEqual(logical_bytes(partial,[],191,0),598016)
        self.assertEqual(logical_bytes(combine,[],0,0),40960)
        partial[20]=combine[20]=1
        self.assertEqual(logical_bytes(partial,[],47,0),528896)
        self.assertEqual(logical_bytes(combine,[],15,0),2560)
        self.assertEqual(task_overheads(combine,{},self.cost),(20.,0.))
        with self.assertRaises(ValueError):logical_bytes(partial,[],48,0)
        partial[24]=1024
        with self.assertRaises(ValueError):logical_bytes(partial,[],0,0)
        partial[20]=64;partial[22]=8192;partial[24]=2731
        self.assertEqual(logical_bytes(partial,[],0,0),794720)
        self.assertEqual(logical_bytes(partial,[],191,0),794432)
        partial[21]=combine[21]=3072;combine[22]=8192;partial[23]=combine[23]=2
        partial[24]=combine[24]=4096;combine[20]=64
        self.assertEqual(logical_bytes(partial,[],191,0),1187840)
        self.assertEqual(logical_bytes(combine,[],95,0),32768)

    def test_separate_key_partials_store_once_and_pack_once(self):
        partial=self.stage(0,1);partial[20:26]=[64,1024,4096,7,586,1]
        combine=partial.copy();combine[17]=6
        self.assertEqual(logical_bytes(partial,[],0,0),176960)
        self.assertEqual(logical_bytes(partial,[],223,0),175232)
        self.assertEqual(logical_bytes(combine,[],31,0),61440)
        wrong=partial.copy();wrong[21]=2048
        with self.assertRaises(ValueError):logical_bytes(wrong,[],0,0)
        combine[18]=1
        with self.assertRaises(ValueError):logical_bytes(combine,[],0,0)
        pack=self.stage(25);pack[20]=64;pack[23:28]=[8,768,1,128,4]
        self.assertEqual(logical_bytes(pack,[],63,0),24576)
        with self.assertRaises(ValueError):logical_bytes(pack,[],64,0)

    def test_local_tail_partials_are_private_body_traffic(self):
        local=self.stage(0,2,1);local[20:24]=[1024,2048,8192,0]
        base=local.copy();base[31]=1
        for k in (6144,8192):
            local[22]=base[22]=k
            self.assertEqual(logical_bytes(local,[],13,0),logical_bytes(base,[],13,0))
            self.assertEqual(logical_bytes(local,[],14,0)-logical_bytes(base,[],14,0),56*16*128)
        local[22]=7168
        with self.assertRaises(ValueError):logical_bytes(local,[],14,0)
        local[22]=6144
        local[20]=64
        with self.assertRaises(ValueError):logical_bytes(local,[],14,0)

    def test_coordinate_prices_use_observed_regions_without_zero_fallback(self):
        stage=self.stage(0,2,1);stage[19:23]=[1024,1024,2048,8192]
        fixture=dict(body_service_regions=dict(regular=dict(tasks=896,median=10.),tail=dict(tasks=128,median=70.)))
        for k in (6144,8192):
            stage[22]=k
            result=tile_body_prices(stage,fixture)
            self.assertEqual([result[i] for i in (13,14,15,16)],[10.,70.,70.,10.])
            self.assertEqual(result.count(70.),128)
        stage[22]=7168
        with self.assertRaises(ValueError):tile_body_prices(stage,fixture)
        stage[22]=6144
        fixture['body_service_regions']['tail']['median']=0
        with self.assertRaises(ValueError):tile_body_prices(stage,fixture)


if __name__=='__main__':unittest.main()
