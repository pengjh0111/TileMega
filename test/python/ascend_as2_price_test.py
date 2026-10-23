"""CPU-only accounting checks; fixture costs are not hardware calibration."""
import unittest
from tilemega.bench.as2_native_prices import task_overheads
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


if __name__=='__main__':unittest.main()
