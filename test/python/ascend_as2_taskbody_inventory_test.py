"""Source-derived work counts, not synthetic timing or NPU verification."""
import unittest
from tilemega.bench.as2_taskbody_inventory import stage_counts


class WorkLedger(unittest.TestCase):
    def stage(self,variant,kind=0,epilogue=2):
        s=[0]*33;s[17:23]=[kind,epilogue,96,16,12288,2048];s[31]=variant
        return s

    def test_strided_swiglu_removes_row_pair_calls_without_removing_tasks(self):
        old=stage_counts(self.stage(1));tile=stage_counts(self.stage(4))
        self.assertEqual(old['pointwise16_calls'],6144)
        self.assertEqual(tile['swiglu_tile_calls'],96)
        self.assertEqual(old['tasks'],tile['tasks'])
        self.assertEqual(old['mmad_object_lifetimes'],tile['mmad_object_lifetimes'])

    def test_lm_head_scans_are_labelled_upper_bounds_not_measured_work(self):
        s=self.stage(1,epilogue=3);s[19:23]=[1187,16,151936,1024]
        count=stage_counts(s)
        self.assertEqual(count['argmax_row_calls'],16*1187)
        self.assertEqual(count['argmax_scalar_scan_upper_bound_elements'],16*151936)
        s[31]=6
        self.assertEqual(stage_counts(s)['argmax_batched_load_calls'],1187)
        self.assertNotIn('argmax_scalar_scan_upper_bound_elements',stage_counts(s))
        s[17]=15;s[19]=16;s[24]=1187;s[31]=0
        self.assertEqual(stage_counts(s)['argmax_merge_scalar_reads'],16*1187)
        s[31]=1
        self.assertEqual(stage_counts(s)['argmax_merge_vector_calls'],16)

    def test_partial_and_combine_lifetimes_are_distinct(self):
        s=self.stage(1,epilogue=0);s[19:24]=[48,1,2048,6144,3]
        self.assertEqual(stage_counts(s)['mmad_object_lifetimes'],48)
        s[17:20]=[6,1,16]
        result=stage_counts(s)
        self.assertNotIn('mmad_object_lifetimes',result)
        self.assertEqual((result['combine_row_calls'],result['residual_row_calls']),(16,16))


if __name__=='__main__':unittest.main()
