#!/usr/bin/env python3
"""Focused host checks for trace joins, barriers, and readonly pinning."""
import csv,json,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from ledger import stages,steps,write
from cm_report import fit
from fidelity import tau
from analyze import trace_v2
from tilemega.serving.execution import pin_prefill
class Diagnostics(unittest.TestCase):
    def test_stage_duration_uses_previous_latest_release(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)
            write(p/'runtime_stages.tsv',[dict(stage=0,kind_name='kGemm',name='l0.qkv',weight_bytes=1000,extent=0,width=32),dict(stage=2,kind_name='kGemm',name='l0.o',weight_bytes=2000,extent=0,width=32)])
            write(p/'stage_trace.tsv',[dict(iteration=0,stage=0,past=575,worker=0,t_begin=10,t_tasks_end=90,t_release=110,tasks=1),dict(iteration=0,stage=0,past=575,worker=1,t_begin=12,t_tasks_end=100,t_release=115,tasks=1),dict(iteration=0,stage=2,past=575,worker=0,t_begin=116,t_tasks_end=170,t_release=215,tasks=1)])
            rows=stages(p,1,{'test':100})
            self.assertEqual(rows[0]['duration_ns'],105);self.assertEqual(rows[1]['duration_ns'],100)
            self.assertEqual(rows[0]['tail_ns'],5)
    def test_actual_reducer_owner_join(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)
            fields=dict(slot=0,worker=1,stage=3,logical_task=4,smid=0,wait_begin=10,ready=20,run_begin=21,run_end=90,publish_end=95,wait_begin_idx=0,wait_count=1)
            write(p/'slots.tsv',[fields]);write(p/'waits.tsv',[dict(wait_index=0,event_index=7)])
            write(p/'events.tsv',[dict(event_index=7,publish_ns=19)])
            write(p/'reducers.tsv',[dict(stage=4,task=1,producer_stage=3,producer_task=4,worker=1,publish_ns=89)])
            task,hol,reducer=trace_v2(p)
            self.assertEqual(task[0]['wait_ns'],10);self.assertTrue(reducer[0]['owner_found']);self.assertEqual(reducer[0]['complete_ns'],89)
    def test_unidentifiable_byte_fit_is_explicit(self):
        self.assertFalse(fit([(100,3),(100,5)])['identifiable'])
        self.assertEqual(fit([(0,3),(10,23)])['byte_ns_bias'],2)
    def test_rank_tau(self):
        self.assertEqual(tau({1:1,2:2,3:3},{1:3,2:2,3:1}),-1)
    def test_prefill_pins_use_manifest_geometry(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp);manifest=p/'plan.json';classes=p/'classes.tsv'
            manifest.write_text(json.dumps(dict(gemms=[dict(index=0,tile_m=16,tile_n=128,tile_k=64,stages=4,split_k=1)],kappa=1,residency=1,attention_kv_block=1088,attention_query_rows=16)))
            classes.write_text('class\tgemm\n0\t0\n');options=pin_prefill(manifest,classes,p/'pin')
            cases=json.loads((p/'pin/prefill_cases.json').read_text())
            self.assertEqual(cases['cases'][0]['geometries'][0]['tile_n'],128)
            self.assertIn('--evaluate-configs',options)
if __name__=='__main__':unittest.main()
