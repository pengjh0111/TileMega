#!/usr/bin/env python3
import importlib.util,json,tempfile,unittest
from pathlib import Path
from choose_defaults_r12c import choose
from pin_case import pin
from fixed_builds import trace_command
class Tools(unittest.TestCase):
    def test_defaults(self):
        samples={b:{'P-base':[10,10,10],'P-D64K':[9.8]*3,'P-D128K':[9.7]*3,'P-noWD':[9.8]*3} for b in ('1','16')}
        result=choose(samples);self.assertEqual(result['lookahead_bytes'],131072);self.assertEqual(result['watchdog'],0)
        samples['16']['P-D128K']=[10]*3;self.assertEqual(choose(samples)['lookahead_bytes'],65536)
    def test_pin_uses_manifest_not_predicted_geometry(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);m=d/'m.json';c=d/'classes.tsv'
            m.write_text(json.dumps(dict(gemms=[dict(index=0,tile_m=16,tile_n=64,tile_k=64,stages=2,split_k=4)],kappa=1,residency=1,grid=128,attention_kv_block=256,attention_query_rows=4)))
            c.write_text('class\tgemm\top\ttile_m\ttile_n\ttile_k\tstages\tsplit_k\n0\t0\tqkv\t16\t128\t128\t8\t8\n')
            pin(m,c,c,d/'out');case=json.loads((d/'out/cases.json').read_text())['cases'][0];self.assertEqual(case['geometries'][0]['tile_n'],64)
    def test_dn_repartition_maps_geometry_by_gemm_index(self):
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);m=d/'m.json';source=d/'source.tsv';target=d/'target.tsv'
            geometry=dict(tile_m=16,tile_n=64,tile_k=64,stages=2,split_k=4)
            data=dict(gemms=[dict(index=i,**geometry) for i in range(2)],kappa=1,residency=1,grid=128,attention_kv_block=256,attention_query_rows=4)
            m.write_text(json.dumps(data));source.write_text('class\tgemm\n0\t0\n1\t1\n');target.write_text('class\tgemm\n0\t0\n0\t1\n')
            record=pin(m,source,target,d/'out');self.assertEqual(record['expected_partition'],[[0,1]])
            data['gemms'][1]['split_k']=8;m.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'geometry differs within class'):
                pin(m,source,target,d/'out2')
    def test_trace_output_can_be_a_reused_candidate(self):
        cmd=trace_command('nvcc -shared /tmp/plan.top1.candidate.so.cu -o /tmp/plan.top1.candidate.so -arch=sm_89','/tmp/trace/plan.so')
        self.assertEqual(cmd[cmd.index('-o')+1],'/tmp/trace/plan.so')
        self.assertIn('/tmp/plan.top1.candidate.so.cu',cmd)
        self.assertIn('-DTILEMEGA_TRACE_V2=1',cmd)

    def test_queue_dependencies_and_timing_guards(self):
        root=Path(__file__).resolve().parent
        steps=json.loads((root/'definitions/queue_a.json').read_text())+json.loads((root/'definitions/queue_b.json').read_text())
        names={s['name'] for s in steps};self.assertEqual(len(names),len(steps))
        pending={s['name']:set(s.get('after',[])+s.get('after_any',[])) for s in steps}
        for deps in pending.values():self.assertTrue(deps<=names)
        while pending:
            ready={n for n,deps in pending.items() if not deps}
            self.assertTrue(ready,'dependency cycle')
            pending={n:deps-ready for n,deps in pending.items() if n not in ready}
        for s in steps:
            self.assertGreater(s['timeout_s'],0)
            if not s['gpu'] and s['name'].startswith('B'):
                self.assertEqual(s['command'][:2],['flock','/root/r12c_work/gpu.lock'])
            if s['name'].startswith(('A1_','A2_','A3_','A4_','B1_','B3_','B5_','B8_')):
                self.assertTrue(s['gpu'])
if __name__=='__main__':unittest.main()
