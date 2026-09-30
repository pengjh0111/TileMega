#!/usr/bin/env python3
import importlib.util,json,tempfile,unittest
from pathlib import Path
from choose_defaults_r12c import choose
from pin_case import pin
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
if __name__=='__main__':unittest.main()
