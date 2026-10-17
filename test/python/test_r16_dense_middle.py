# SPDX-License-Identifier: BSD-3-Clause
"""Middle GPU publication requires both qualified representative processes."""
import json
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from advance_r16_dense_middle import selected_rows,packing_pair

class DenseMiddleTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def sample(self,batch,rnd,token=3):
        d=self.root/f'B{batch}-{rnd}';d.mkdir();(d/'tokens.json').write_text(json.dumps([[token]*1024]*batch))
        p=d/'measurements.json';p.write_text(json.dumps(dict(generation_runs=[dict(N=1024,run=3,tokens_file='tokens.json')])));return dict(pid=batch*100+rnd,measurement=str(p))
    def cell(self,batch):
        return dict(model='m',batch=batch,status='complete',selected='h1',candidates=[dict(candidate='h1',samples=[self.sample(batch,r)for r in range(3)])])
    def test_one_representative_cannot_release_middle(self):
        report=dict(cells=[self.cell(1)])
        self.assertIsNone(selected_rows(report,'m'))
        report['cells'].append(self.cell(16));self.assertEqual(len(selected_rows(report,'m')),2)
        report['cells'][1]['status']='pending';self.assertIsNone(selected_rows(report,'m'))
    def test_disagreeing_final_tokens_reject_representative(self):
        report=dict(cells=[self.cell(1),self.cell(16)])
        p=Path(report['cells'][0]['candidates'][0]['samples'][2]['measurement']).parent/'tokens.json';p.write_text(json.dumps([[9]*1024]))
        with self.assertRaisesRegex(ValueError,'disagree on tokens'):selected_rows(report,'m')
    def test_phase_packing_mismatch_cannot_reuse_binary(self):
        paths=[self.root/'prefill.so',self.root/'decode.so'];b=dict(name='w',dtype='bf16',constant=16,recipe=dict(kind='tile_pages',tile_n=128,tile_k=64))
        for p in paths:Path(str(p)+'.plan.json').write_text(json.dumps(dict(buffers=[b])))
        with patch('advance_r16_dense_middle.native_identity'):
            self.assertEqual(packing_pair(*paths)['shared_names'],1)
            bad=dict(b,recipe=dict(kind='tile_pages',tile_n=64,tile_k=64));Path(str(paths[1])+'.plan.json').write_text(json.dumps(dict(buffers=[bad])))
            with self.assertRaisesRegex(ValueError,'packing conflict'):packing_pair(*paths)

if __name__=='__main__':unittest.main()
