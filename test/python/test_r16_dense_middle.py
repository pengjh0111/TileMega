# SPDX-License-Identifier: BSD-3-Clause
"""Middle GPU publication requires both qualified representative processes."""
import json
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from advance_r16_dense_middle import selected_rows,packing_pair,Campaign,read,family

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
    def test_blocked_cell_does_not_prevent_other_finalists(self):
        campaign=Campaign.__new__(Campaign);campaign.repo=self.root;campaign.out=self.root
        published=[]
        def publish(gpu,name,jobs):
            if 'first' in name:raise ValueError('one invalid immutable queue')
            published.append((name,jobs))
        campaign.publish=publish;cells=[];state={}
        for model in ['first','second']:
            c=dict(id='mma',model=model,batch=2,gpu=2,pilot_job=model+'_pilot',
                physical_device_expected=dict(uuid='u',pci='p',numa_node=0),python='python3',
                script='frozen.py',file='candidate.json',root=str(self.root),env={})
            cells.append(dict(model=model,batch=2,gpu=2,candidates=[c]));state[c['pilot_job']]=dict(status='done')
        with patch('advance_r16_dense_middle.inspect_measurement',return_value=dict(e2e_seconds=1.0)):
            campaign.finals(state,{},dict(cells=cells))
        self.assertEqual(len(published),1);self.assertIn('second',published[0][0]);self.assertEqual(len(published[0][1]),3)
        self.assertEqual([c['status']for c in read(self.root/'selection-report.json')['cells']],['blocked','pending'])
    def test_legacy_absent_inactive_fields_do_not_force_recompile(self):
        c=dict(decode='binary',arm='h1',mode='L2',loop=0)
        identity=dict(plan=dict(gemms=[dict(impl='wgmma')],nonpaged_la=False),defines={})
        with patch('advance_r16_dense_middle.native_identity',return_value=identity):old=family(c)[0]
        identity['plan'].update(activation_tma=0,qkv_epilogue=0,moe_activation_bulk=0,cluster_dim=1,combine='stage')
        with patch('advance_r16_dense_middle.native_identity',return_value=identity):self.assertEqual(old,family(c)[0])
        identity['plan']['activation_tma']=1
        with patch('advance_r16_dense_middle.native_identity',return_value=identity):self.assertNotEqual(old,family(c)[0])

if __name__=='__main__':unittest.main()
