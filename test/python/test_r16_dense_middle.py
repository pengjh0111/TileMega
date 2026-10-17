# SPDX-License-Identifier: BSD-3-Clause
"""Middle GPU publication requires both qualified representative processes."""
import json
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from advance_r16_dense_middle import selected_rows,packing_pair,Campaign,read,family,body_signature,numeric_reference,file_sha,selection_pool_current

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
    def test_stale_selection_pool_cannot_release_middle(self):
        path=self.root/'plan.json';path.write_text('{}')
        report=dict(plans={str(path.resolve()):file_sha(path)})
        self.assertTrue(selection_pool_current(report,[path]))
        path.write_text('{"new_executor_probe":true}')
        self.assertFalse(selection_pool_current(report,[path]))
        self.assertFalse(selection_pool_current({},[path]))
    def test_executor_axis_reuses_body_but_preserves_transport(self):
        signature=dict(arm='h1',mode='L2',loop=0,coordinates=dict(activation_tma=1,cluster_dim=1))
        other=dict(signature,mode='L1',loop=1)
        self.assertEqual(body_signature(signature),body_signature(other))
        other['coordinates']=dict(activation_tma=0,cluster_dim=1)
        self.assertNotEqual(body_signature(signature),body_signature(other))
    def test_executor_reference_requires_hash_and_same_binary(self):
        prompt=self.root/'prompt.json';prompt.write_text('[]')
        reference=dict(model='m',batch=1,config_sha256='c',seed=0,prompt_ids=str(prompt),prefill='p',decode='d')
        path=self.root/'candidate.json';path.write_text(json.dumps(reference))
        candidate=dict(reference,numeric_reference_candidate=dict(path=str(path),sha256=file_sha(path)))
        with patch('advance_r16_dense_middle.native_identity',side_effect=lambda p:dict(artifact_id=p)):
            self.assertEqual(numeric_reference(candidate),reference)
            candidate['decode']='other'
            with self.assertRaisesRegex(ValueError,'same binary pair'):numeric_reference(candidate)
            candidate['decode']='d';path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'candidate changed'):numeric_reference(candidate)
    def test_unfinished_executor_probe_blocks_middle(self):
        campaign=Campaign.__new__(Campaign);campaign.rep=self.root
        cells=[dict(model='m',batch=b,candidates=[dict(execution_probe=True,pilot_job=f'probe{b}')])for b in (1,16)]
        (self.root/'qwen06-plan.json').write_text(json.dumps(dict(cells=cells)))
        (self.root/'dense-plan.json').write_text(json.dumps(dict(cells=[])))
        paths=[self.root/n for n in ['qwen06-plan.json','dense-plan.json']]
        (self.root/'selection-report.json').write_text(json.dumps(dict(plans={str(p.resolve()):file_sha(p)for p in paths})))
        with patch('advance_r16_dense_middle.selected_rows',return_value=[]):
            self.assertIsNone(campaign.verify_representatives('m',dict(probe1=dict(status='done'),probe16=dict(status='pending')),{}))
    def test_legacy_absent_inactive_fields_do_not_force_recompile(self):
        c=dict(decode='binary',arm='h1',mode='L2',loop=0)
        identity=dict(plan=dict(gemms=[dict(impl='wgmma')],nonpaged_la=False),defines={})
        with patch('advance_r16_dense_middle.native_identity',return_value=identity):old=family(c)[0]
        identity['plan'].update(activation_tma=0,qkv_epilogue=0,moe_activation_bulk=0,cluster_dim=1,combine='stage')
        with patch('advance_r16_dense_middle.native_identity',return_value=identity):self.assertEqual(old,family(c)[0])
        identity['plan']['gemms'][0]['split_k']=4
        with patch('advance_r16_dense_middle.native_identity',return_value=identity):self.assertNotEqual(old,family(c)[0])
        identity['plan']['gemms'][0].pop('split_k')
        identity['plan']['activation_tma']=1
        with patch('advance_r16_dense_middle.native_identity',return_value=identity):self.assertNotEqual(old,family(c)[0])

if __name__=='__main__':unittest.main()
