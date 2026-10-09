#!/usr/bin/env python3
"""CPU checks for review selection and final queue coverage."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from phase_d_final import outliers,reviewed_choice,validate_round
from make_phase_d_final import queue
from choose_r14 import CELLS

class FinalTests(unittest.TestCase):
    def fixture(self):
        candidates=[];rows=[]
        for mode,loop,values in [('L1',0,[3.54,3.55,3.53]),('L2',0,[3.50,3.51,3.52]),('L2',1,[3.52,3.53,3.51])]:
            candidates.append(dict(library='plan.so',mode=mode,loop=loop,samples_ms=[3.5,7.4,3.6],median_ms=3.6,measurements=['old']))
            rows.append(dict(library='plan.so',mode=mode,loop=loop,samples_ms=values,median_ms=0,
                measurements=[dict(round=r,path=f'{mode}/{loop}/{r}',execution_identity=dict(spill=False)) for r in range(3)]))
        return dict(candidates=candidates,selected=candidates[2]),dict(accepted=True,canary_rounds=[],candidates=rows)
    def test_review_preserves_original_and_uses_measured_minimum(self):
        original,measured=self.fixture();saved=copy.deepcopy(original)
        result=reviewed_choice(original,measured)
        self.assertEqual(original,saved);self.assertEqual((result['selected']['mode'],result['selected']['loop']),('L2',0))
        self.assertEqual(result['selected']['median_ms'],3.51)
        self.assertEqual(result['selected']['original_confirmation_samples_ms'],[3.5,7.4,3.6])
        self.assertEqual(result['selected']['original_confirmation_measurements'],['old'])
    def test_reject_incomplete_or_unstable_recollection(self):
        for change in ('missing','rejected','unstable'):
            original,measured=self.fixture()
            if change=='missing':measured['candidates'].pop()
            elif change=='rejected':measured['accepted']=False
            else:measured['candidates'][0]['samples_ms']=[3.5,7,3.5]
            with self.assertRaises(ValueError):reviewed_choice(original,measured)
    def test_canary(self):
        self.assertEqual(outliers([1,1.01,1]),[]);self.assertEqual(outliers([1,1.04,1]),[1])
        with self.assertRaises(ValueError):outliers([])
    def test_queue_covers_every_cell_and_obeys_dependencies(self):
        steps=queue(49140);byname={row['name']:row for row in steps}
        self.assertEqual(len(steps),41);self.assertEqual(len(byname),41)
        for cell in CELLS:
            for r in range(3):
                row=byname[f'D2_{cell}_r{r}'];self.assertTrue(row['gpu'])
                self.assertEqual(row['needs_free_mib'],42793)
                self.assertEqual(byname[f'D2_validate_{cell}_r{r}']['after'],[row['name']])
            protocol=byname[f'D3_protocol_{cell}']
            self.assertEqual(protocol['retry_args'],['--resume'])
            self.assertIn('50',protocol['command']);self.assertIn(protocol['name']+'_cases.json',' '.join(protocol['command']))
        self.assertEqual(len(byname['D_final_analyze']['after']),16)
        self.assertTrue(all(not row['gpu'] for row in steps if 'validate' in row['name'] or 'canary' in row['name']))
    def test_actual_execution_identity_is_required(self):
        arm=dict(label='R14F',decode='plan.so',mode='auto',decode_loop='auto',expected_mode='L2',expected_loop=True,prefill_identity='prefill')
        row=dict(exit_code=0,execution_identity=dict(artifact_id='decode',executor='L2',loop=True,trace=False),decode_loop_used=True,
                 prefill_execution_identity=dict(artifact_id='prefill',executor='L1'))
        with tempfile.TemporaryDirectory() as tmp,patch('phase_d_final.HERE',Path(tmp)),patch('phase_d_final.arms',return_value={'llama_B1':[arm]}),patch('phase_d_final.verify',return_value={'artifact_id':'decode'}):
            path=Path(tmp)/'raw/D2_llama_B1_r0/llama_B1/round0.json';path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'arms':{'R14F':row}}));validate_round('llama_B1',0)
            for invalid in ('loop','trace','prefill'):
                broken=copy.deepcopy(row)
                if invalid=='loop':broken['decode_loop_used']=False
                elif invalid=='trace':broken['execution_identity']['trace']=True
                else:broken['prefill_execution_identity']['artifact_id']='other'
                path.write_text(json.dumps({'arms':{'R14F':broken}}))
                with self.assertRaises(ValueError):validate_round('llama_B1',0)

if __name__=='__main__':unittest.main()
