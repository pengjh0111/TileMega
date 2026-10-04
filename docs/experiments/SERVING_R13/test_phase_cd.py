#!/usr/bin/env python3
"""Exercise partial evidence and token mismatches before publishing C/D."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import phase_cd_r13 as c
from choose_r13 import phase_c,final
class Checks(unittest.TestCase):
    def fixture(self,root,label,round_,tokens,exit_code=0):
        out=root/'raw'/f'C_C-WL_llama_B1_r{round_}'/'llama_B1'/label/f'round{round_}'
        out.mkdir(parents=True,exist_ok=True)
        (out/'tokens_N1024_run1.json').write_text(json.dumps(tokens))
        path=out.parent.parent/f'round{round_}.json'
        data=json.loads(path.read_text()) if path.exists() else {'arms':{}}
        data['arms'][label]=dict(exit_code=exit_code,out=str(out),e2e_seconds=3.1,ttft_seconds=.031)
        path.write_text(json.dumps(data))
    def test_missing_round_cannot_prove_equality(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(c,'HERE',Path(tmp)):
            for r in (0,1):
                for label in ('N-control','N-tiled'):self.fixture(Path(tmp),label,r,[[1,2]])
            self.assertEqual(c.numerical('C-WL','llama_B1'),{})
    def test_changed_token_rejects_layout(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(c,'HERE',Path(tmp)):
            for r in range(3):
                self.fixture(Path(tmp),'N-control',r,[[1,2]])
                self.fixture(Path(tmp),'N-tiled',r,[[1,3]] if r==2 else [[1,2]])
            report=c.numerical('C-WL','llama_B1')
            self.assertTrue(report['N-control']);self.assertFalse(report['N-tiled'])
    def test_failed_samples_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(c,'HERE',Path(tmp)):
            self.fixture(Path(tmp),'N-tiled',0,[[1,2]],75)
            self.assertEqual(c.samples('C-WL','llama_B1'),({},{}))
    def test_absent_trigger_evidence_selects_nothing(self):
        self.assertEqual(phase_c({})['selected'],[])
    def test_default_final_requires_resolvable_difference(self):
        self.assertEqual(final({'llama_B1':{'B0-D':[3,3.1,3.2],'R13F':[2.99,3.05,3.1]}})['llama_B1'],'B0-D')
if __name__=='__main__':unittest.main()
