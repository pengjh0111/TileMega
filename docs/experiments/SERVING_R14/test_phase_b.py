#!/usr/bin/env python3
"""Check matrix coverage and dependency gates before publishing the queue."""
import json,unittest
from pathlib import Path
H=Path(__file__).resolve().parent
class Matrix(unittest.TestCase):
    def test_unique_builds_and_identity_parents(self):
        jobs=json.loads((H/'phase_b_builds.json').read_text());paths=[j['out'] for j in jobs]
        self.assertEqual(len(paths),len(set(paths)))
        for j in jobs:
            if j.get('base_so'):self.assertIn(str(Path(j['base_so']).parent),paths)
            else:
                self.assertTrue(Path(j['manifest']).exists());self.assertTrue(Path(j['classes']).exists())
    def test_matrices_and_numeric_gate(self):
        queue=json.loads((H/'queue_phase_b.pending.json').read_text());steps={s['name']:s for s in queue}
        self.assertEqual(steps['B0b']['after'],['Bpre_numeric'])
        for cell in ('llama_B1','llama_B16','qwen3_B1','qwen3_B16'):
            for i in (1,2,3,5):
                arms=json.loads((H/f'B{i}_arms.declared.json').read_text())[cell]
                self.assertEqual(arms[0]['label'],'baseline');self.assertGreater(len(arms),1)
                for r in range(3):self.assertEqual(steps[f'B{i}_{cell}_r{r}']['after'],['B0c_'+cell])
        for cell in ('llama_B1','qwen3_B1','llama_B16'):self.assertIn('B6_protocol_'+cell,steps)
    def test_gemv_not_mma_geometry(self):
        for j in json.loads((H/'phase_b_builds.json').read_text()):
            if j['label'].startswith('GV_') and not j.get('base_so'):
                self.assertEqual(j['batch'],1)
                self.assertTrue(all(g['values']['impl']=='gemv' and g['values']['stages']==2 for g in j['gemm_overrides']))
if __name__=='__main__':unittest.main()
