#!/usr/bin/env python3
"""Selection must not turn missing correctness or missing measurements into wins."""
import copy,unittest
from choose_r14 import baseline,phase_c,retention,final,CELLS
class Selection(unittest.TestCase):
    def test_baseline_requires_correctness(self):
        rows={c:{"R13D'":dict(c1=True,r13_tokens_identical=True,tpot=[3,3,3])} for c in CELLS}
        rows['llama_B1']["N1'"]=dict(c1=False,tpot=[2,2,2])
        self.assertEqual(baseline(rows)['llama_B1'],"R13D'")
        rows['llama_B1']["N1'"]['c1']=True
        self.assertEqual(baseline(rows)['llama_B1'],"N1'")
        rows['qwen3_B16']["R13D'"]['r13_tokens_identical']=False
        with self.assertRaises(ValueError):baseline(rows)
    def test_missing_is_pending_and_boundaries_are_literal(self):
        self.assertTrue(all(x['status']=='pending' for x in phase_c({}).values()))
        rows={'C-AT3b':{'llama_B1':dict(applicable=True,attention_page_wait_fraction=.30)},
              'C-RW1':{'llama_B16':dict(applicable=True,gemm_over_load_fraction=.20)}}
        self.assertEqual(phase_c(rows)['C-AT3b']['status'],'triggered')
        self.assertEqual(phase_c(rows)['C-RW1']['status'],'not_triggered')
    def test_synchronization_and_loader_gates(self):
        row=dict(correct=True,baseline=[4,4,4],candidate=[3,3,3],fresh_passed=49,loader_floor_fraction=.98)
        self.assertFalse(retention('C-RW2',{'llama_B16':row})['retain'])
        row['fresh_passed']=50
        self.assertTrue(retention('C-RW2',{'llama_B16':row})['retain'])
        row['loader_floor_fraction']=.96
        self.assertFalse(retention('C-RW2',{'llama_B16':row})['retain'])
    def test_final_fallback_and_gate_are_separate(self):
        b=dict(c1=True,c2=True,tpot=[3,3,3],paired_vllm_ratios=[1,1,1])
        c=dict(c1=True,c2=True,tpot=[2.99,3,3.01],paired_vllm_ratios=[1.2,1.2,1.2])
        rows={cell:dict(baseline=copy.deepcopy(b),R14F=copy.deepcopy(c)) for cell in CELLS}
        result=final(rows)
        self.assertTrue(all(r['selected']=='baseline' and not r['gate_pass'] for r in result.values()))
if __name__=='__main__':unittest.main()
