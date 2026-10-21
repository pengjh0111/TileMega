"""Reject failed or incomplete correctness before native solver selection."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from measure_sm90r2_candidate import require_correctness


class NativeSelectionCorrectness(unittest.TestCase):
    def test_accepts_complete_native_evidence(self):
        require_correctness(dict(pass_=False,**{'pass':True,'batch':16,'max_new_tokens':256}),
            {'pass':True,'same_binary':True,'checks':[
                {'pass':True,'decode_loop':0},{'pass':True,'decode_loop':1}]},16)

    def test_rejects_wrong_batch_or_short_teacher_forcing(self):
        good={'pass':True,'batch':1,'max_new_tokens':256}
        c2={'pass':True,'same_binary':True,'checks':[
            {'pass':True,'decode_loop':0},{'pass':True,'decode_loop':1}]}
        for key,value in (('pass',False),('batch',16),('max_new_tokens',128)):
            bad=dict(good,**{key:value})
            with self.assertRaises(ValueError):require_correctness(bad,c2,1)

    def test_rejects_cross_binary_or_missing_execution_arm(self):
        c1={'pass':True,'batch':1,'max_new_tokens':256}
        c2={'pass':True,'same_binary':True,'checks':[
            {'pass':True,'decode_loop':0},{'pass':True,'decode_loop':1}]}
        variants=[]
        bad=copy.deepcopy(c2);bad['same_binary']=False;variants.append(bad)
        bad=copy.deepcopy(c2);bad['checks'][1]['pass']=False;variants.append(bad)
        bad=copy.deepcopy(c2);bad['checks'][1]['decode_loop']=0;variants.append(bad)
        for value in variants:
            with self.assertRaises(ValueError):require_correctness(c1,value,1)


if __name__=='__main__':unittest.main()
