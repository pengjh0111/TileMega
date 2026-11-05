from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from sm90r6_analyze_d0 import predictions

class AttentionMergeAttribution(unittest.TestCase):
    def test_native_and_legacy_merge_names_follow_physical_owner(self):
        for name in ('merge','attention_merge'):
            for mode in ('la','stage'):
                with self.subTest(name=name,mode=mode):
                    data=dict(coordinates=dict(attention=dict(merge=mode),emb='stage'),
                        spaces=[dict(kind='attention',first_start_ns=0,last_end_ns=10),
                                dict(kind=name,first_start_ns=10,last_end_ns=15)],
                        additive_stages=dict(stages=[dict(kind='attention',instances=1,stage_ns=10),
                                                   dict(kind=name,instances=1,stage_ns=5)]))
                    flow,additive,_=predictions(data)
                    expected={'attention':15} if mode=='la' else {'attention':10,'attention_merge':5}
                    self.assertEqual(dict(flow),expected)
                    self.assertEqual(dict(additive),expected)

if __name__=='__main__':unittest.main()
