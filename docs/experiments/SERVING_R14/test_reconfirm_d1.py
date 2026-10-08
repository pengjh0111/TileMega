import unittest
from reconfirm_d1 import canary,finalists

class Reconfirmation(unittest.TestCase):
    def test_canary_uses_registered_two_percent_threshold(self):
        self.assertEqual(canary([3.51,7.48,3.50]),[1])
        self.assertEqual(canary([3.50,3.51,3.52]),[])
    def test_only_three_confirmed_finalists_are_recollected(self):
        rows=[dict(mode=mode,loop=loop,samples_ms=[1,1,1]) for mode,loop in [('L1',0),('L2',0),('L2',1)]]
        self.assertEqual(len(finalists(dict(candidates=rows+[dict(samples_ms=[],error='budget')]))),3)
        with self.assertRaises(ValueError):finalists(dict(candidates=rows[:2]))
    def test_missing_reference_is_rejected(self):
        rows=[dict(mode='L2',loop=0,samples_ms=[1,1,1]) for _ in range(3)]
        with self.assertRaises(ValueError):finalists(dict(candidates=rows))

if __name__=='__main__':unittest.main()
