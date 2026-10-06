import unittest
from tilemega.serving.integrated_selection import integrated_ms,successive_halving
class Selection(unittest.TestCase):
    def test_constant(self):
        self.assertEqual(integrated_ms({str(p):{'mean_ms':2.5} for p in (64,575,1000)}),2.5)
    def test_trajectory(self):
        rows={str(p):{'mean_ms':1+p/1000} for p in (64,575,1000)}
        expected=sum(1+min(p,1000)/1000 for p in range(64,1088))/1024
        self.assertAlmostEqual(integrated_ms(rows),expected)
    def test_halving_keeps_evidence_and_fresh_final_rounds(self):
        calls=[]
        def measure(c,r):
            calls.append((c['id'],r));return {'by_past':{str(p):{'mean_ms':c['id']+1} for p in (64,575,1000)},'execution_identity':{'execution_id':str(c['id'])}}
        winner,rows=successive_halving([{'id':i} for i in range(8)],measure)
        self.assertEqual(winner['id'],0);self.assertEqual(sum(x['eliminated_round'] is None for x in rows),3)
        self.assertEqual(len(winner['samples_ms']),3);self.assertEqual(len(calls),8+4+9)
        self.assertEqual([i for i,r in calls if r=='final1'],[1,2,0])
    def test_failed_candidate_and_guard_exit(self):
        def measure(c,r):
            if c['id']==0:raise RuntimeError('unavailable loop')
            return {'by_past':{str(p):{'mean_ms':1} for p in (64,575,1000)},'execution_identity':{'execution_id':str(c['id'])}}
        winner,rows=successive_halving([{'id':i} for i in range(4)],measure)
        self.assertEqual(winner['id'],1);self.assertIn('error',rows[0])
        def occupied(c,r):raise SystemExit(75)
        with self.assertRaises(SystemExit) as result:successive_halving([{'id':0}],occupied)
        self.assertEqual(result.exception.code,75)
    def test_identity_change_rejected(self):
        def measure(c,r):
            return {'by_past':{str(p):{'mean_ms':1} for p in (64,575,1000)},'execution_identity':{'execution_id':r}}
        with self.assertRaises(ValueError):successive_halving([{'id':0}],measure)
    def test_invalid(self):
        with self.assertRaises(ValueError):integrated_ms({str(p):{'mean_ms':float('nan')} for p in (64,575,1000)})
if __name__=='__main__':unittest.main()
