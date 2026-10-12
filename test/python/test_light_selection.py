"""R15 bounds pilot admission while covering each declared marginal value."""
import unittest
from tilemega.serving.attention_selection import variants
from tilemega.serving.execution import execution_combinations
from tilemega.serving.integrated_selection import light_candidates,successive_halving

class LightSelection(unittest.TestCase):
    def test_eight_candidates_cover_ec_body_and_execution(self):
        candidates=[]
        for pg in ('l2','pages'):
            for variant in variants(dict(capacity=1088,pg=pg,attention_q_per_kv=4,attention_head_dim=64)):
                for mode,loop in execution_combinations(pg,'measure','measure'):
                    if mode=='L1' and variant['nonpaged_la']:continue
                    candidates.append(dict(pg=pg,mode=mode,loop=loop,variant=variant))
        chosen=light_candidates(candidates)
        self.assertEqual(len(chosen),8)
        for field in ('ec','attention_impl','nonpaged_la'):
            self.assertEqual({r['variant'][field] for r in chosen},{r['variant'][field] for r in candidates})
        self.assertEqual({(r['mode'],r['loop']) for r in chosen},{(r['mode'],r['loop']) for r in candidates})
        self.assertFalse(any(r['mode']=='L1' and r['variant']['nonpaged_la'] for r in chosen))
        with self.assertRaises(ValueError):light_candidates(candidates,2)

    def test_one_pilot_then_three_alternating_final_rounds(self):
        calls=[]
        def measure(candidate,label):
            calls.append((candidate['id'],label))
            return dict(execution_identity={'id':candidate['id']},
                        by_past={str(p):dict(mean_ms=1+candidate['id']) for p in (64,575,1000)})
        winner,rows=successive_halving([dict(id=i) for i in range(8)],measure,pilot_rounds=1)
        self.assertEqual(winner['id'],0)
        self.assertEqual(sum(label.startswith('pilot') for _,label in calls),8)
        self.assertEqual([i for i,label in calls if label=='final0'],[0,1,2])
        self.assertEqual([i for i,label in calls if label=='final1'],[1,2,0])
        self.assertEqual([i for i,label in calls if label=='final2'],[2,0,1])
        self.assertEqual(sum(r['eliminated_round']=='pilot0' for r in rows),5)

if __name__=='__main__':unittest.main()
