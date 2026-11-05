import copy
from pathlib import Path
import sys
import unittest
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from sm90r6_diagnose_a0 import summarize, triggered, diagnose
from sm90d1_common import write,sha


class Diagnosis(unittest.TestCase):
    def test_trigger_uses_pabl_range_threshold(self):
        m = dict(arms={'A0': dict(tpot_ms=10, range_pct=.1), 'F': dict(tpot_ms=9, range_pct=.2)},
                 comparisons={'F': dict(change_pct=-10, threshold_pct=.5)})
        self.assertTrue(triggered(m))
        m['arms']['F']['range_pct'] = 6; m['comparisons']['F']['threshold_pct'] = 12
        self.assertFalse(triggered(m))
        m['comparisons']['F']['change_pct'] = 0
        with self.assertRaises(ValueError): triggered(m)

    def test_partial_diagnostic_keeps_protocol_and_rejects_wrong_exclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);base=root/'partial';cell='olmoe_1b_7b-B1'
            directory=base/cell;(directory/'measurement').mkdir(parents=True)
            write(directory/'arms.json',{})
            measured=dict(protocol='P-ABL partial seven-arm',excluded_arm='AB1a',
                source_sha256=sha(directory/'arms.json'),
                arms={arm:dict(tpot_ms=1,range_pct=.1) for arm in ('A0','F')},
                comparisons={'F':dict(change_pct=0,threshold_pct=.5)})
            path=directory/'measurement/result.json';write(path,measured)
            result=diagnose(root,cell,root/'unused',root/'good',base)
            self.assertFalse(result['triggered'])
            self.assertEqual(result['measurement_protocol'],measured['protocol'])
            measured['excluded_arm']='AB1b';write(path,measured)
            with self.assertRaisesRegex(ValueError,'P-ABL manifest'):
                diagnose(root,cell,root/'unused',root/'bad',base)

    def fixture(self):
        candidates = [dict(coordinates=i, pool_ns=p, structural_class=s)
                      for i, p, s in [(0, 20, 'A'), (1, 10, 'F'), (2, 30, 'F')]]
        predictions = [dict(coordinates=c['coordinates'], pool_ns=c['pool_ns'], replay_ns=r,
            decomposition=dict(original_ns=c['pool_ns'], synchronization_ns=0, fixed_ns=0,
                               contention_ns=0, chain_ns=c['pool_ns'])) for c, r in zip(candidates, [20, 30, 10])]
        return dict(evaluated=candidates, finalists=[candidates[2]]), predictions

    def test_rank_covers_all_E_and_class_can_enter_without_F(self):
        receipt, predictions = self.fixture(); result = summarize(receipt, predictions, 0, 1)
        self.assertEqual((result['F_pool_rank'], result['F_replay_rank']), (1, 3))
        self.assertTrue(result['F_structure_in_finalists'])
        self.assertEqual(result['delta_A0_minus_F']['chain_ns'], 10)

    def test_partial_replay_or_changed_prediction_is_rejected(self):
        receipt, predictions = self.fixture()
        for bad in (predictions[:2], copy.deepcopy(predictions)):
            if len(bad) == 3: bad[0]['pool_ns'] += 1
            with self.assertRaises(ValueError): summarize(receipt, bad, 0, 1)


if __name__ == '__main__': unittest.main()
