"""Do not merge distinct native implementations or repeat one measured process."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from summarize_r16 import aggregate_dnn


class DnnAggregation(unittest.TestCase):
    def rows(self, candidate, artifact):
        return [dict(model='mbv1', system='tilemega', batch=1, precision='bf16',
            compile_mode='', candidate_id=candidate, candidate_receipt_sha256=candidate,
            native_artifact_ids=artifact, round=r, pid=100+r, gpu=6,
            median_ms=1+r/10, p90_ms=2+r/10, exported_program_sha256='same-checkpoint',
            acceptance='provisional_container_pid_namespace') for r in range(3)]

    def test_distinct_bodies_and_binaries_stay_separate(self):
        rows=aggregate_dnn(self.rows('mma', 'binary-a')+self.rows('wgmma', 'binary-b'))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row['protocol_complete'] for row in rows))
        self.assertEqual({row['native_artifact_ids'] for row in rows}, {'binary-a','binary-b'})

    def test_repeated_process_or_checkpoint_is_rejected(self):
        rows=self.rows('mma', 'binary-a')
        for change in ('pid', 'exported_program_sha256', 'round'):
            broken=copy.deepcopy(rows)
            broken[1][change]=broken[0][change] if change != 'exported_program_sha256' else 'other-checkpoint'
            with self.subTest(change=change), self.assertRaises(ValueError):
                aggregate_dnn(broken)


if __name__ == '__main__':
    unittest.main()
