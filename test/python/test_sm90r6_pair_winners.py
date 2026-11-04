import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from sm90r6_pair_winners import summarize


class PairBlocks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.tokens = Path(self.temp.name) / 'tokens.json'
        self.tokens.write_text(json.dumps([[1, 2, 3]]))

    def block(self, labels, values, count):
        return [dict(arm=label, tpot_ms=value, tokens=str(self.tokens))
                for _ in range(count) for label, value in zip(labels, values)]

    def test_shared_arm_keeps_pair_specific_measurements(self):
        pairs = [('reuse', 'rederive'), ('reuse', 'materialize'), ('rederive', 'materialize')]
        rows = (self.block(pairs[0], [10, 11], 3) +
                self.block(pairs[1], [20, 18], 3) + self.block(pairs[2], [30, 30], 3))
        result = summarize(rows, pairs, 'qwen3_1_7b')
        self.assertEqual([r['decision'] for r in result], ['slower', 'faster', 'indistinguishable'])
        self.assertAlmostEqual(result[1]['change_pct'], -10)
        self.assertEqual(result[1]['arms']['reuse']['tpot_ms'], 20)

    def test_moe_requires_five_processes_per_side(self):
        with self.assertRaises(ValueError):
            summarize(self.block(('a', 'b'), [1, 1], 3), [('a', 'b')], 'qwen3_30b_a3b')

    def test_changed_alternation_is_rejected(self):
        rows = self.block(('a', 'b'), [1, 1], 3)
        rows[1], rows[2] = rows[2], rows[1]
        with self.assertRaises(ValueError):
            summarize(rows, [('a', 'b')], 'qwen3_1_7b')

    def test_cross_arm_tokens_must_match(self):
        rows = self.block(('a', 'b'), [1, 1], 3)
        other = self.tokens.with_name('other.json'); other.write_text('[[1, 2, 4]]')
        rows[-1]['tokens'] = str(other)
        with self.assertRaises(ValueError):
            summarize(rows, [('a', 'b')], 'qwen3_1_7b')


if __name__ == '__main__':
    unittest.main()
