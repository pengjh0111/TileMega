"""Comparison tables must not combine partial protocols or distinct checkpoints."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from compare_r16_results import compare, geomeans


class Comparisons(unittest.TestCase):
    def row(self, system, export='same'):
        return dict(model='resnet18', batch=1, system=system, precision='bf16',
            compile_mode='', candidate_id='', native_artifact_ids='binary',
            exported_program_sha256=export, source_checkpoint_sha256='',
            protocol_complete=True, fresh_processes=3, gpu='6', acceptance='strict',
            median_ms=2, p90_ms=3, e2e_ms=100, ttft_ms=1, tpot_ms=2, tokens_per_second=10)

    def test_only_complete_identical_dnn_exports_compare(self):
        tm, base = self.row('tilemega'), self.row('tensorrt')
        base['median_ms'] = 1
        ratios = compare([tm, base, self.row('tensorrt', 'nonzero-variant')], 'dnn')
        self.assertEqual(len(ratios), 1)
        self.assertEqual(ratios[0]['median_ms_speedup'], .5)
        self.assertEqual(geomeans(ratios, 'dnn')[0]['matched_cells'], 1)
        base['fresh_processes'] = 2
        self.assertFalse(compare([tm, base], 'dnn'))

    def test_unpaired_llm_is_explicit_and_bad_times_are_rejected(self):
        tm, base = self.row('TM-seed'), self.row('vllm')
        ratio, = compare([tm, base], 'llm')
        self.assertFalse(ratio['checkpoint_shared'])
        self.assertIn('MoE routing unpaired', ratio['checkpoint_scope'])
        base['tpot_ms'] = float('nan')
        with self.assertRaises(ValueError):
            compare([tm, base], 'llm')

    def test_ambiguous_artifacts_are_not_averaged_together(self):
        tm, base = self.row('tilemega'), self.row('tensorrt')
        rows = compare([tm, tm, base], 'dnn')
        with self.assertRaises(ValueError):
            geomeans(rows, 'dnn')


if __name__ == '__main__':
    unittest.main()
