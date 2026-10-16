"""Comparison tables must not combine partial protocols or distinct checkpoints."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from compare_r16_results import compare, geomeans, candidate_rows


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

    def test_gpu_index_never_proves_physical_pairing(self):
        tm, base = self.row('tilemega'), self.row('tensorrt')
        row, = compare([tm, base], 'dnn')
        self.assertTrue(row['same_gpu_index'])
        self.assertFalse(row['same_gpu'])
        self.assertEqual(row['device_pairing'], 'unknown_physical_gpu')
        tm.update(physical_gpu_uuid='GPU-card-a', physical_gpu_identity_status='verified', hardware_boot_id='boot-a')
        base.update(physical_gpu_uuid='card-b', physical_gpu_identity_status='verified', hardware_boot_id='boot-a')
        row, = compare([tm, base], 'dnn')
        self.assertFalse(row['same_gpu'])
        self.assertEqual(row['device_pairing'], 'different_physical_gpu')
        base.update(gpu='7', physical_gpu_uuid='card-a')
        row, = compare([tm, base], 'dnn')
        self.assertTrue(row['same_gpu'])
        self.assertFalse(row['same_gpu_index'])
        self.assertEqual(row['boot_pairing'], 'same_boot')

    def test_geomeans_keep_physical_evidence_and_boot_scope_separate(self):
        tm, base = self.row('tilemega'), self.row('tensorrt')
        unknown, = compare([tm, base], 'dnn')
        for row in (tm, base):
            row.update(physical_gpu_uuid='card-a', physical_gpu_identity_status='verified', hardware_boot_id='boot-a')
        known, = compare([tm, base], 'dnn')
        base['hardware_boot_id']='boot-b'
        other_boot, = compare([tm, base], 'dnn')
        self.assertEqual(len(geomeans([unknown, known, other_boot], 'dnn')), 3)

    def test_final_candidate_arms_keep_source_and_executor(self):
        item = dict(model='resnet18', batch=1, candidate='h1', source_head='source-a',
            executor='L2', device_loop=0, protocol_complete=True, fresh_processes=3,
            status='completed', default_promoted=False, gpu=6, acceptance='strict',
            physical_gpu_uuid='card-a', physical_gpu_identity_status='verified', hardware_boot_id='boot-a',
            source_checkpoint_sha256='weights', native_artifact_ids={'prefill':'p','decode':'d'},
            e2e_seconds=.1, ttft_seconds=.001, tpot_seconds=.002, output_tokens_per_second=10)
        report = dict(evidence='verified_independent_receipt_analysis', candidates=[
            item, dict(item, source_head='source-b'), dict(item, executor='L1', device_loop=1),
            dict(item, fresh_processes=2, protocol_complete=False)])
        cells = candidate_rows(report)
        self.assertEqual(len(cells), 3)
        self.assertTrue(all(row['physical_gpu_uuid']=='card-a' and row['hardware_boot_id']=='boot-a' for row in cells))
        self.assertEqual(len({row['comparison_arm'] for row in cells}), 3)
        self.assertEqual(len(compare(cells+[self.row('vllm')], 'llm')), 3)
        report['evidence'] = 'pilot only'
        with self.assertRaises(ValueError):
            candidate_rows(report)


if __name__ == '__main__':
    unittest.main()
