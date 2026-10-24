"""CPU synthetic data only: no final candidates, timings, or GPU evidence."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import summarize_sm90r2_final as summary


def fixtures():
    baselines, records = [], []
    for model in summary.MODELS:
        for batch in summary.BATCHES:
            for system in summary.SYSTEMS:
                # torch.compile is faster here to catch accidental inclusion
                # in the prompt's three-engine comparator.
                scale = {'torch.compile': .5, 'vllm': 2, 'sglang': 3, 'tensorrt-llm': 4}[system]
                baselines.append(dict(model=model, batch=str(batch), system=system,
                    fresh_processes='3', protocol_complete='True', tpot_ms=scale,
                    ttft_ms=10 * scale, e2e_ms=1000 * scale,
                    tokens_per_second=batch * 1024 / scale, acceptance='provisional',
                    physical_gpu_uuid=''))
            count = 5 if model == 'llama3_1_8b' else 3
            pairs, blanks = [[.002, .001] for _ in range(count)], [[.002, .002]] * 2
            result = dict(schema=1, kind='sm90r2_final_tilemega_pair', model=model, batch=batch,
                processes_per_arm=count, physical_uuid='synthetic-GPU', ownership_verified=False,
                user_final_e2e_amendment=copy.deepcopy(summary.paired.final.AMENDMENT),
                paired_tpot_seconds=pairs, reference_blank_tpot_seconds=blanks,
                arm_metrics=dict(candidate=dict(tpot_seconds=.001, ttft_seconds=.01,
                    e2e_seconds=1, output_tokens_per_second=batch * 1024)),
                **summary.paired.paired_decision(pairs, blanks))
            records.append(dict(source_head='a' * 40, config_key='b' * 64, comparison=result))
    return records, baselines


class FinalSummary(unittest.TestCase):
    def test_ratios_have_consistent_direction_and_three_engine_comparator(self):
        records, rows = fixtures()
        result = summary.summarize(records, summary.baseline_table(rows))
        best = next(r for r in result['ratios'] if r['system'] == 'fastest_engine_tpot')
        self.assertEqual(best['baseline_system'], 'vllm')
        self.assertEqual(best['tpot_ratio'], 2)
        self.assertEqual(best['ttft_ratio'], 2)
        self.assertEqual(best['e2e_ratio'], 2)
        self.assertEqual(best['throughput_ratio'], 2)
        compile_row = next(r for r in result['ratios'] if r['system'] == 'torch.compile')
        self.assertEqual(compile_row['tpot_ratio'], .5)
        self.assertEqual(len(result['cells']), 25)
        self.assertEqual(len(result['ratios']), 125)
        self.assertTrue(result['g_perf_all_cells'])
        self.assertFalse(result['ownership_verified'])
        self.assertEqual(result['external_baselines_run'], 0)
        self.assertTrue(all(c['c1'] == c['c2'] == 'not_performed' for c in result['cells']))

    def test_cohort_medians_are_not_rederived_from_other_medians(self):
        records, rows = fixtures()
        records[0]['comparison']['arm_metrics']['candidate']['ttft_seconds'] = .02
        result = summary.summarize(records, summary.baseline_table(rows))
        cell = next(c for c in result['cells'] if c['model'] == records[0]['comparison']['model'] and c['batch'] == 1)
        self.assertEqual(cell['tpot_ms'], 1)
        self.assertEqual(cell['ttft_ms'], 20)

    def test_fastest_engine_ratios_use_one_engine_for_all_metrics(self):
        records, rows = fixtures()
        for row in rows:
            if row['system'] == 'vllm':
                row['ttft_ms'] = 100
            elif row['system'] == 'sglang':
                row['ttft_ms'] = 1
        result = summary.summarize(records, summary.baseline_table(rows))
        self.assertEqual(next(r for r in result['ratios'] if r['system'] == 'fastest_engine_tpot')['ttft_ratio'], 10)

    def test_complete_unique_baseline_and_final_matrices_are_required(self):
        records, rows = fixtures()
        for value in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError): summary.baseline_table(value)
        for value in (records[:-1], records + [records[0]]):
            with self.assertRaises(ValueError): summary.summarize(value, summary.baseline_table(rows))
        rows[0]['protocol_complete'] = 'False'
        with self.assertRaisesRegex(ValueError, 'incomplete'): summary.baseline_table(rows)

    def test_mixed_heads_unbound_coordinates_and_invalid_metrics_are_rejected(self):
        original, rows = fixtures(); baselines = summary.baseline_table(rows)
        mutations = (lambda r: r.update(source_head='c' * 40),
                     lambda r: r.update(config_key='development'),
                     lambda r: r['comparison']['arm_metrics']['candidate'].update(tpot_seconds=.01),
                     lambda r: r['comparison']['arm_metrics']['candidate'].update(ttft_seconds=float('nan')),
                     lambda r: r['comparison']['arm_metrics']['candidate'].update(e2e_seconds=True),
                     lambda r: r['comparison']['user_final_e2e_amendment'].update(c1='pass'))
        for mutate in mutations:
            records = copy.deepcopy(original); mutate(records[0])
            with self.assertRaises(ValueError): summary.summarize(records, baselines)

    def test_incomplete_8b_cohort_and_forged_decision_are_rejected(self):
        records, rows = fixtures(); baselines = summary.baseline_table(rows)
        records[-1]['comparison']['paired_tpot_seconds'] = [[.002, .001]] * 3
        records[-1]['comparison']['processes_per_arm'] = 3
        with self.assertRaisesRegex(ValueError, 'incomplete'): summary.summarize(records, baselines)
        records, _ = fixtures(); records[0]['comparison']['decision_threshold_pct'] = 0
        with self.assertRaisesRegex(ValueError, 'decision'): summary.summarize(records, baselines)

    def test_undistinguished_or_near_threshold_cell_prevents_global_perf_claim(self):
        for count in (3, 5):
            records, rows = fixtures(); value = records[0]['comparison']
            pairs, blanks = [[.001, .000998]] * count, [[.001, .001]] * 2
            value.update(processes_per_arm=count, paired_tpot_seconds=pairs,
                reference_blank_tpot_seconds=blanks, **summary.paired.paired_decision(pairs, blanks))
            value['arm_metrics']['candidate']['tpot_seconds'] = .000998
            result = summary.summarize(records, summary.baseline_table(rows))
            self.assertFalse(result['g_perf_all_cells'])

    def test_geometric_mean_is_log_mean_and_real_frozen_baseline_is_complete(self):
        records, rows = fixtures(); baselines = summary.baseline_table(rows)
        result = summary.summarize(records, baselines)
        self.assertAlmostEqual(next(r for r in result['ratio_summary'] if r['system'] == 'vllm')['tpot_geomean'], 2)
        path = summary.ROOT / 'docs/results/R16_final_baselines.csv'
        self.assertEqual(summary.sha(path), summary.FROZEN_BASELINE_SHA256)
        with path.open() as stream: real = summary.baseline_table(list(csv.DictReader(stream)))
        self.assertEqual(len(real), 100)

    def test_cli_recollects_all_pairs_before_export_and_rejects_stale_summary(self):
        records, _ = fixtures()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); runs = root / 'runs/sm90r2'; runs.mkdir(parents=True)
            baseline = summary.ROOT / 'docs/results/R16_final_baselines.csv'
            def bound(path): return dict(path=str(path), sha256=summary.sha(path))
            entries, artifacts = [], {}
            for i, record in enumerate(records):
                folder = runs / str(i); (folder / 'queue').mkdir(parents=True)
                files = {k: folder / name for k, name in (
                    ('pair_manifest', 'manifest.json'), ('preparation', 'preparation.json'),
                    ('queue', 'queue/queue_final_pair.json'), ('state', 'state.json'),
                    ('comparison', 'comparison.json'))}
                for name, path in files.items():
                    summary.write(path, record['comparison'] if name == 'comparison' else {})
                entries.append({k: bound(p) for k, p in files.items()})
                artifacts[files['pair_manifest']] = dict(candidate=dict(manifest=dict(
                    source_head=record['source_head'], coordinates=dict(sha256=record['config_key']))),
                    synthetic_comparison=copy.deepcopy(record['comparison']))
            manifest = runs / 'matrix.json'
            summary.write(manifest, dict(schema=1, kind='sm90r2_final_matrix',
                frozen_baseline=bound(baseline), cells=entries))
            def collect(artifact, preparation, state, out):
                summary.write(out, artifact['synthetic_comparison'])
            with patch.object(summary, 'ROOT', root), \
                 patch.object(summary.paired, 'validate_manifest', side_effect=artifacts.__getitem__) as validate, \
                 patch.object(summary.paired, 'collect', side_effect=collect) as recollect, \
                 patch.object(sys, 'argv', ['summary', '--manifest', str(manifest), '--out', str(runs / 'out')]):
                summary.main()
                self.assertEqual(validate.call_count, 25)
                self.assertEqual(recollect.call_count, 25)
                result = json.loads((runs / 'out/summary.json').read_text())
                self.assertEqual(result['gpu_jobs_started'], 0)
                self.assertEqual(len(result['inputs']), 25)
                self.assertTrue((runs / 'out/SM90R2_ratios.csv').is_file())
            altered = copy.deepcopy(records[0]['comparison']); altered['ownership_verified'] = True
            path = Path(entries[0]['comparison']['path']); summary.write(path, altered)
            entries[0]['comparison'] = bound(path)
            summary.write(manifest, dict(schema=1, kind='sm90r2_final_matrix',
                frozen_baseline=bound(baseline), cells=entries))
            with patch.object(summary, 'ROOT', root), \
                 patch.object(summary.paired, 'validate_manifest', side_effect=artifacts.__getitem__), \
                 patch.object(summary.paired, 'collect', side_effect=collect), \
                 patch.object(sys, 'argv', ['summary', '--manifest', str(manifest), '--out', str(runs / 'stale')]):
                with self.assertRaisesRegex(ValueError, 'recollection'): summary.main()
            self.assertFalse((runs / 'stale').exists())


if __name__ == '__main__':
    unittest.main()
