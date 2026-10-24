"""CPU-only synthetic comparison fixtures; no final artifacts or GPU evidence."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import run_sm90r2_final_pair as pair


class FinalTileMegaPairs(unittest.TestCase):
    def test_frozen_reference_requires_index_identity_hash_head_and_coordinates(self):
        plan = dict(phase='decode', seq=1, batch_lo=1, batch_hi=1, capacity=1088,
                    pg='off', requested_residency=0, attention_impl='simt', attention_kv_block=128,
                    gemms=[dict(tile_m=16, tile_n=32, tile_k=128, stages=4, split_k=16, impl='mma16')])
        identity = dict(trace=False, source=dict(head=pair.REFERENCE_HEAD), artifact_id='indexed', plan=plan)
        binding = dict(head=pair.REFERENCE_HEAD, artifact_id='indexed', binary_sha256='a')
        config = dict(coordinates=dict(decode=pair.final.prefill_signature(plan)))
        pair.validate_reference_identity(identity, binding, config, 'decode', 1, 'a')
        for mutate in (lambda x: x.update(trace=True), lambda x: x.update(artifact_id='other'),
                       lambda x: x['source'].update(head='a' * 40),
                       lambda x: x['plan']['gemms'][0].update(tile_n=64)):
            value = copy.deepcopy(identity); mutate(value)
            with self.assertRaisesRegex(ValueError, 'frozen indexed'):
                pair.validate_reference_identity(value, binding, config, 'decode', 1, 'a')
        with self.assertRaisesRegex(ValueError, 'frozen indexed'):
            pair.validate_reference_identity(identity, binding, config, 'decode', 1, 'changed')

    def test_pair_specs_bookend_the_complete_final_cohort(self):
        for count in (3, 5):
            specs = pair.pair_specs(count)
            self.assertEqual(specs[0]['kind'], 'blank_before')
            self.assertEqual(specs[-1]['kind'], 'blank_after')
            self.assertEqual([s['fresh_process'] for s in specs[1:-1]], list(range(count)))
            self.assertTrue(all(s['kind'] == 'comparison' for s in specs[1:-1]))

    def test_threshold_includes_blanks_both_ranges_and_session_drift(self):
        result = pair.paired_decision([[100, 90]] * 3, [[98, 102], [100, 100]])
        self.assertAlmostEqual(result['decision_threshold_pct'], 4)
        self.assertAlmostEqual(result['paired_change_pct'], -10)
        self.assertEqual(result['decision'], 'faster')
        result = pair.paired_decision([[100, 90], [100, 91], [100, 92]], [[100, 100]] * 2)
        self.assertAlmostEqual(result['decision_threshold_pct'], 2 / 91 * 100)
        result = pair.paired_decision([[100, 90]] * 3, [[104, 104]] * 2)
        self.assertEqual(result['blank_process_range_pct'], 0)
        self.assertEqual(result['reference_process_range_pct'], 0)
        self.assertAlmostEqual(result['decision_threshold_pct'], 4 / 104 * 100)

    def test_delta_uses_pair_ratios_and_threshold_floor_is_unchanged(self):
        result = pair.paired_decision([[100, 98], [200, 190], [300, 290]], [[100, 100]] * 2)
        self.assertAlmostEqual(result['paired_change_pct'], (290 / 300 - 1) * 100)
        result = pair.paired_decision([[100, 90]] * 3, [[100, 100]] * 2)
        self.assertEqual(result['decision_threshold_pct'], .5)
        self.assertNotIn('g_perf_pass', result)

    def test_near_threshold_three_process_result_requests_five(self):
        result = pair.paired_decision([[100, 99.4]] * 3, [[100, 100]] * 2)
        self.assertTrue(result['needs_five_processes'])
        self.assertEqual(result['decision'], 'needs_five_processes')
        result = pair.paired_decision([[100, 99.4]] * 5, [[100, 100]] * 2)
        self.assertFalse(result['needs_five_processes'])
        self.assertEqual(result['decision'], 'faster')
        result = pair.paired_decision([[100, 99.8]] * 5, [[100, 100]] * 2)
        self.assertEqual(result['decision'], 'indistinguishable')

    def test_incomplete_pairs_or_invalid_timings_are_rejected(self):
        for pairs, blanks in [([[100, 90]] * 2, [[100, 100]] * 2),
                              ([[100, 90]] * 3, [[100, 100]]), ([[100]] * 3, [[100, 100]] * 2),
                              ([[100, 0]] * 3, [[100, 100]] * 2),
                              ([[100, float('nan')]] * 3, [[100, 100]] * 2)]:
            with self.assertRaises(ValueError):
                pair.paired_decision(pairs, blanks)

    def test_queue_is_inactive_and_uses_one_physical_wrapper_per_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); policy = root / 'policy.json'
            pair.write(policy, dict(idle_power_mapping_provenance=dict(uuid_by_gpu={'1': 'GPU-test'})))
            bound = dict(candidate=dict(files=dict(guard_policy=policy), runner_sha256='candidate'),
                         processes=3, path=root / 'manifest.json', manifest_sha256='manifest', runner_sha256='paired')
            out = root / 'preparation'; pair.prepare_queue(bound, 1, out)
            jobs = pair.read(out / 'queue/queue_final_pair.json')
            self.assertEqual(len(jobs), 5)
            for job in jobs:
                self.assertEqual(sum(str(c).endswith('run_r16_with_device_identity.py') for c in job['command']), 1)
                self.assertTrue(any(str(c).endswith('run_sm90r2_final_pair.py') for c in job['command']))
                self.assertFalse(any(str(c).endswith('run_sm90r2_final_e2e.py') for c in job['command']))
            prepared = pair.checked_preparation(bound, out)
            self.assertEqual(prepared['gpu_jobs_started'], 0)
            self.assertFalse(prepared['active_queue_publication'])
            pair.write(out / 'queue/queue_final_pair.json', [])
            with self.assertRaisesRegex(ValueError, 'preparation changed'):
                pair.checked_preparation(bound, out)

    def test_outer_guard_rejection_invalidates_both_finished_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); policy = root / 'policy.json'
            pair.write(policy, dict(idle_power_mapping_provenance=dict(uuid_by_gpu={'1': 'GPU-test'})))
            bound = dict(candidate=dict(files=dict(guard_policy=policy), runner_sha256='candidate'),
                         processes=3, path=root / 'manifest.json', manifest_sha256='manifest', runner_sha256='paired')
            out = root / 'preparation'; pair.prepare_queue(bound, 1, out)
            prepared = pair.read(out / 'preparation.json'); attempt = root / 'attempt'
            (attempt / 'pair').mkdir(parents=True)
            pair.write(attempt / 'guard_result.json', dict(code=75))
            for name in ('paired-process', 'physical-device-identity'):
                pair.write(attempt / ('pair/' + name + '.json'), {})
            pair.write(attempt / 'pair/physical-device-command.json', dict(actual_returncode=0))
            pair.write(root / 'state.json', {name: dict(status='done', exit_code=0, out=str(attempt))
                                            for name in prepared['jobs']})
            with self.assertRaisesRegex(ValueError, 'guard/wrapper/session'):
                pair.collect(bound, out, root / 'state.json', root / 'result.json')
            self.assertFalse((root / 'result.json').exists())

    def test_strict_candidate_cannot_reduce_process_count(self):
        with self.assertRaisesRegex(ValueError, 'fresh process'):
            pair.final.run(dict(manifest=dict(model='llama3_1_8b')), Path('/unused'), 0, processes=3)
        with self.assertRaisesRegex(ValueError, 'fresh process'):
            pair.final.run(dict(manifest=dict(model='qwen3_0_6b')), Path('/unused'), 0, processes=1)


if __name__ == '__main__':
    unittest.main()
