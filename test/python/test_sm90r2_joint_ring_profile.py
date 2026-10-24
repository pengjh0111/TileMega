import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from sm90r2_joint_ring_profile import (CACHE_STATE, DIMENSIONS, ENDPOINT_MODEL,
                                      JointRingProfiles as NativeProfiles, MissingJointRingProfile, quote_stage)


def JointRingProfiles(value):
    return NativeProfiles(value, expected_producer_header_sha256='a'*64)


def fixture(iterations=4):
    """Synthetic CPU values only; never a calibration observation table."""
    row = dict(profile_id='synthetic_cpu_fixture', batch=1, tile_n=32, tile_k=256,
               slots=3, warps_n=1, warps_k=4, register_stages=1, grid=132,
               active_ctas=128, compiled_residency=1, mma='mma_sync', input_k=256*iterations, tasks_per_cta=1,
               prefix_bytes=0, prefetch_l2_bytes=0, dynamic_shared_bytes=136320,
               workspace_offset=0, continuous_task_drain=False, cache_state=CACHE_STATE,
               evidence='synthetic_cpu_fixture', aggregation='per_cta_sample_median',
               processes=5, issue_to_finalize_ns=10000,
               first_full_ns=1000, main_span_ns=8000, finalize_tail_ns=3000,
               finite_prefix=[dict(sequence=q, full_ns=1000+200*q, release_ns=1100+200*q)
                              for q in range(min(4, iterations))])
    return dict(schema=1, endpoint_model=ENDPOINT_MODEL, profiles=[row],
                source_binding={key: 'a'*64 for key in ('source_digest', 'producer_header_sha256',
                                'probe_sha256', 'binary_sha256', 'collector_sha256')})


def dimensions(row):
    return {key: row[key] for key in DIMENSIONS}


class JointRingProfileTest(unittest.TestCase):
    def test_single_paired_total_prices_real_consumer_and_finalize_once(self):
        table = fixture()
        profile = JointRingProfiles(table).find(dimensions(table['profiles'][0]))
        quote = quote_stage(profile, barrier_ns=1000, initialization_ns=500, epilogue_ns=450,
                            issue_ns=20, receive_ns=30)
        self.assertEqual(quote['stage_ns'], 12000)
        self.assertEqual(quote['startup_ns'], 500)
        self.assertEqual(quote['flow_ns'], 10000)
        for key in ('read_startup_ns', 'compute_ns', 'activation_ns'):
            self.assertEqual(quote[key], 0)
        self.assertTrue(quote['joint_ring_profile']['diagnostic_only'])
        self.assertEqual(quote['joint_ring_profile']['fit_status'], 'joint_ring_diagnostic_pending_refit')

    def test_marginal_medians_need_not_sum_to_total(self):
        table = fixture()
        row = table['profiles'][0]
        self.assertNotEqual(sum(row[key] for key in ('first_full_ns', 'main_span_ns', 'finalize_tail_ns')),
                            row['issue_to_finalize_ns'])
        profile = JointRingProfiles(table).find(dimensions(row))
        self.assertEqual(quote_stage(profile, barrier_ns=0, initialization_ns=0, epilogue_ns=0)['stage_ns'], 10000)

    def test_i_one_and_i_less_than_slots_are_finite_observations(self):
        for iterations in (1, 2):
            with self.subTest(iterations=iterations):
                table = fixture(iterations)
                self.assertEqual(len(table['profiles'][0]['finite_prefix']), iterations)
                profile = JointRingProfiles(table).find(dimensions(table['profiles'][0]))
                self.assertEqual(profile['issue_to_finalize_ns'], 10000)

    def test_no_nearest_shape_or_active_grid_task_prefix_fallback(self):
        table = fixture()
        reader = JointRingProfiles(table)
        for key in DIMENSIONS:
            with self.subTest(dimension=key):
                request = dimensions(table['profiles'][0]); request[key] += 1
                with self.assertRaises(MissingJointRingProfile) as error:
                    reader.find(request)
                self.assertEqual(error.exception.measurement['metric'], 'native_joint_ring_profile')
        request = dimensions(table['profiles'][0]); request['active_ctas'] = 64
        with self.assertRaises(MissingJointRingProfile):
            reader.find(request)
        with self.assertRaises(MissingJointRingProfile):
            reader.find(dimensions(table['profiles'][0]), 'cold_dram')
        with self.assertRaises(MissingJointRingProfile):
            reader.find(dimensions(table['profiles'][0]), mma='wgmma')

    def test_missing_nonmonotone_and_duplicate_prefix_reject(self):
        for mode in ('missing', 'nonmonotone', 'duplicate'):
            with self.subTest(mode=mode):
                table = fixture(); prefix = table['profiles'][0]['finite_prefix']
                if mode == 'missing': prefix.pop()
                if mode == 'nonmonotone': prefix[1]['full_ns'] = 100
                if mode == 'duplicate': prefix[1]['sequence'] = 0
                with self.assertRaises(ValueError): JointRingProfiles(table)

    def test_unbound_source_wrong_endpoint_duplicate_profiles_reject(self):
        for mode in ('binding', 'endpoint', 'duplicate_id', 'duplicate_coordinate'):
            with self.subTest(mode=mode):
                table = fixture()
                if mode == 'binding': table['source_binding']['probe_sha256'] = '/tmp/raw'
                if mode == 'endpoint': table['endpoint_model'] = 'pure_dma'
                if mode.startswith('duplicate'):
                    row = copy.deepcopy(table['profiles'][0])
                    if mode == 'duplicate_coordinate': row['profile_id'] += '_other'
                    table['profiles'].append(row)
                with self.assertRaises(ValueError): JointRingProfiles(table)
        with self.assertRaisesRegex(ValueError, 'producer header differs'):
            NativeProfiles(fixture(), expected_producer_header_sha256='b'*64)

    def test_unmeasured_continuous_task_and_prefix_scope_reject(self):
        for key, value in (('tasks_per_cta', 2), ('continuous_task_drain', True),
                           ('prefix_bytes', 16), ('prefetch_l2_bytes', 16), ('processes', 3),
                           ('evidence', 'inferred')):
            with self.subTest(key=key):
                table = fixture(); table['profiles'][0][key] = value
                with self.assertRaises(ValueError): JointRingProfiles(table)

    def test_invalid_numbers_and_impossible_spans_reject(self):
        for key, value in (('issue_to_finalize_ns', 0), ('main_span_ns', float('nan')),
                           ('first_full_ns', 11000), ('finalize_tail_ns', -1),
                           ('grid', True), ('tile_n', 1.5)):
            with self.subTest(key=key):
                table = fixture(); table['profiles'][0][key] = value
                with self.assertRaises(ValueError): JointRingProfiles(table)
        with self.assertRaises(ValueError):
            quote_stage(fixture()['profiles'][0], barrier_ns=-1, initialization_ns=0, epilogue_ns=0)


if __name__ == '__main__':
    unittest.main()
