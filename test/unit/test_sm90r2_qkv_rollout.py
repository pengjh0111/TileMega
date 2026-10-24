import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import rollout_sm90r2_qkv_stage as adapter


def source_fixture():
    """Fictional source IDs, with the supported exact body contract."""
    gemm = dict(tile_n=32, slot_bytes=32768, slots=2, warps_n=1,
                warps_k=4, register_stages=1)
    descriptor = dict(
        domain=dict(batch=16, shapes=[[4096, 1024]],
                    coordinates=dict(gemms=dict(qkv=gemm), swiglu_u=8)),
        plan_binding=dict(epilogue='store', input_norm=True, output_sums=False),
        source_gemm=dict(family='qkv', stage=1, gemm=0, norm_ss=5, ss_out=None))
    ctas, events = [], []
    kinds = dict(full=13, empty=18, consumer_return=28)
    for cta in range(132):
        tasks = []
        if cta < 128:
            blocks = [dict(sequence=q, slot=q, generation=0, previous_reader=-1,
                           weight_bytes=32768, activation_bytes=16384,
                           prefix_bytes=0, remainder_requests=1,
                           activation_arrivals=32, empty_arrivals=128)
                      for q in range(2)]
            tasks.append(dict(task=cta, blocks=blocks))
            for q in range(2):
                for kind in ('full', 'empty'):
                    events.append(dict(stage=0, cta=cta, kind=kinds[kind], sequence=q))
            events.append(dict(stage=0, cta=cta, kind=kinds['consumer_return'], sequence=-1))
        ctas.append(dict(cta=cta, tasks=tasks))
    stage = dict(kind='qkv', layer=0, tasks=128, ring_slots=2,
                 incoming_order='none', prefix_bytes=0, l2_hint_bytes=0,
                 inactive_zero_bytes_per_cta=0, ctas=ctas,
                 service_descriptor=descriptor)
    return dict(work=dict(grid=132, num_sms=132, requested_residency=1,
                          stages=[stage]), flow=dict(events=events), kinds=kinds)


def fixtures():
    paired = dict(reference_binary_sha256='a' * 64, profiles={
        c: dict(progress_lower_ns=[1, 1], progress_upper_ns=[2, 4], return_ns=6)
        for c in range(128)})
    initial = dict(initialized_before_entry=True, status='fictional_test_only',
                   program_binary_sha256='a' * 64, shared_transfer_inflight=[],
                   shared_resource_available_ns=0, ctas={
                       c: dict(smid=c, initialization_ready_ns=0,
                               producer_entry_ns=0, consumer_entry_ns=0,
                               empty_ready_ns=[0, 0]) for c in range(132)})
    policy = dict(kind='equal_weight_processor_sharing', status='fictional_test_only',
                  cache_preparation='warm_touch', traffic='A+B',
                  request_work_bytes=49152, request_work_ns=2,
                  packet_response_ns=0, propagation_observation_lag_ns=3)
    return initial, paired, policy


class QkvRolloutTest(unittest.TestCase):
    def setUp(self):
        self.source = source_fixture()
        self.initial, self.paired, self.policy = fixtures()

    def run_model(self, **kwargs):
        return adapter.rollout(self.source, self.initial, self.paired, self.policy, **kwargs)

    def test_dynamic_shared_capacity(self):
        self.assertEqual(adapter.fluid_completions([0, 0], 10), [20, 20])
        self.assertEqual(adapter.fluid_completions([0, 5], 10), [15, 20])
        self.assertEqual(adapter.fluid_completions([0, 20], 10), [10, 30])

    def test_connected_response_is_charged_once(self):
        result = self.run_model()
        self.assertEqual({v['returned_ns'] for v in result['ctas']}, {256 * 2 + 3 + 6})
        self.assertIsNone(result['conditional_stage_end_ns'])
        self.assertFalse(result['prediction_available'])
        self.assertFalse(result['solver_eligible'])
        exact, bounds = result['source_event_times_ns'], result['source_event_upper_bounds_ns']
        self.assertEqual(len(exact), 512)
        self.assertEqual(len(bounds), 128)
        self.assertTrue(set(exact).isdisjoint(bounds))
        self.assertTrue(all(self.source['flow']['events'][i]['kind'] ==
                            self.source['kinds']['consumer_return'] for i in bounds))

    def test_initial_state_translation(self):
        before = self.run_model()
        for state in self.initial['ctas'].values():
            for key in ('initialization_ready_ns', 'producer_entry_ns', 'consumer_entry_ns'):
                state[key] += 100
            state['empty_ready_ns'] = [100, 100]
        self.initial['shared_resource_available_ns'] = 100
        self.assertEqual(self.run_model()['conditional_body_end_ns'],
                         before['conditional_body_end_ns'] + 100)

    def test_initial_credit_and_service_causality(self):
        self.initial['ctas'][0]['empty_ready_ns'][1] = 1000
        with self.assertRaises(ValueError):
            self.run_model()
        self.initial['ctas'][0]['initialization_ready_ns'] = 1000
        before = self.run_model()
        self.assertEqual(before['ctas'][0]['packets'][1]['start_ns'], 1000)
        self.policy['request_work_ns'] *= 10
        self.assertGreater(self.run_model()['conditional_body_end_ns'],
                           before['conditional_body_end_ns'])

    def test_observed_future_issue_is_rejected(self):
        self.initial['ctas'][0]['future_issue_ns'] = [0, 999]
        with self.assertRaises(ValueError):
            self.run_model()

    def test_actual_binding_flags_and_rejected_policy(self):
        self.source['work']['stages'][0]['service_descriptor']['plan_binding']['input_norm'] = False
        with self.assertRaises(ValueError):
            self.run_model()
        self.source = source_fixture()
        self.policy['status'] = 'failed_validation'
        self.assertNotIn('conditional_body_end_ns', self.run_model())
        self.policy['traffic'] = 'B_only'
        with self.assertRaises(ValueError):
            self.run_model()

    def test_boundary_service_is_separate(self):
        boundary = dict(status='supplied_calibration', producer_publication_ns=0,
                        cta_join_ns=0, grid_arrival_ns=0, grid_arrival_occupancy_ns=0,
                        grid_release_ns=500, initial_grid_resource_available_ns=0)
        result = self.run_model(boundary=boundary)
        self.assertEqual(result['conditional_stage_end_ns'],
                         result['conditional_body_end_ns'] + 500)
        self.assertFalse(result['prediction_available'])

    def test_paired_response_order(self):
        self.paired['profiles'][0]['progress_upper_ns'] = [8, 4]
        with self.assertRaises(ValueError):
            self.run_model()


if __name__ == '__main__':
    unittest.main()
