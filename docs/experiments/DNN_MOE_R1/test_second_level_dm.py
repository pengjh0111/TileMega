#!/usr/bin/env python3
"""Host protocol tests; synthetic records are not GPU measurements."""
import unittest

from second_level_dm import BudgetExhausted, ForwardRace, family_shortlist


def candidate(i, family='dense', score=None):
    key = f'{i:064x}'
    return dict(identity_id=key, correctness_identity=key, correctness_passed=True,
                family=family, role='solver_candidate', model_ns=score or i+1,
                spill=bool(i%2))


def observation(request, value, process):
    return dict(phase=request['phase'], round=request['round'], identity_id=request['identity_id'],
                median_ns=value, process_id=process, guarded=True, correctness_passed=True)


class SecondLevelTests(unittest.TestCase):
    def test_families_have_independent_top_m(self):
        rows = [candidate(i, family, score) for i, family, score in
                [(0, 'dense', 1), (1, 'dense', 2), (2, 'dense', 3),
                 (3, 'pages', 100), (4, 'pages', 101), (5, 'pages', 102)]]
        self.assertEqual([r['identity_id'] for r in family_shortlist(rows, 2)],
                         [f'{i:064x}' for i in (0, 1, 3, 4)])

    def test_half_elimination_and_three_alternating_rounds(self):
        rows = [candidate(i, 'family'+str(i%4)) for i in range(8)]
        race = ForwardRace(rows, 2, 1800, now=lambda: 0)
        costs = {f'{i:064x}': 100+i for i in range(8)}
        # Model ranking deliberately disagrees with these synthetic timings.
        costs[f'{7:064x}'] = 1
        while request := race.next_request():
            race.record(observation(request, costs[request['identity_id']],
                                    f'host-fixture:{len(race.records)}'))
        result = race.result()
        self.assertEqual(result['winner'], f'{7:064x}')
        self.assertEqual([len(r['eliminated']) for r in result['eliminations']], [4, 1])
        final = [r for r in result['measurements'] if r['phase'] == 'final']
        self.assertEqual(len(final), 9)
        self.assertEqual([r['identity_id'] for r in final[3:6]],
                         [r['identity_id'] for r in final[1:3]]+[final[0]['identity_id']])
        self.assertEqual(set(r['identity_id'] for r in result['candidates']), set(costs))
        self.assertIsNone(race.next_request())

    def test_final_uses_three_round_medians_including_spills(self):
        race = ForwardRace([candidate(0, 'mma'), candidate(1, 'gemv')], 1, 10, now=lambda: 0)
        # The single noisy first round cannot displace candidate 1, whose
        # unchanged identity also records a spill. Spills do not veto timing.
        samples = {f'{0:064x}': [1, 100, 100], f'{1:064x}': [10, 10, 10]}
        while request := race.next_request():
            race.record(observation(request, samples[request['identity_id']][request['round']],
                                    f'host-fixture:{len(race.records)}'))
        self.assertEqual(race.result()['winner'], f'{1:064x}')

    def test_expiry_includes_prior_build_time_and_never_seals(self):
        now = [8]
        race = ForwardRace([candidate(0)], 1, 10, started_at=0, now=lambda: now[0])
        request = race.next_request()
        now[0] = 10
        with self.assertRaises(BudgetExhausted):
            race.record(observation(request, 1, 'host-fixture:0'))
        self.assertIsNone(race.winner)
        with self.assertRaises(ValueError):
            race.result()

    def test_controls_failed_gates_and_identity_drift_are_rejected(self):
        for change in [dict(role='control'), dict(correctness_passed=False),
                       dict(correctness_identity='f'*64), dict(family=''), dict(model_ns=float('nan'))]:
            row = candidate(0);row.update(change)
            with self.assertRaises(ValueError):
                family_shortlist([row], 1)
        with self.assertRaises(ValueError):
            family_shortlist([candidate(0), candidate(0)], 1)

    def test_wrong_round_unprotected_jobs_and_reused_processes_fail(self):
        race = ForwardRace([candidate(0)], 1, 10, now=lambda: 0)
        request = race.next_request()
        for change in [dict(round=1), dict(identity_id='f'*64), dict(guarded=False),
                       dict(correctness_passed=False), dict(median_ns=0)]:
            row = observation(request, 1, 'host-fixture:0');row.update(change)
            with self.assertRaises(ValueError):
                race.record(row)
        race.record(observation(request, 1, 'host-fixture:0'))
        with self.assertRaises(ValueError):
            race.record(observation(race.next_request(), 1, 'host-fixture:0'))


if __name__ == '__main__':
    unittest.main()
