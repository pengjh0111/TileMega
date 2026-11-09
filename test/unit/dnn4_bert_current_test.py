#!/usr/bin/env python3
"""Reject incomplete or stale BERT follow-up admissions without GPU execution."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from dnn4_bert_current import select_successors, validate_pp_coordinates


class AdmissionTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.base = Path(self.folder.name)
        self.save('current.json', dict(library=str(self.base/'control.so')))
        self.spec = ('current.json', 'old_pair.json', {'TILEMEGA_F3_REPEAT': '1'}, .03)
        self.save('guard.json', dict(code=0))
        self.processes = [dict(name=f'{tool}-{i+1}', pid=100+index,
            exit_code=0, guard=str(self.base/'guard.json'), binary_sha256='frozen',
            output_hash='same') for index, (tool, i) in enumerate(
                (tool, i) for tool, count in
                [('unit', 5), ('memcheck', 1), ('initcheck', 1),
                 ('racecheck', 5), ('synccheck', 5)] for i in range(count))]
        for name in ['qk_unit', 'pv_unit']:
            self.save(name+'.json', dict(passed=True, processes=self.processes))

    def save(self, name, value):
        path = self.base/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return str(path)

    def candidate(self, mode='qk', reference=None):
        name = f'{mode}_wgmma_b8_v1'
        units = [str(self.base/'qk_unit.json')]
        if mode == 'pv':
            units.append(str(self.base/'pv_unit.json'))
        decision = dict(accepted=True, reference=reference or str(self.base/'control.so'),
            library=str(self.base/(mode+'.so')), gate=str(self.base/(mode+'.gate.json')),
            unit_gates=units, pair=dict(gpu=3, ratio=.98, threshold=.005,
                measurements=[dict(guard=dict(code=0)) for _ in range(6)]))
        self.save(name+'/experiment.json', decision)
        self.save(name+'/final_result.json', copy.deepcopy(decision))
        return name, decision

    def test_pending_and_rejected_keep_current(self):
        name, decision = self.candidate()
        (self.base/name/'final_result.json').unlink()
        self.assertEqual(select_successors(self.base, 8, self.spec), self.spec)
        decision['accepted'] = False
        self.save(name+'/experiment.json', decision)
        self.save(name+'/final_result.json', decision)
        self.assertEqual(select_successors(self.base, 8, self.spec), self.spec)

    def test_complete_successor_chain(self):
        self.candidate()
        name, _ = self.candidate('pv', str(self.base/'qk.so'))
        final, pair, profile, minimum = select_successors(self.base, 8, self.spec)
        self.assertEqual(final, name+'/final_result.json')
        self.assertEqual(pair, name+'/experiment.json')
        self.assertEqual(profile['TILEMEGA_F3_WGMMA_QK'], '1')
        self.assertEqual(profile['TILEMEGA_F3_WGMMA_PV'], '1')
        self.assertEqual(minimum, 0)

    def test_stale_reference_is_rejected(self):
        self.candidate(reference=str(self.base/'unadmitted.so'))
        with self.assertRaisesRegex(ValueError, 'current incumbent'):
            select_successors(self.base, 8, self.spec)

    def test_component_corruption_is_rejected(self):
        self.candidate()
        variants = [self.processes[:-1],
            [dict(p, pid=100) for p in self.processes],
            [dict(p, name='unit-1') for p in self.processes],
            [dict(p, binary_sha256=str(i)) for i, p in enumerate(self.processes)]]
        for processes in variants:
            with self.subTest(processes=processes):
                self.save('qk_unit.json', dict(passed=True, processes=processes))
                with self.assertRaisesRegex(ValueError, 'protocol is incomplete'):
                    select_successors(self.base, 8, self.spec)

    def test_pair_and_final_must_match(self):
        name, decision = self.candidate()
        decision['pair']['ratio'] = .999
        self.save(name+'/experiment.json', decision)
        with self.assertRaisesRegex(ValueError, 'component receipts'):
            select_successors(self.base, 8, self.spec)
        self.save(name+'/final_result.json', decision)
        with self.assertRaisesRegex(ValueError, 'same-card pair'):
            select_successors(self.base, 8, self.spec)

    def test_rejected_guard_is_not_evidence(self):
        self.candidate()
        self.save('guard.json', dict(code=75))
        with self.assertRaisesRegex(ValueError, 'protocol is incomplete'):
            select_successors(self.base, 8, self.spec)

    def test_in_situ_requires_qualified_windows(self):
        _, decision = self.candidate()
        name = 'matrix_existing_retune_b16_v1'
        decision['unit_gates'].append(str(self.base/'pv_unit.json'))
        self.save(name+'/experiment.json', decision)
        self.save(name+'/final_result.json', decision)
        self.save(name+'/tune_selection.json', dict(eligible_variants=[0, 1, 2, 4, 5]))
        self.save(name+'/trace_qualification.json', dict(passed=False, relative_difference=.06))
        with self.assertRaisesRegex(ValueError, 'unqualified in-situ'):
            select_successors(self.base, 16, self.spec)
        self.save(name+'/trace_qualification.json', dict(passed=True, relative_difference=.03))
        self.assertEqual(select_successors(self.base, 16, self.spec)[0], name+'/final_result.json')

    def test_reselection_inherits_accepted_math_profile(self):
        _, decision = self.candidate()
        self.save('matrix_unit.json', dict(passed=True, processes=self.processes))
        decision['unit_gates'] += [str(self.base/'pv_unit.json'), str(self.base/'matrix_unit.json')]
        self.save('qk_wgmma_b16_v1/experiment.json', decision)
        self.save('qk_wgmma_b16_v1/final_result.json', decision)
        tuned = copy.deepcopy(decision)
        tuned['reference'] = decision['library']
        tuned['library'] = str(self.base/'tuned.so')
        name = 'attention_existing_retune_b16_v1'
        self.save(name+'/experiment.json', tuned)
        self.save(name+'/final_result.json', tuned)
        self.save(name+'/trace_qualification.json', dict(passed=True, relative_difference=.03))
        self.save(name+'/tune_selection.json', dict(eligible_variants=[0, 1, 2, 4, 5]))
        final, _, profile, _ = select_successors(self.base, 16, self.spec)
        self.assertEqual(final, name+'/final_result.json')
        self.assertEqual(profile['TILEMEGA_F3_WGMMA_QK'], '1')
        self.assertEqual(profile['TILEMEGA_F3_MATRIX_STORE'], '1')
        self.assertNotIn('TILEMEGA_F3_WGMMA_PV', profile)

    def test_pp_requires_fixed_coordinates_and_three_percent(self):
        _, decision = self.candidate()
        units = [self.save(f'pp_unit_{i}.json', dict(passed=True, processes=self.processes))
                 for i in range(4)]
        decision['unit_gates'] = units
        name = 'ppstore_ffn1_b16_v1'
        stages = [dict(body='other') for _ in range(63)]
        for index in range(4, 60, 5):
            stages[index] = dict(tile_m=128, tile_n=128, tk=64, slots=5,
                                 split_k=1, store='tma', mc=0)
        current = dict(stages=stages)
        candidate = copy.deepcopy(current)
        for index in range(4, 60, 5):
            candidate['stages'][index].update(tile_m=256, sched='pp')
        self.save('control.so.plan.json', dict(native_forward=current))
        self.save('qk.so.plan.json', dict(native_forward=candidate))
        self.save(name+'/experiment.json', decision)
        self.save(name+'/final_result.json', decision)
        with self.assertRaisesRegex(ValueError, 'same-card pair'):
            select_successors(self.base, 16, self.spec)
        decision['pair']['ratio'] = .96
        self.save(name+'/experiment.json', decision)
        self.save(name+'/final_result.json', decision)
        self.assertEqual(select_successors(self.base, 16, self.spec)[3], .03)
        candidate['stages'][5]['slots'] = 6
        self.save('qk.so.plan.json', dict(native_forward=candidate))
        with self.assertRaisesRegex(ValueError, 'outside its fixed FFN1'):
            select_successors(self.base, 16, self.spec)

    def test_pp_normalization_requires_both_inherited_components(self):
        _, decision = self.candidate()
        decision['unit_gates'] = [self.save(f'component_{i}.json',
            dict(passed=True, processes=self.processes)) for i in range(6)]
        stages = [dict(body='other') for _ in range(63)]
        for index in range(4, 60, 5):
            stages[index] = dict(tile_m=128, tile_n=128, tk=64, slots=5,
                                 split_k=1, store='tma', mc=0)
        current = dict(stages=stages)
        candidate = copy.deepcopy(current)
        for index in range(4, 60, 5):
            candidate['stages'][index].update(tile_m=256, sched='pp')
        self.save('control.so.plan.json', dict(native_forward=current))
        self.save('qk.so.plan.json', dict(native_forward=candidate))
        name = 'ppnorm_ffn1_b16_v1'
        decision['pair']['ratio'] = .96
        self.save(name+'/experiment.json', decision)
        self.save(name+'/final_result.json', decision)
        result = select_successors(self.base, 16, self.spec)
        self.assertEqual(result[2]['TILEMEGA_F3_PP_SIDE_NORM'], '1')
        self.assertEqual(result[3], .03)
        decision['unit_gates'].pop()
        self.save(name+'/experiment.json', decision)
        self.save(name+'/final_result.json', decision)
        with self.assertRaisesRegex(ValueError, 'component receipts'):
            select_successors(self.base, 16, self.spec)
        self.save('control.so.plan.json', dict(native_forward=candidate))
        validate_pp_coordinates(self.base/'qk.so', self.base/'control.so', allow_existing=True)
        candidate['stages'][6]['slots'] = 3
        self.save('qk.so.plan.json', dict(native_forward=candidate))
        with self.assertRaisesRegex(ValueError, 'outside its fixed FFN1'):
            validate_pp_coordinates(self.base/'qk.so', self.base/'control.so', allow_existing=True)

    def test_wg64_requires_all_components_and_fixed_slots(self):
        _, decision = self.candidate()
        decision['unit_gates'] = [self.save(f'wg64_component_{i}.json',
            dict(passed=True, processes=self.processes)) for i in range(8)]
        stages = [dict(body='other') for _ in range(63)]
        for index in range(4, 60, 5):
            stages[index] = dict(tile_m=128, tile_n=128, tk=64, slots=5,
                                 split_k=1, store='tma', mc=0)
        current = dict(stages=stages)
        candidate = copy.deepcopy(current)
        for index in range(4, 60, 5):
            candidate['stages'][index].update(slots=7, sched='pp')
        self.save('control.so.plan.json', dict(native_forward=current))
        self.save('qk.so.plan.json', dict(native_forward=candidate))
        name = 'pp64_ffn1_b16_v1'
        decision['pair']['ratio'] = .96
        self.save(name+'/experiment.json', decision)
        self.save(name+'/final_result.json', decision)
        result = select_successors(self.base, 16, self.spec)
        self.assertEqual(result[3], .03)
        candidate['stages'][4]['slots'] = 6
        self.save('qk.so.plan.json', dict(native_forward=candidate))
        with self.assertRaisesRegex(ValueError, 'outside its fixed FFN1'):
            select_successors(self.base, 16, self.spec)
        decision['unit_gates'].pop()
        self.save(name+'/experiment.json', decision)
        self.save(name+'/final_result.json', decision)
        with self.assertRaisesRegex(ValueError, 'component receipts'):
            select_successors(self.base, 16, self.spec)

    def test_attention_store_preserves_all_coordinates(self):
        _, decision = self.candidate()
        (self.base/'qk_wgmma_b8_v1/final_result.json').unlink()
        current = dict(stages=[dict(body='other') for _ in range(63)])
        self.save('control.so.plan.json', dict(native_forward=current))
        self.save('qk.so.plan.json', dict(native_forward=current))
        name = 'attention_store_b8_v1'
        self.save(name+'/experiment.json', decision)
        self.save(name+'/final_result.json', decision)
        result = select_successors(self.base, 8, self.spec)
        self.assertEqual(result[2]['TILEMEGA_F3_ATTENTION_VECTOR_STORE'], '1')
        self.assertEqual(result[3], 0)
        current['stages'][4]['slots'] = 7
        self.save('qk.so.plan.json', dict(native_forward=current))
        with self.assertRaisesRegex(ValueError, 'changed the current coordinates'):
            select_successors(self.base, 8, self.spec)

    def test_attention_bundle_needs_joint_component_and_same_coordinates(self):
        _, decision = self.candidate()
        (self.base/'qk_wgmma_b8_v1/final_result.json').unlink()
        decision['unit_gates'] = [self.save(f'joint_component_{i}.json',
            dict(passed=True, processes=self.processes)) for i in range(4)]
        current = dict(stages=[dict(body='other') for _ in range(63)])
        self.save('control.so.plan.json', dict(native_forward=current))
        self.save('qk.so.plan.json', dict(native_forward=current))
        name = 'attention_bundle_b8_v1'
        self.save(name+'/experiment.json', decision)
        self.save(name+'/final_result.json', decision)
        profile = select_successors(self.base, 8, self.spec)[2]
        for feature in ('TILEMEGA_F3_WGMMA_QK', 'TILEMEGA_F3_WGMMA_PV',
                        'TILEMEGA_F3_ATTENTION_VECTOR_STORE'):
            self.assertEqual(profile[feature], '1')
        decision['unit_gates'].pop()
        self.save(name+'/experiment.json', decision)
        self.save(name+'/final_result.json', decision)
        with self.assertRaisesRegex(ValueError, 'component receipts'):
            select_successors(self.base, 8, self.spec)
        decision['unit_gates'].append(str(self.base/'joint_component_3.json'))
        self.save(name+'/experiment.json', decision)
        self.save(name+'/final_result.json', decision)
        current['stages'][4]['slots'] = 7
        self.save('qk.so.plan.json', dict(native_forward=current))
        with self.assertRaisesRegex(ValueError, 'changed the current coordinates'):
            select_successors(self.base, 8, self.spec)

    def test_ffn1_only_candidate_requires_matrix_components(self):
        _, decision = self.candidate()
        (self.base/'qk_wgmma_b8_v1/final_result.json').unlink()
        decision['unit_gates'] = [self.save(f'ffn_component_{i}.json',
            dict(passed=True, processes=self.processes)) for i in range(2)]
        current = dict(stages=[dict(body='other') for _ in range(63)])
        candidate = copy.deepcopy(current)
        for index in range(4, 60, 5):
            candidate['stages'][index].update(tile_m=128, tile_n=192, tk=64,
                slots=4, split_k=1, store='tma', mc=0)
        self.save('control.so.plan.json', dict(native_forward=current))
        self.save('qk.so.plan.json', dict(native_forward=candidate))
        name = 'ffn1_tma192_matrix_b8_v1'
        self.save(name+'/experiment.json', decision)
        self.save(name+'/final_result.json', decision)
        profile = select_successors(self.base, 8, self.spec)[2]
        self.assertEqual(profile['TILEMEGA_F3_MATRIX_STORE'], '1')
        self.assertEqual(profile['TILEMEGA_F3_REGISTER_BUDGET'], '3')
        candidate['stages'][1]['tile_n'] = 192
        self.save('qk.so.plan.json', dict(native_forward=candidate))
        with self.assertRaisesRegex(ValueError, 'outside its fixed group'):
            select_successors(self.base, 8, self.spec)
        decision['unit_gates'].pop()
        self.save(name+'/experiment.json', decision)
        self.save(name+'/final_result.json', decision)
        with self.assertRaisesRegex(ValueError, 'component receipts'):
            select_successors(self.base, 8, self.spec)

    def test_ffn2_only_candidate_needs_both_new_components(self):
        _, decision = self.candidate()
        decision['unit_gates'] = [self.save(f'ffn2_component_{i}.json',
            dict(passed=True, processes=self.processes)) for i in range(4)]
        current = dict(stages=[dict(body='other') for _ in range(63)])
        candidate = copy.deepcopy(current)
        for index in range(5, 61, 5):
            candidate['stages'][index].update(tile_m=256, tile_n=64, tk=64,
                slots=4, split_k=1, store='tma', mc=0)
        self.save('control.so.plan.json', dict(native_forward=current))
        self.save('qk.so.plan.json', dict(native_forward=candidate))
        name = 'ffn2_tma256x64_b16_v1'
        self.save(name+'/experiment.json', decision)
        self.save(name+'/final_result.json', decision)
        self.assertEqual(select_successors(self.base, 16, self.spec)[0],
                         name+'/final_result.json')
        candidate['stages'][4]['tile_m'] = 256
        self.save('qk.so.plan.json', dict(native_forward=candidate))
        with self.assertRaisesRegex(ValueError, 'outside its fixed group'):
            select_successors(self.base, 16, self.spec)
        decision['unit_gates'].pop()
        self.save(name+'/experiment.json', decision)
        self.save(name+'/final_result.json', decision)
        with self.assertRaisesRegex(ValueError, 'component receipts'):
            select_successors(self.base, 16, self.spec)


if __name__ == '__main__':
    unittest.main()
