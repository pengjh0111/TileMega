#!/usr/bin/env python3
"""Reject incomplete or stale BERT follow-up admissions without GPU execution."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from dnn4_bert_current import (select_successors, validate_pp_coordinates,
    validate_boundary_selection, validate_intermittent, validate_timing_gpu,
    validate_followup_allocation, validate_whole_program_selection,
    validate_fixed_grid_integration)


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

    def test_ffn1_tma_keeps_current_profile_and_other_stages(self):
        for batch, columns, slots, name, count in [
                (4, 128, 5, 'ffn1_tma128_phase72_b4_admitted_v1', 1),
                (8, 128, 5, 'ffn1_tma128_matrix_b8_v1', 3),
                (16, 256, 3, 'ffn1_tma256_matrix_b16_v2', 4)]:
            with self.subTest(batch=batch):
                _, decision = self.candidate()
                (self.base/'qk_wgmma_b8_v1/final_result.json').unlink()
                decision['unit_gates'] = [self.save(f'tma_component_{i}/result.json',
                    dict(passed=True, processes=self.processes)) for i in range(count)]
                flags = [f'-DDNN4_FFN1_TMA{columns}_TEST=1',
                    '-DTILEMEGA_F3_SIDE_NORM=1',
                    f'-DTILEMEGA_F3_REGISTER_BUDGET={2 if columns == 128 else 3}']
                if batch == 8:
                    flags += ['-DDNN4_FFN1_TMA128_B8_TEST=1',
                              '-DTILEMEGA_F3_MATRIX_STORE=1']
                if batch == 16:
                    flags += ['-DTILEMEGA_F3_MATRIX_STORE=1',
                              '-DTILEMEGA_F3_CYCLIC_SLOTS=1']
                build = self.save('tma_build.json', dict(command=flags,
                    binary_sha256='frozen'))
                self.save('tma_component_0/plan.json', dict(build=build))
                current = dict(stages=[dict(body='gemm', slots=6) for _ in range(63)])
                candidate = copy.deepcopy(current)
                for index in range(4, 60, 5):
                    candidate['stages'][index].update(tile_m=128, tile_n=columns,
                        tk=64, slots=slots, split_k=1, store='tma', mc=0)
                self.save('control.so.plan.json', dict(native_forward=current))
                self.save('qk.so.plan.json', dict(native_forward=candidate))
                self.save(name+'/experiment.json', decision)
                self.save(name+'/final_result.json', decision)
                profile = dict(self.spec[2], TILEMEGA_F3_SIDE_NORM='1',
                    TILEMEGA_F3_REGISTER_BUDGET='2' if columns == 128 else '3')
                if batch == 8:
                    profile['TILEMEGA_F3_MATRIX_STORE'] = '1'
                if batch == 16:
                    profile.update(TILEMEGA_F3_MATRIX_STORE='1',
                        TILEMEGA_F3_ATTENTION_VECTOR_STORE='1')
                spec = (self.spec[0], self.spec[1], profile, 0)
                result = select_successors(self.base, batch, spec)
                self.assertEqual(result[0], name+'/final_result.json')
                self.assertEqual(result[2], profile)
                self.save('tma_build.json', dict(command=flags[:-1],
                    binary_sha256='frozen'))
                with self.assertRaisesRegex(ValueError, 'exact compiled profile'):
                    select_successors(self.base, batch, spec)
                self.save('tma_build.json', dict(command=flags, binary_sha256='frozen'))
                candidate['stages'][5]['slots'] = 3
                self.save('qk.so.plan.json', dict(native_forward=candidate))
                with self.assertRaisesRegex(ValueError, 'outside its fixed group'):
                    select_successors(self.base, batch, spec)
                self.save(name+'/experiment.json', dict(decision, accepted=False))

    def test_qkv_requires_exact_components_and_preserves_other_groups(self):
        _, decision = self.candidate()
        decision['unit_gates'] = [self.save(f'qkv_component_{i}/result.json',
            dict(passed=True, processes=self.processes)) for i in range(4)]
        flags = ['-DDNN4_QKV_TMA192_TEST=1', '-DTILEMEGA_F3_SIDE_NORM=1',
                 '-DTILEMEGA_F3_REGISTER_BUDGET=3', '-DTILEMEGA_F3_MATRIX_STORE=1',
                 '-DTILEMEGA_F3_CYCLIC_SLOTS=1']
        build = self.save('qkv_build.json', dict(command=flags, binary_sha256='frozen'))
        self.save('qkv_component_0/plan.json', dict(build=build))
        current = dict(stages=[dict(body='other') for _ in range(63)])
        candidate = copy.deepcopy(current)
        for index in range(1, 60, 5):
            candidate['stages'][index].update(tile_m=128, tile_n=192, tk=64,
                slots=4, split_k=1, store='tma', mc=0)
        self.save('control.so.plan.json', dict(native_forward=current))
        self.save('qk.so.plan.json', dict(native_forward=candidate))
        name = 'qkv_tma192_matrix_b16_v1'
        self.save(name+'/experiment.json', decision)
        self.save(name+'/final_result.json', decision)
        profile = dict(self.spec[2], TILEMEGA_F3_MATRIX_STORE='1',
                       TILEMEGA_F3_ATTENTION_VECTOR_STORE='1')
        spec = (self.spec[0], self.spec[1], profile, 0)
        self.assertEqual(select_successors(self.base, 16, spec)[0], name+'/final_result.json')
        self.save('qkv_build.json', dict(command=flags[:-1], binary_sha256='frozen'))
        with self.assertRaisesRegex(ValueError, 'exact compiled profile'):
            select_successors(self.base, 16, spec)
        self.save('qkv_build.json', dict(command=flags, binary_sha256='wrong'))
        with self.assertRaisesRegex(ValueError, 'exact compiled profile'):
            select_successors(self.base, 16, spec)
        self.save('qkv_build.json', dict(command=flags, binary_sha256='frozen'))
        candidate['stages'][4]['slots'] = 3
        self.save('qk.so.plan.json', dict(native_forward=candidate))
        with self.assertRaisesRegex(ValueError, 'outside its fixed group'):
            select_successors(self.base, 16, spec)

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

    def test_b4_cyclic_requires_both_components_and_identical_coordinates(self):
        _, decision = self.candidate()
        decision['unit_gates'] += [str(self.base/'pv_unit.json')]
        current = dict(stages=[dict(body='other') for _ in range(63)])
        self.save('control.so.plan.json', dict(native_forward=current))
        self.save('qk.so.plan.json', dict(native_forward=current))
        name = 'cyclic_phase72_b4_v1'
        self.save(name+'/experiment.json', decision)
        self.save(name+'/final_result.json', decision)
        result = select_successors(self.base, 4, self.spec)
        self.assertEqual(result[2]['TILEMEGA_F3_CYCLIC_SLOTS'], '1')
        self.assertEqual(result[3], 0)
        current['stages'][4]['slots'] = 3
        self.save('qk.so.plan.json', dict(native_forward=current))
        with self.assertRaisesRegex(ValueError, 'changed the current coordinates'):
            select_successors(self.base, 4, self.spec)
        decision['unit_gates'].pop()
        self.save(name+'/experiment.json', decision)
        self.save(name+'/final_result.json', decision)
        with self.assertRaisesRegex(ValueError, 'component receipts'):
            select_successors(self.base, 4, self.spec)

    def test_b2_ffn1_requires_exact_group_and_three_strict_components(self):
        _, decision = self.candidate()
        decision['unit_gates'] = [self.save(f'ffn1_reg_component_{i}.json',
            dict(passed=True, processes=self.processes)) for i in range(3)]
        current = dict(stages=[dict(body='other') for _ in range(63)])
        candidate = copy.deepcopy(current)
        for index in range(4, 60, 5):
            candidate['stages'][index].update(tile_m=128, tile_n=64, tk=64,
                slots=8, split_k=1, store='reg', mc=0)
        self.save('control.so.plan.json', dict(native_forward=current))
        self.save('qk.so.plan.json', dict(native_forward=candidate))
        name = 'ffn1_reg64_side216_b2_admitted_v1'
        self.save(name+'/experiment.json', decision)
        self.save(name+'/final_result.json', decision)
        profile = select_successors(self.base, 2, self.spec)[2]
        self.assertEqual(profile['TILEMEGA_F3_SIDE_NORM'], '1')
        self.assertEqual(profile['TILEMEGA_F3_REGISTER_BUDGET'], '3')
        candidate['stages'][5]['tile_n'] = 64
        self.save('qk.so.plan.json', dict(native_forward=candidate))
        with self.assertRaisesRegex(ValueError, 'outside its fixed group'):
            select_successors(self.base, 2, self.spec)
        decision['unit_gates'].pop()
        self.save(name+'/experiment.json', decision)
        self.save(name+'/final_result.json', decision)
        with self.assertRaisesRegex(ValueError, 'component receipts'):
            select_successors(self.base, 2, self.spec)

    def test_b2_ffn2_preserves_its_admitted_ffn1_predecessor(self):
        _, decision = self.candidate()
        units = [self.save(f'ffn_chain_component_{i}.json',
            dict(passed=True, processes=self.processes)) for i in range(4)]
        current = dict(stages=[dict(body='other') for _ in range(63)])
        first = copy.deepcopy(current)
        for index in range(4, 60, 5):
            first['stages'][index].update(tile_m=128, tile_n=64, tk=64,
                slots=8, split_k=1, store='reg', mc=0)
        second = copy.deepcopy(first)
        for index in range(5, 61, 5):
            second['stages'][index].update(tile_m=128, tile_n=64, tk=64,
                slots=8, split_k=4, store='reg', mc=0)
        self.save('control.so.plan.json', dict(native_forward=current))
        self.save('qk.so.plan.json', dict(native_forward=first))
        self.save('split.so.plan.json', dict(native_forward=second))
        prior = 'ffn1_reg64_side216_b2_admitted_v1'
        decision['unit_gates'] = units[:3]
        self.save(prior+'/experiment.json', decision)
        self.save(prior+'/final_result.json', decision)
        successor = copy.deepcopy(decision)
        successor.update(reference=decision['library'], library=str(self.base/'split.so'),
                         unit_gates=units)
        name = 'ffn2_reg64_split4_b2_v1'
        self.save(name+'/experiment.json', successor)
        self.save(name+'/final_result.json', successor)
        self.assertEqual(select_successors(self.base, 2, self.spec)[0],
                         name+'/final_result.json')
        second['stages'][4]['slots'] = 7
        self.save('split.so.plan.json', dict(native_forward=second))
        with self.assertRaisesRegex(ValueError, 'outside its fixed group'):
            select_successors(self.base, 2, self.spec)
        successor['unit_gates'].pop()
        self.save(name+'/experiment.json', successor)
        self.save(name+'/final_result.json', successor)
        with self.assertRaisesRegex(ValueError, 'component receipts'):
            select_successors(self.base, 2, self.spec)


class BoundaryAdmissionTest(unittest.TestCase):
    setUp = AdmissionTest.setUp
    save = AdmissionTest.save

    def fixture(self):
        folder = self.base/'boundary_cache216_b4_admitted_v1'
        current = dict(schema=4, stages=[dict(body='embedding'),
            dict(body='gemm', tile_n=64), dict(body='layer_norm')])
        self.save('control.so.plan.json', dict(native_forward=current))
        variants = []
        for columns in [32, 64, 96, 128, 192, 256]:
            candidate = copy.deepcopy(current)
            candidate['stages'][1]['tile_n'] = columns
            variants.append(candidate)
        assembled = variants[3]
        specifications = []
        for index, (name, coordinate) in enumerate([
                ('tuned', assembled), ('existing_variant_0', variants[0]),
                ('existing_variant_1', variants[1]), ('existing_variant_5', variants[5])]):
            library = self.base/(name+'.so')
            self.save(library.name+'.plan.json', dict(native_forward=coordinate))
            traces = []
            for process in range(3):
                trace = self.base/f'windows-{index}-{process}'
                self.save(str(trace/'samples.json'),
                          [[dict(stage_ns=100+10*index)] for _ in range(50)])
                traces.append(str(trace))
            receipt = self.save(f'windows-{index}.json', dict(status='passed', traces=traces,
                processes=[dict(pid=200+i, exit_code=0) for i in range(3)]))
            specifications.append(dict(name=name, library=str(library), tune_priority=index,
                tune_window_ns=100+10*index, tune_window_source=receipt))
        self.save(str(folder/'tune_inputs.json'),
                  dict(groups=[dict(stages=[1])], variants=variants))
        self.save(str(folder/'trace_qualification.json'),
                  dict(passed=True, relative_difference=.018))
        self.save(str(folder/'tune_selection.json'), dict(eligible_variants=[0, 1, 3, 5],
            coordinates=assembled, groups=[dict(group=0, stages=[1], candidate=3,
                windows_ns=[70, 80, None, 60, None, 90])]))
        self.save(str(folder/'candidate_spec.json'), specifications)
        self.save(str(folder/'selection.json'), dict(winner_library=specifications[0]['library'],
            measurements=[dict(arm=arm, guard=dict(code=0)) for arm in 'ABCDDCBA']))
        self.save(str(folder/'instruction_proof.json'), dict(passed=True, sass_words=123,
            prototype_library=specifications[0]['library']))
        return folder, specifications[0]['library'], str(self.base/'control.so')

    def test_qualified_selection_and_main_proof(self):
        validate_boundary_selection(*self.fixture())

    def test_successor_requires_explicit_current_profile_units(self):
        folder, library, reference = self.fixture()
        units = []
        for name, flags in [('alias', ['-DDNN4_ALIAS_REPEAT_TEST=1',
                                       '-DTILEMEGA_F3_REPEAT_ARITHMETIC=1']),
                            ('side', ['-DTILEMEGA_F3_SIDE_NORM=1'])]:
            build = self.save(name+'/build.json', dict(binary_sha256='frozen',
                command=flags+['-DTILEMEGA_F3_REGISTER_BUDGET=3']))
            self.save(name+'/plan.json', dict(build=build))
            units.append(self.save(name+'/result.json',
                                   dict(passed=True, processes=self.processes)))
        decision = dict(accepted=True, library=library, reference=reference,
            gate=str(self.base/'gate.json'), unit_gates=units,
            pair=dict(gpu=3, ratio=.96, threshold=.01,
                      measurements=[dict(guard=dict(code=0)) for _ in range(6)]))
        self.save(str(folder/'experiment.json'), decision)
        self.save(str(folder/'final_result.json'), decision)
        selected = select_successors(self.base, 4, self.spec)
        self.assertEqual(selected[0], folder.name+'/final_result.json')
        self.assertEqual(selected[2]['TILEMEGA_F3_REGISTER_BUDGET'], '3')
        build = json.loads((self.base/'alias/build.json').read_text())
        build['command'].remove('-DTILEMEGA_F3_REGISTER_BUDGET=3')
        self.save('alias/build.json', build)
        with self.assertRaisesRegex(ValueError, 'exact compiled profile'):
            select_successors(self.base, 4, self.spec)

    def test_missing_coverage_or_unqualified_windows(self):
        folder, library, reference = self.fixture()
        path = folder/'trace_qualification.json'
        self.save(str(path), dict(passed=True, relative_difference=.051))
        with self.assertRaisesRegex(ValueError, 'four-variant coverage'):
            validate_boundary_selection(folder, library, reference)
        self.save(str(path), dict(passed=True, relative_difference=.018))
        path = folder/'tune_selection.json';selection = json.loads(path.read_text())
        selection['eligible_variants'] = [0, 1, 3]
        self.save(str(path), selection)
        with self.assertRaisesRegex(ValueError, 'four-variant coverage'):
            validate_boundary_selection(folder, library, reference)

    def test_changed_unselected_stage_is_rejected(self):
        folder, library, reference = self.fixture()
        path = folder/'tune_selection.json';selection = json.loads(path.read_text())
        selection['coordinates']['stages'][0]['tile_n'] = 64
        self.save(str(path), selection)
        with self.assertRaisesRegex(ValueError, 'unselected coordinates'):
            validate_boundary_selection(folder, library, reference)

    def test_unmeasured_tie_priority_or_missing_main_proof_is_rejected(self):
        folder, library, reference = self.fixture()
        path = folder/'candidate_spec.json';specs = json.loads(path.read_text())
        specs[0]['tune_window_ns'] = 1
        self.save(str(path), specs)
        with self.assertRaisesRegex(ValueError, 'measured windows'):
            validate_boundary_selection(folder, library, reference)
        specs[0]['tune_window_ns'] = 100;specs[0]['tune_priority'] = 3
        self.save(str(path), specs)
        with self.assertRaisesRegex(ValueError, 'tie priority was not measured'):
            validate_boundary_selection(folder, library, reference)
        specs[0]['tune_priority'] = 0;self.save(str(path), specs)
        self.save(str(folder/'instruction_proof.json'),
                  dict(passed=False, sass_words=123, prototype_library=library))
        with self.assertRaisesRegex(ValueError, 'instruction proof'):
            validate_boundary_selection(folder, library, reference)


class Gpu0ContinuationTest(unittest.TestCase):
    setUp = AdmissionTest.setUp
    save = AdmissionTest.save

    def timing(self, order):
        rows = []
        for index, arm in enumerate(order):
            guard = self.base/f'clocks-{index}.jsonl'
            guard.write_text('\n'.join(json.dumps(dict(gpu_index=0, label=label))
                for label in ('before-measurement', 'after-measurement')))
            path = self.save(f'measure-{index}.json', dict(gpu_index=0, pid=1000+index,
                median_ms=1, samples_ms=[1]*200, guard_log=str(guard),
                measurement=dict(warmup=50, iterations=200)))
            rows.append(dict(arm=arm, median_ms=1, measurement=path, guard=dict(code=0)))
        return dict(gpu=0, measurements=rows)

    def reservations(self, count):
        out = self.base/'scheduler'
        sources = []
        for index in range(count):
            job = out/'jobs'/str(index)
            attempt = job/'raw/attempt1'
            self.save(str(attempt/'guard_result.json'), dict(code=0))
            self.save(str(job/'queue/queue_single.json'), [dict(
                command=['/root/python/worker.py'],
                env=dict(CUDA_VISIBLE_DEVICES='0', TILEMEGA_DEVICE_INDEX='0'))])
            sources.append(self.save(str(job/'scheduler/state.json'), {str(index): dict(
                status='done', exit_code=0, out=str(attempt), started=70*index,
                ended=70*index+1)}))
        self.save(str(out/'intermittent.json'), dict(gpu=0, jobs=count,
            minimum_gap_s=30, source_states=sources))
        return out, sources

    def test_gpu0_individual_receipts_and_complete_pair(self):
        validate_timing_gpu(self.timing('ABABAB'), 0, 'ABABAB')
        folder, _ = self.reservations(17)
        validate_intermittent(folder, 17)

    def test_mixed_card_incomplete_process_and_reused_pid_are_rejected(self):
        result = self.timing('ABABAB')
        path = Path(result['measurements'][1]['measurement'])
        measured = json.loads(path.read_text())
        measured['gpu_index'] = 3;self.save(str(path), measured)
        with self.assertRaisesRegex(ValueError, 'same-card timing'):
            validate_timing_gpu(result, 0, 'ABABAB')
        measured['gpu_index'] = 0;measured['samples_ms'].pop();self.save(str(path), measured)
        with self.assertRaisesRegex(ValueError, 'complete same-card timing'):
            validate_timing_gpu(result, 0, 'ABABAB')
        measured['samples_ms'].append(1);measured['pid'] = 1000;self.save(str(path), measured)
        with self.assertRaisesRegex(ValueError, 'reused a timing process'):
            validate_timing_gpu(result, 0, 'ABABAB')

    def test_missing_gap_grouped_job_and_original_entry_are_rejected(self):
        folder, sources = self.reservations(2)
        state = json.loads(Path(sources[1]).read_text());state['1']['started'] = 20
        self.save(sources[1], state)
        with self.assertRaisesRegex(ValueError, 'release the card'):
            validate_intermittent(folder, 2)
        state['1']['started'] = 70;state['extra'] = state['1'];self.save(sources[1], state)
        with self.assertRaisesRegex(ValueError, 'grouped reservation'):
            validate_intermittent(folder, 2)
        del state['extra'];self.save(sources[1], state)
        queue = Path(sources[1]).parent.parent/'queue/queue_single.json'
        job = json.loads(queue.read_text());job[0]['command'] = ['/root/repo/test.py']
        self.save(str(queue), job)
        with self.assertRaisesRegex(ValueError, 'invalid per-process receipt'):
            validate_intermittent(folder, 2)

    def test_gpu0_exception_requires_b4_and_explicit_authorization(self):
        decision = dict(pair=dict(gpu=0))
        with self.assertRaisesRegex(ValueError, 'same-card pair'):
            validate_followup_allocation(self.base, 'qk_wgmma_b8_v1', decision, {})
        authorization = self.save('authorization.json', dict(allowed_gpu_indices=[3],
            minimum_idle_gap_s=30, python_entry_root='/root/python'))
        self.save('allocation.json', dict(gpu=0, original_anchor_gpu=3,
            reanchored=False, authorization=authorization))
        with self.assertRaisesRegex(ValueError, 'allocation provenance'):
            validate_followup_allocation(self.base, 'boundary_cache216_b4_admitted_v1', decision, {})


class WholeProgramAdmissionTest(unittest.TestCase):
    setUp = AdmissionTest.setUp
    save = AdmissionTest.save

    def fixture(self):
        folder = self.base/'whole_program216_b8_admitted_v1'
        variants = [dict(stages=[dict(body='gemm', tile_n=n)])
                    for n in [32, 64, 96, 128, 192, 256]]
        specifications = []
        for index, name in enumerate(['current']+[f'existing_variant_{i}' for i in range(3)]):
            library = self.base/('control.so' if index == 0 else name+'.so')
            self.save(library.name+'.plan.json', dict(native_forward=variants[5 if index == 0 else index-1]))
            self.save(library.name+'.identity.json', dict(artifact_id=f'id-{index}'))
            Path(str(library)+'.cu').write_text('#define TILEMEGA_F3_REGISTER_BUDGET 3\n')
            gate = self.base/f'correctness-{index}/corr-1/check.json'
            self.save(str(gate.parent.parent/'result.json'), dict(status='passed', passes='5/5',
                execution=dict(artifact_id=f'id-{index}'),
                processes=[dict(pid=100*index+p, exit_code=0) for p in range(5)]))
            proof = self.save(f'proof-{index}.json', dict(passed=True, sass_words=123,
                prototype_library=str(library)))
            specifications.append(dict(name=name, library=str(library), gate=str(gate),
                instruction_proof=proof, tune_priority=index))
        self.save(str(folder/'candidate_spec.json'), specifications)
        self.save(str(folder/'tune_inputs.json'), dict(variants=variants))
        rows, ranking = [], []
        for index, arm in enumerate('ABCDDCBA'):
            candidate = ord(arm)-ord('A'); spec = specifications[candidate]
            median = [1.0, .8, .9, 1.1][candidate]
            guard = self.base/f'clocks-{index}.jsonl'
            guard.write_text('\n'.join(json.dumps(dict(gpu_index=3, label=label))
                for label in ('before-measurement', 'after-measurement')))
            measured = self.save(f'measure-{index}.json', dict(gpu_index=3, pid=1000+index,
                median_ms=median, samples_ms=[median]*200, guard_log=str(guard),
                measurement=dict(warmup=50, iterations=200), library_path=spec['library'],
                numerical_gate_path=spec['gate'], artifact_id=f'id-{candidate}'))
            rows.append(dict(arm=arm, median_ms=median, measurement=measured, guard=dict(code=0)))
        for index in [1, 2, 0, 3]:
            ranking.append(dict(arm='ABCD'[index], library=specifications[index]['library'],
                median_ms=[1.0, .8, .9, 1.1][index], process_range_ratio=0.0,
                tune_priority=index))
        self.save(str(folder/'selection.json'), dict(protocol='P-SEL', batch=8, gpu=3,
            measurements=rows, ranking=ranking, threshold=.005, tied=False,
            candidate_ms=.8, winner='B', winner_library=specifications[1]['library']))
        return folder, specifications[1]['library'], specifications[0]['library']

    def test_complete_whole_program_selection_without_trace_assembly(self):
        validate_whole_program_selection(*self.fixture())

    def test_worktree_library_alias_matches_measured_canonical_path(self):
        folder, library, reference = self.fixture()
        alias = self.base/'worktree'
        alias.symlink_to(self.base, target_is_directory=True)
        path = folder/'candidate_spec.json';specs = json.loads(path.read_text())
        specs[0]['library'] = str(alias/Path(reference).name)
        self.save(str(path), specs)
        validate_whole_program_selection(folder, library, reference)

    def test_assembled_or_changed_coordinates_are_rejected(self):
        folder, library, reference = self.fixture()
        path = folder/'candidate_spec.json';specs = json.loads(path.read_text())
        specs[1]['name'] = 'tuned';self.save(str(path), specs)
        with self.assertRaisesRegex(ValueError, 'assembled candidate'):
            validate_whole_program_selection(folder, library, reference)
        specs[1]['name'] = 'existing_variant_0';self.save(str(path), specs)
        self.save(library+'.plan.json', dict(native_forward=dict(stages=[])))
        with self.assertRaisesRegex(ValueError, 'fixed candidate coordinates'):
            validate_whole_program_selection(folder, library, reference)

    def test_tie_claim_and_wrong_winner_are_rejected(self):
        folder, library, reference = self.fixture()
        path = folder/'selection.json';result = json.loads(path.read_text())
        result['tied'] = True;self.save(str(path), result)
        with self.assertRaisesRegex(ValueError, 'untied four-program'):
            validate_whole_program_selection(folder, library, reference)
        result['tied'] = False;result['winner'] = 'C';self.save(str(path), result)
        with self.assertRaisesRegex(ValueError, 'measured noise threshold'):
            validate_whole_program_selection(folder, library, reference)

    def test_stale_model_gate_or_partial_cohort_is_rejected(self):
        folder, library, reference = self.fixture()
        path = self.base/'correctness-1/result.json';checked = json.loads(path.read_text())
        checked['execution']['artifact_id'] = 'other';self.save(str(path), checked)
        with self.assertRaisesRegex(ValueError, 'five fresh model processes'):
            validate_whole_program_selection(folder, library, reference)
        checked['execution']['artifact_id'] = 'id-1';checked['processes'].pop()
        self.save(str(path), checked)
        with self.assertRaisesRegex(ValueError, 'five fresh model processes'):
            validate_whole_program_selection(folder, library, reference)

    def test_substituted_measured_artifact_is_rejected(self):
        folder, library, reference = self.fixture()
        path = self.base/'measure-1.json';measured = json.loads(path.read_text())
        measured['library_path'] = reference;self.save(str(path), measured)
        with self.assertRaisesRegex(ValueError, 'checked artifact'):
            validate_whole_program_selection(folder, library, reference)


class FixedGridIntegrationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for name in ('control', 'candidate', 'main0', 'main1', 'body', 'body_main',
                     'attention', 'attention_main', 'host'):
            (self.base/name).write_bytes(name.encode())
        self.decision = dict(reference=str(self.base/'control'), library=str(self.base/'candidate'),
            gate=self.save('candidate_gate', dict(cases=[dict(outputs=[dict(output_sha256='same')])])),
            reference_gate=self.save('reference_gate', dict(cases=[dict(outputs=[dict(output_sha256='same')])])))
        proofs = []
        for flag, name in enumerate(('control', 'candidate')):
            proofs.append(dict(flag=flag, passed=True, words=2, reference=str(self.base/name),
                reference_sha256=self.digest(self.base/name), library=str(self.base/f'main{flag}'),
                library_sha256=self.digest(self.base/f'main{flag}')))
        self.main = dict(passed=True, proofs=proofs)
        self.units = []
        rows = []
        for suite in ('body', 'attention'):
            digest = self.digest(self.base/suite)
            self.units.append(self.save(suite+'_gate', dict(processes=[dict(binary_sha256=digest)])))
            rows.append(dict(suite=suite, passed=True, words=2, reference_binary=str(self.base/suite),
                reference_sha256=digest, main_binary=str(self.base/(suite+'_main')),
                main_sha256=self.digest(self.base/(suite+'_main'))))
        self.host = dict(passed=True, passes=[dict(returncode=0)]*5, gpu_executed=False,
            binary=str(self.base/'host'), binary_sha256=self.digest(self.base/'host'))
        self.save('integration', dict(main_proof=self.save('main', self.main),
            unit_proof=self.save('units', dict(passed=True, rows=rows)),
            host_guard=self.save('host_check', self.host),
            runtime_header_sha256=self.digest(ROOT/'include/tilemega/Codegen/tasks/Sm90F3Runtime.cuh'),
            model_bit_proof=self.save('model_bits', dict(passed=True,
                candidate_gate=self.decision['gate'], reference_gate=self.decision['reference_gate'], hashes=['same']))))

    def save(self, name, value):
        path = self.base/(name+'.json')
        path.write_text(json.dumps(value))
        return str(path)

    def check(self):
        validate_fixed_grid_integration(self.base, self.decision, self.units)

    def test_complete_integration(self):
        self.check()

    def test_substituted_candidate_is_rejected(self):
        (self.base/'candidate').write_bytes(b'rebuilt without qualification')
        with self.assertRaisesRegex(ValueError, 'stale artifact'):
            self.check()

    def test_missing_default_proof_is_rejected(self):
        self.main['proofs'] = self.main['proofs'][1:]
        self.save('main', self.main)
        with self.assertRaisesRegex(ValueError, 'instruction proof is incomplete'):
            self.check()

    def test_incomplete_host_gate_is_rejected(self):
        self.host['passes'].pop()
        self.save('host_check', self.host)
        with self.assertRaisesRegex(ValueError, 'host rejection gate'):
            self.check()

    def test_changed_unit_binary_is_rejected(self):
        (self.base/'body_main').write_bytes(b'untested body')
        with self.assertRaisesRegex(ValueError, 'fixture proof differs'):
            self.check()

    def test_changed_output_bits_are_rejected(self):
        self.save('candidate_gate', dict(cases=[dict(outputs=[dict(output_sha256='different')])]))
        with self.assertRaisesRegex(ValueError, 'changed model output'):
            self.check()


if __name__ == '__main__':
    unittest.main()
