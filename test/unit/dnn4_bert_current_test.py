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
from dnn4_bert_current import select_successors


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


if __name__ == '__main__':
    unittest.main()
