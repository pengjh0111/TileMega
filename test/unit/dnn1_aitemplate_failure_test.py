"""Terminal publication rejects incomplete or infrastructure-only AIT failures."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import dnn1_aitemplate_failure as failure


class FailureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cell = dict(model='mbv1', batch=2, gpu=1, graph_mode=True,
            source_sha256={'runner': 'sha'})
        self.identity = dict(model='mbv1', batch=2, recipe=dict(schema=7,
            precision='bf16', graph_mode=True), files={'runner': 'sha'})
        self.scheduler = {'ait-mbv1-b2': dict(status='failed', exit_code=1,
            attempts=1, started=1, run_s=5000)}
        self.value = dict(schema=1, system='aitemplate', status='N/A', final=True,
            repair_pending=False, model='mbv1', batch=2, failure_kind='compiler_build',
            reason='GCC11 internal compiler error during full engine build')
        self.refresh()

    def write(self, name, value, text=False):
        path = self.root / name
        path.write_text(value if text else json.dumps(value))
        return dict(path=str(path), sha256=failure.base.sha(path))

    def refresh(self):
        self.value.update(cell=self.write('cell.json', self.cell),
            attempt_identity=self.write('attempt.json', dict(identity=self.identity)),
            scheduler=self.write('scheduler.json', self.scheduler),
            outer_guard=self.write('guard.json', dict(code=self.scheduler['ait-mbv1-b2']['exit_code'])),
            command=self.write('command.json', dict(command=['sample'])),
            stderr=self.write('stderr.log', 'gcc: internal compiler error: Segmentation fault', True),
            original_failure=self.write('failure.json', dict(error='Build has failed.')))

    def test_actual_compiler_failure_preserves_its_frozen_identity(self):
        self.assertEqual(failure.validate(self.value)[1], self.identity)

    def test_repair_pending_and_incomplete_dispositions_are_rejected(self):
        for key, value in [('repair_pending', True), ('final', False)]:
            changed = copy.deepcopy(self.value)
            changed[key] = value
            with self.assertRaises(ValueError):
                failure.validate(changed)

    def test_admission_busy_and_cancelled_payloads_are_not_model_failures(self):
        for status, code in [('pending', 75), ('failed', 75), ('failed', 143), ('failed', -15)]:
            self.scheduler['ait-mbv1-b2'].update(status=status, exit_code=code)
            self.refresh()
            with self.assertRaisesRegex(ValueError, 'actual failed payload'):
                failure.validate(self.value)

    def test_cross_source_or_model_declarations_are_rejected(self):
        self.identity['files']['runner'] = 'different'
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'source binding'):
            failure.validate(self.value)
        self.identity['files']['runner'] = 'sha'
        self.identity['batch'] = 4
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'identity'):
            failure.validate(self.value)

    def test_missing_or_tampered_original_error_cannot_publish(self):
        Path(self.value['stderr']['path']).write_text('rewritten')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            failure.validate(self.value)
        self.refresh()
        self.value['original_failure'] = self.write('failure.json', {})
        with self.assertRaisesRegex(ValueError, 'original exception'):
            failure.validate(self.value)

    def test_build_timeout_requires_four_hours_and_no_completed_engine(self):
        self.value['failure_kind'] = 'build_timeout'
        self.scheduler['ait-mbv1-b2'].update(exit_code=124, run_s=14400)
        self.refresh()
        self.assertEqual(failure.validate(self.value)[2]['exit_code'], 124)
        self.scheduler['ait-mbv1-b2']['run_s'] = 100
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'four-hour'):
            failure.validate(self.value)
        self.scheduler['ait-mbv1-b2']['run_s'] = 14400
        self.refresh()
        (self.root / 'compiled.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'four-hour'):
            failure.validate(self.value)


if __name__ == '__main__':
    unittest.main()
