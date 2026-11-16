"""Reject stale, incomplete, or mixed-card boundary cohorts without CUDA."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
import dnn4_boundary_cohort_qualify as module


class CohortQualificationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.library, self.control = self.base/'trace.so', self.base/'model.so'
        self.library.write_bytes(b'trace');self.control.write_bytes(b'control')
        plan = dict(native_forward=dict(stages=[{}, {}]))
        identity = dict(artifact_id='trace', trace=True,
                        defines={'TILEMEGA_TRACE_STAGE': '3'}, plan=plan)
        control = dict(artifact_id='control', trace=False, plan=copy.deepcopy(plan))
        mock = patch.object(module, 'verify', side_effect=lambda p:
                            identity if Path(p) == self.library else control)
        mock.start();self.addCleanup(mock.stop)
        timing = self.save('timing.json', dict(gpu_index=3, artifact_id='control', pid=10,
            measurement=dict(warmup=50, iterations=200), samples_ms=[1.0]*200, median_ms=1.0))
        self.smoke = self.save('smoke.json', dict(protocol='P-SMOKE', gpu=3, batch=8,
            candidate_ms=1.0, input_sha256='inputs', measurements=[dict(
                guard=dict(code=0), measurement=str(timing))]))
        processes, traces = [], []
        for i in range(3):
            output = self.base/f'child{i}'
            trace = output/'trace'
            self.save(f'child{i}/trace/result.json', dict(pid=20+i, gpu=3, artifact_id='trace',
                reference_sha256=module.sha(self.control), input_sha256='inputs',
                trace_level=3, warmup=20, samples=50, start_sm_clock_mhz=1980,
                end_sm_clock_mhz=1980))
            self.save(f'child{i}/trace/samples.json',
                [[dict(stage=s, stage_ns=500000) for s in range(2)] for _ in range(50)])
            processes.append(dict(pid=20+i, exit_code=0, out=str(output)))
            traces.append(str(trace))
        self.cohort = dict(status='passed', processes=processes, traces=traces,
                           guard=str(self.save('guard.json', dict(code=0))))
        self.path = self.save('cohort.json', self.cohort)

    def save(self, name, value):
        path = self.base/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def check(self):
        return module.qualify(self.smoke, self.path, self.library, self.control, 3)

    def test_three_fresh_children_on_reference_card(self):
        self.assertTrue(self.check()['passed'])

    def test_partial_cohort_is_rejected(self):
        self.cohort['processes'].pop()
        self.save('cohort.json', self.cohort)
        with self.assertRaisesRegex(ValueError, 'three successful'):
            self.check()

    def test_wrong_card_is_rejected(self):
        path = self.base/'child1/trace/result.json'
        value = module.read(path);value['gpu'] = 0
        self.save('child1/trace/result.json', value)
        with self.assertRaisesRegex(ValueError, 'artifact or card'):
            self.check()

    def test_rebuilt_reference_is_rejected(self):
        self.control.write_bytes(b'another build')
        with self.assertRaisesRegex(ValueError, 'artifact or card'):
            self.check()

    def test_duplicate_stage_is_rejected(self):
        samples = module.read(self.base/'child0/trace/samples.json')
        samples[0][1]['stage'] = 0
        self.save('child0/trace/samples.json', samples)
        with self.assertRaisesRegex(ValueError, 'invalid stage windows'):
            self.check()

    def test_reused_child_pid_is_rejected(self):
        value = module.read(self.base/'child1/trace/result.json');value['pid'] = 20
        self.save('child1/trace/result.json', value)
        self.cohort['processes'][1]['pid'] = 20
        self.save('cohort.json', self.cohort)
        with self.assertRaisesRegex(ValueError, 'reused a process'):
            self.check()

    def test_rejected_reservation_is_rejected(self):
        self.save('guard.json', dict(code=75))
        with self.assertRaisesRegex(ValueError, 'three successful'):
            self.check()


if __name__ == '__main__':
    unittest.main()
