"""Check boundary qualification rejects incomplete or mixed process evidence."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import dnn4_boundary_qualify as qualification


class BoundaryQualificationTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.library, self.control = self.root/'trace.so', self.root/'model.so'
        self.library.write_bytes(b'trace');self.control.write_bytes(b'nontrace')
        plan = dict(native_forward=dict(stages=[{}, {}]))
        self.identity = dict(defines={'TILEMEGA_TRACE_STAGE': '3'}, trace=True,
                             artifact_id='trace', plan=plan)
        self.control_identity = dict(trace=False, artifact_id='control', plan=copy.deepcopy(plan))
        context = patch.object(qualification, 'verify', side_effect=lambda path:
            self.identity if Path(path) == self.library else self.control_identity)
        context.start();self.addCleanup(context.stop)
        timing = self.root/'timing.json'
        self.save(timing, dict(gpu_index=0, artifact_id='control', pid=10,
            measurement=dict(warmup=50, iterations=200), samples_ms=[1]*200, median_ms=1))
        self.nontrace = self.root/'smoke.json'
        self.save(self.nontrace, dict(protocol='P-SMOKE', gpu=0, batch=2,
            candidate_ms=1, input_sha256='input',
            measurements=[dict(guard=dict(code=0), measurement=str(timing))]))
        self.traces = []
        for index, duration in enumerate([1e6, 1.02e6, 2e6]):
            output = self.root/f'process{index}'
            attempt = output/'attempt'
            self.save(output/'scheduler/state.json', {'trace': dict(status='done', exit_code=0,
                       out=str(attempt))})
            self.save(attempt/'guard_result.json', dict(code=0))
            self.save(attempt/'trace/result.json', dict(pid=20+index, gpu=0, trace_level=3,
                samples=50, warmup=20, artifact_id='trace', input_sha256='input',
                reference_sha256=qualification.sha(self.control),
                start_sm_clock_mhz=1980, end_sm_clock_mhz=1980))
            self.save(attempt/'trace/samples.json', [[dict(stage=s, stage_ns=duration/2)
                for s in range(2)] for _ in range(50)])
            self.traces.append(output)

    def save(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def run_qualification(self):
        return qualification.qualify(self.nontrace, self.traces, self.library, self.control)

    def mutate_trace(self, **values):
        path = self.traces[2]/'attempt/trace/result.json'
        self.save(path, dict(qualification.read(path), **values))

    def test_process_median_ignores_one_slow_process_without_dropping_it(self):
        result = self.run_qualification()
        self.assertEqual(result['trace_ns'], 1.02e6)
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['processes']), 3)
        self.assertAlmostEqual(result['relative_difference'], .02)

    def test_reused_process_is_rejected(self):
        self.mutate_trace(pid=20)
        with self.assertRaisesRegex(ValueError, 'reused a process'):
            self.run_qualification()

    def test_other_card_or_reference_output_cannot_qualify(self):
        for change in [dict(gpu=3), dict(reference_sha256='other output reference')]:
            with self.subTest(change=change):
                self.mutate_trace(**dict(dict(gpu=0,
                    reference_sha256=qualification.sha(self.control)), **change))
                with self.assertRaisesRegex(ValueError, 'guarded reference'):
                    self.run_qualification()

    def test_missing_stage_is_rejected(self):
        path = self.traces[2]/'attempt/trace/samples.json'
        samples = qualification.read(path);samples[0].pop()
        self.save(path, samples)
        with self.assertRaisesRegex(ValueError, 'complete stage windows'):
            self.run_qualification()

    def test_overhead_failure_is_recorded_without_changing_threshold(self):
        path = self.nontrace
        smoke = qualification.read(path)
        smoke['candidate_ms'] = .9
        self.save(path, smoke)
        timing = Path(smoke['measurements'][0]['measurement'])
        self.save(timing, dict(qualification.read(timing), median_ms=.9))
        result = self.run_qualification()
        self.assertFalse(result['passed'])
        self.assertGreater(result['relative_difference'], .05)


if __name__ == '__main__':
    unittest.main()
