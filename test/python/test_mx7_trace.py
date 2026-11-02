"""Trace attribution uses release differences and compares durations across ranks."""
import csv
import hashlib
import json
import shutil
from pathlib import Path
import tempfile
import unittest

from tilemega.tp.trace import compress_trace, read_trace, summarize
from tilemega.tp.analyze_trace import analyze_capture
from tilemega.tp import trace


class TraceTests(unittest.TestCase):
    def capture(self, root):
        root.mkdir()
        shutil.copy2(trace.__file__, root / 'collector.py')
        records, ranks = [], []
        for rank in range(4):
            path = root / f'rank{rank}.csv'
            with path.open('w', newline='') as stream:
                writer = csv.writer(stream, lineterminator='\n')
                writer.writerow(('slot', 'stage', 'cta', 'start_ns', 'work_end_ns', 'release_ns'))
                for slot in (64, 65):
                    for stage in range(6):
                        for cta in range(2):
                            start = 100000 * rank + slot * 1000 + stage * (20 + rank) + cta
                            writer.writerow((slot, stage, cta, start, start + 5, start + 10))
            record = compress_trace(path)
            records.append(record)
            ranks.append(read_trace(Path(record['path']), layers=1, grid=2))
        data = dict(kind='mx7_tp_stage_capture', tp=4, batch=1, gpu_indices=[1,3,4,5],
                    trace_slots=66, decode_samples=2, raw_traces=records, timing_eligible=False,
                    collector={'sha256': hashlib.sha256((root / 'collector.py').read_bytes()).hexdigest()})
        capture = root / 'capture.json'
        capture.write_text(json.dumps(data))
        return capture, summarize(ranks, layers=1)

    def test_separate_cpu_analysis_preserves_stage_values_and_raw_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture, expected = self.capture(root / 'capture')
            guard = root / 'guard.json'
            guard.write_text(json.dumps(dict(code=0, gpu_indices=['1','3','4','5'])))
            report = analyze_capture(capture, guard, root / 'analysis/trace.json', layers=1, grid=2)
            self.assertEqual(report['stages'], expected)
            self.assertEqual(report['raw_traces'], json.loads(capture.read_text())['raw_traces'])
            self.assertFalse(report['timing_eligible'])

    def test_cpu_analysis_rejects_an_invalidated_gpu_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture, _ = self.capture(root / 'capture')
            guard = root / 'guard.json'
            guard.write_text(json.dumps(dict(code=75, gpu_indices=['1','3','4','5'])))
            output = root / 'analysis/trace.json'
            with self.assertRaisesRegex(ValueError, 'did not pass'):
                analyze_capture(capture, guard, output, layers=1, grid=2)
            self.assertFalse(output.exists())

    def test_release_attribution_and_lossless_raw_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            ranks = []
            for rank in range(2):
                path = Path(directory) / f'rank{rank}.csv'
                with path.open('w', newline='') as stream:
                    writer = csv.writer(stream, lineterminator='\n')
                    writer.writerow(('slot', 'stage', 'cta', 'start_ns', 'work_end_ns', 'release_ns'))
                    for slot in (64, 65):
                        base = 100000 * rank + slot * 1000
                        for stage in range(6):
                            for cta in range(2):
                                start = base + stage * (20 + rank * 2) + cta
                                writer.writerow((slot, stage, cta, start, start + 5, start + 10 + rank))
                original = hashlib.sha256(path.read_bytes()).hexdigest()
                archived = compress_trace(path)
                self.assertFalse(path.exists())
                self.assertEqual(archived['uncompressed_sha256'], original)
                ranks.append(read_trace(Path(archived['path']), layers=1, grid=2))
            result = summarize(ranks, layers=1)
            self.assertEqual(result[0]['rank_median_ns'], [11, 12])
            self.assertEqual(result[1]['rank_median_ns'], [20, 22])
            self.assertEqual(result[2]['stage'], 'o')
            self.assertEqual(result[2]['max_rank_difference_ns'], 2)
            self.assertEqual(result[-1]['stage'], 'head')


if __name__ == '__main__':
    unittest.main()
