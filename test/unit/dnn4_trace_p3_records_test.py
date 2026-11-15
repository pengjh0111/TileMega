"""Reject incomplete or reordered CTA records before boundary selection."""
import gzip
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from dnn4_bert_trace_p3_qualify import record_windows


class RecordsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.raw = bytearray(63 * 132 * 352)
        for stage in range(63):
            for cta in range(132):
                start = 1000 + stage * 100 + cta
                struct.pack_into('<4Q', self.raw, (stage * 132 + cta) * 352,
                                 start, start + 10, start + 20, start + 5)
        self.frames = [[dict(stage=s, stage_ns=151, work_window_ns=141, gate_tail_ns=10)
                        for s in range(63)] for _ in range(50)]
        self.save_frames()
        compressed = gzip.compress(self.raw, mtime=0)
        for sample in range(50):
            (self.folder / f'records_{sample:03d}.bin.gz').write_bytes(compressed)

    def tearDown(self):
        self.temp.cleanup()

    def save_frames(self):
        (self.folder / 'samples.json').write_text(json.dumps(self.frames))

    def reject_record(self, offset, value):
        struct.pack_into('<Q', self.raw, offset, value)
        (self.folder / 'records_000.bin.gz').write_bytes(gzip.compress(self.raw, mtime=0))
        with self.assertRaises(ValueError):
            record_windows(self.folder)

    def test_all_fifty_frames(self):
        self.assertEqual(record_windows(self.folder), 63 * 151)

    def test_missing_cta(self):
        self.reject_record(131 * 352, 0)

    def test_first_ready_outside_work(self):
        self.reject_record(24, 999)

    def test_cross_stage_overlap_on_one_cta(self):
        self.reject_record(16, 1200)

    def test_segment_data_in_boundary_trace(self):
        self.reject_record(32, 1)

    def test_recomputed_window_differs(self):
        self.frames[0][0]['stage_ns'] = 150
        self.save_frames()
        with self.assertRaises(ValueError):
            record_windows(self.folder)

    def test_missing_frame(self):
        self.frames.pop()
        self.save_frames()
        with self.assertRaises(ValueError):
            record_windows(self.folder)

    def test_record_size(self):
        (self.folder / 'records_000.bin.gz').write_bytes(gzip.compress(self.raw[:-1], mtime=0))
        with self.assertRaises(ValueError):
            record_windows(self.folder)


if __name__ == '__main__':
    unittest.main()
