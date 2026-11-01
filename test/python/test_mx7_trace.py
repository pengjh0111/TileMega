"""Trace attribution uses release differences and compares durations across ranks."""
import csv
import hashlib
from pathlib import Path
import tempfile
import unittest

from tilemega.tp.trace import compress_trace, read_trace, summarize


class TraceTests(unittest.TestCase):
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
