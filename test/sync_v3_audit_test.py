#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('r4_verify',ROOT/'docs/experiments/SYNC_V3/verify.py')
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class LitmusAuditTest(unittest.TestCase):
    def test_negative_control_passes_are_not_counted_as_mismatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp);(raw/'log').mkdir()
            (raw/'log/no_barrier_a0_g128_t4096.txt').write_text('RESULT status=pass nonce=0x1\n'*50)
            r=verify.litmus_counts(raw)[('no_barrier',0,128,4096)]
            self.assertEqual((r['runs'],r['pass_count'],r['mismatch']),(50,50,0))

    def test_launch_error_is_not_an_expected_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp);(raw/'log').mkdir()
            (raw/'log/no_fence_a0_g64_t1024.txt').write_text('RESULT status=launch_error\n')
            r=verify.litmus_counts(raw)[('no_fence',0,64,1024)]
            self.assertEqual((r['runs'],r['mismatch'],r['other']),(1,0,1))

    def test_missing_runs_are_not_inferred_from_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp);(raw/'log').mkdir()
            (raw/'status.txt').write_text('PASS\n')
            (raw/'litmus.tsv').write_text('thread0_fence\t50\tPASS\n')
            self.assertEqual(verify.litmus_counts(raw),{})


if __name__=='__main__':
    unittest.main()
