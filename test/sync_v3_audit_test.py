#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import tempfile
import json
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


class RawCorrectnessTest(unittest.TestCase):
    def make_cell(self, root, failed=None):
        (root/'log').mkdir()
        for model in ('gqa2','mha4'):
            (root/'log'/f'{model}_p0_full.build.json').write_text(json.dumps(
                dict(exit_code=0,binary_sha256='one-binary')))
            for seq in (4,128):
                folder=root/'correctness'/f'{model}_s{seq}_p3'
                folder.mkdir(parents=True)
                for i in range(50):
                    (folder/f'r{i}.log').write_text('RESULT status='+('MISMATCH' if i==failed else 'PASS')+'\n')
                    (folder/f'r{i}.json').write_text(json.dumps(dict(exit_code=int(i==failed),round=i,
                        time_ns=i,binary_sha256='one-binary')))

    def test_single_failed_process_fails_fifty_run_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp);self.make_cell(raw,failed=17)
            with self.assertRaises(ValueError):verify.audit_raw.correctness(raw)

    def test_duplicate_timestamps_do_not_count_as_fresh_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp);self.make_cell(raw)
            path=raw/'correctness/gqa2_s4_p3/r49.json'
            meta=json.loads(path.read_text());meta['time_ns']=0;path.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):verify.audit_raw.correctness(raw)

    def test_raw_logs_are_required_despite_pass_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp);(raw/'summary.md').write_text('All 200/200 PASS')
            with self.assertRaises(ValueError):verify.audit_raw.correctness(raw)


if __name__=='__main__':
    unittest.main()
