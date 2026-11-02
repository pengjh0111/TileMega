"""Unavailable baselines must not revive earlier measurements from another group."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2] / 'experiments/mx7/baseline_results.py'
spec = importlib.util.spec_from_file_location('mx7_baseline_results', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BaselineResultsTests(unittest.TestCase):
    def test_latest_unavailable_replaces_earlier_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'baselines.csv'
            rows = [dict(record_type='final_two_fresh_processes', tp=4, batch=4, system='sglang',
                         status=status) for status in ('pass', 'N/A')]
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(module.latest_baselines(path)[4, 4, 'sglang']['status'], 'N/A')

    def test_unavailable_group_is_bound_to_failed_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            guard = Path(directory) / 'guard.json'
            guard.write_text(json.dumps(dict(gpu_indices=['1', '2', '3', '5'], code=124)))
            ref = dict(path=str(guard), sha256=hashlib.sha256(guard.read_bytes()).hexdigest())
            row = dict(measurements_json='[]', errors_json=json.dumps([dict(guard=ref)]))
            module.check_group(row, [1, 2, 3, 5])
            with self.assertRaisesRegex(ValueError, 'different physical GPU groups'):
                module.check_group(row, [1, 3, 4, 5])
            guard.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'evidence changed'):
                module.check_group(row, [1, 2, 3, 5])


if __name__ == '__main__':
    unittest.main()
