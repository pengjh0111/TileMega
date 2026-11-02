"""Freeze the completed matrix while retaining historical group diagnostics."""
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
    def frozen_fixture(self, directory, rows=None):
        directory = Path(directory)
        if rows is None:
            rows = [dict(record_type='final_two_fresh_processes', tp=tp, batch=batch,
                         system=system, status='pass', gpu_indices='4,5' if tp == 2 else '1,3,4,5')
                    for tp in (2, 4) for batch in (1, 2, 4, 8, 16)
                    for system in ('vllm', 'sglang', 'torchcompile')]
        path = directory / 'frozen.csv'
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        reference = directory / 'reference.json'
        reference.write_text(json.dumps(dict(kind='mx7_frozen_baseline_reference',
            reused_across_gpu_groups=True, reference_csv=dict(path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))))
        return reference, path, rows

    def test_frozen_matrix_survives_later_group_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, _, rows = self.frozen_fixture(directory)
            later = dict(rows[-1], gpu_indices='1,2,3,5', status='N/A')
            historical = Path(directory) / 'historical.csv'
            with historical.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(later))
                writer.writeheader()
                writer.writerows([*rows, later])
            self.assertEqual(module.latest_baselines(historical)[4, 16, 'torchcompile']['status'], 'N/A')
            frozen = module.frozen_baselines(reference)
            self.assertEqual(len(frozen), 30)
            self.assertEqual(frozen[4, 16, 'torchcompile']['status'], 'pass')
            self.assertEqual(frozen[4, 16, 'torchcompile']['gpu_indices'], '1,3,4,5')

    def test_frozen_matrix_rejects_changed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, path, _ = self.frozen_fixture(directory)
            with path.open('a') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'evidence changed'):
                module.frozen_baselines(reference)

    def test_frozen_matrix_requires_every_passing_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, rows = self.frozen_fixture(directory)
            for incomplete in (rows[:-1], [*rows[:-1], dict(rows[-1], status='N/A')]):
                reference, _, _ = self.frozen_fixture(directory, incomplete)
                with self.assertRaisesRegex(ValueError, 'all 30 passing cells'):
                    module.frozen_baselines(reference)

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
