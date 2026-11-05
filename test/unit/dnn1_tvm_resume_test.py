"""CPU-only resume lineage and strict five-process publication regressions."""
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import dnn1_tvm_resume_collect as collector
import dnn1_tvm_resume_sample as sample


class SeedTest(unittest.TestCase):
    def test_database_copy_preserves_original_and_keeps_old_task_ids_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); prior = root / 'prior'; prior.mkdir()
            (prior / 'database_tuning_record.json').write_text('[0,"fixture"]\n')
            (prior / 'runner_results.jsonl').write_text('{"task_id":7}\n')
            identity = sample.prior_identity({'prior_database': str(prior)})
            destination = root / 'continued'
            sample.seed_database(identity['files'], destination)
            self.assertEqual(sample.prior_identity({'prior_database': str(prior)}), identity)
            self.assertEqual((destination / 'database_tuning_record.json').read_bytes(),
                             (prior / 'database_tuning_record.json').read_bytes())
            self.assertFalse((destination / 'runner_results.jsonl').exists())
            self.assertTrue((destination / 'prior_journals/runner_results.jsonl').exists())
            with self.assertRaises(FileExistsError):
                sample.seed_database(identity['files'], destination)


class CollectorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.attempt = self.root / 'attempt1'
        self.job = self.attempt / 'job'; self.job.mkdir(parents=True)
        self.cell = dict(model='nafnet', batch=2, gpu=5,
            correctness_policy='five-independent-gates-v2', prefix={'available': False},
            exported_program_sha256='export-fixture', source_checkpoint_sha256='checkpoint-fixture')
        self.session = dict(status='complete', model='nafnet', batch=2, gpu=5,
                            ended=60, logical_jobs=[])
        self.outer = []
        for index in range(5):
            role = 'measurement' if index < 3 else 'correctness'
            folder = self.job / str(index); folder.mkdir()
            path = folder / 'measurements.json'
            diag = dict(cache_identity={'source': 'fixed-source-fixture'}, model_sha256='fixed-vm',
                        compiled_library_sha256='fixed-vm')
            value = dict(system='tvm', precision='bf16', model='nafnet', batch=2, gpu_index=5,
                pid=100+index, process_role=role, correctness_passed=True,
                correctness=[{'psnr_min_vs_fp32_db': 50.}], versions={'fixture': '1'},
                configuration={'tvm_diag': diag}, measurement={'warmup': 50, 'iterations': 200},
                samples_ms=[float(index+1)]*200 if index < 3 else [],
                median_ms=float(index+1) if index < 3 else None,
                p90_ms=float(index+1) if index < 3 else None)
            collector.base.write(path, value)
            stamps = [dict(label='before-measurement', time=10*index+2, pids=[1000+index]),
                      dict(label='after-measurement', time=10*index+4, pids=[1000+index])]
            (folder / 'guard.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in stamps))
            self.outer.append(dict(time=10*index+3, owners={str(1000+index): 100}))
            process = dict(returncode=0, pid=100+index, folder=str(folder), measurement=str(path),
                receipt_sha256=collector.base.sha(path), status='pass',
                started=10*index+1, ended=10*index+5,
                process_before={'owners': {}, 'used_mib': 0},
                process_after={'owners': {}, 'used_mib': 0}, drain={'status': 'empty_observed'},
                clock_start={'sm_clock_mhz': 1980}, clock_end={'sm_clock_mhz': 1980})
            self.session['logical_jobs'].append(dict(arm={'arm': 'tvm-bf16', 'system': 'tvm',
                'precision': 'bf16'}, role=role, index=index, status='pass',
                attempts=[process], final=process))

    def collect(self):
        collector.base.write(self.job / 'cell-result.json', self.session)
        collector.base.write(self.attempt / 'guard_result.json', {'code': 0})
        (self.attempt / 'guard.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.outer))
        with patch.object(collector.publisher, 'publish', side_effect=lambda a,b: shutil.copyfile(a,b)), \
             patch.object(collector.auditor, 'audit', return_value={'cells': [{'sampled_windows_clean': True}]}):
            return collector.collect_cell(self.cell, self.attempt, self.root / 'collection')

    def mutate_receipt(self, index, callback, update_hash=True):
        process = self.session['logical_jobs'][index]['final']
        path = Path(process['measurement']); value = collector.base.read(path)
        callback(value); collector.base.write(path, value)
        if update_hash:
            process['receipt_sha256'] = collector.base.sha(path)

    def test_complete_clean_cell_aggregates_every_process(self):
        row = self.collect()
        self.assertEqual(row['status'], 'pass')
        self.assertEqual(float(row['median_ms']), 2.)
        self.assertEqual(row['correctness_passes'], '5/5')

    def test_missing_process_never_publishes_partial_cell(self):
        self.session['logical_jobs'].pop()
        with self.assertRaises(ValueError): self.collect()

    def test_mixed_source_identity_is_rejected(self):
        self.mutate_receipt(1, lambda r: r['configuration']['tvm_diag']['cache_identity'].update(source='other'))
        with self.assertRaisesRegex(ValueError, 'different source'): self.collect()

    def test_mixed_vm_is_rejected(self):
        self.mutate_receipt(1, lambda r: r['configuration']['tvm_diag'].update(model_sha256='other'))
        with self.assertRaisesRegex(ValueError, 'different source'): self.collect()

    def test_dirty_owner_inside_window_is_rejected(self):
        self.outer[0]['owners']['99999'] = 4
        with self.assertRaisesRegex(ValueError, 'dirty PID'): self.collect()

    def test_changed_receipt_is_rejected(self):
        self.mutate_receipt(0, lambda r: r.update(median_ms=99.), update_hash=False)
        with self.assertRaisesRegex(ValueError, 'dirty PID'): self.collect()

    def test_wrong_execution_gpu_is_rejected(self):
        self.mutate_receipt(0, lambda r: r.update(gpu_index=7))
        with self.assertRaisesRegex(ValueError, 'GPU identity'): self.collect()

    def test_transient_memory_before_initialization_is_rejected(self):
        self.session['logical_jobs'][0]['final']['process_before']['used_mib'] = 4
        with self.assertRaisesRegex(ValueError, 'dirty process/memory'): self.collect()

    def test_pass_flag_does_not_override_original_threshold(self):
        self.mutate_receipt(0, lambda r: r.update(correctness=[{'psnr_min_vs_fp32_db': 20.}]))
        with self.assertRaises(ValueError): self.collect()


if __name__ == '__main__':
    unittest.main()
