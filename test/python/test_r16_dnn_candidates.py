# SPDX-License-Identifier: BSD-3-Clause
import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('dnn_candidates', ROOT/'scripts/run_r16_dnn_candidates.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class DnnCandidateTest(unittest.TestCase):
    def fixture(self, root):
        source = root/'source'
        source.mkdir()
        cli = source/'tilemega'
        cli.write_bytes(b'fixed-cli')
        export = root/'runs/r16/dnn/export/mbv1'
        export.mkdir(parents=True)
        for name in ('bridge_b1.json', 'manifest.json', 'inputs_b1.safetensors', 'exported_program.pt2'):
            (export/name).write_bytes(b'shared-'+name.encode())
        (export/'checkpoint').mkdir()
        (export/'checkpoint/model.safetensors').write_bytes(b'fixed-random-checkpoint')
        driver.write(export/'manifest.json', {'checkpoint':{'sha256':driver.sha(export/'checkpoint/model.safetensors')}})
        receipt = root/'receipt.json'
        driver.write(receipt, dict(build_completed=True, snapshot_source_unchanged=True,
                     source_dir=str(source), snapshot_head='fixed-head', cli=str(cli), cli_sha256=driver.sha(cli)))
        target = root/'target.json'
        target.write_text('{}')
        python = root/'venv/bin/python'
        python.parent.mkdir(parents=True)
        python.symlink_to('/usr/bin/python3')
        return argparse.Namespace(root=root, receipt=receipt, out=root/'new-run', target=target,
                                  python=python, models=['mbv1', 'mbv2'], gpu=7, pid_mode='strict', class_cases_helper=None)

    def test_missing_model_does_not_block_other_models_and_venv_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            driver.prepare(args)
            metadata = driver.read(args.out/'preparation.json')
            self.assertEqual(len(metadata['configurations']), 2)
            self.assertEqual(metadata['rejected'][0]['model'], 'mbv2')
            queue = driver.read(args.out/'queue-preview.json')
            self.assertEqual(len(queue), 20)
            self.assertTrue(all(str(args.python) in row['command'] for row in queue))
            self.assertEqual(sum(row['timing'] for row in queue), 6)
            self.assertTrue(all(not row['gpu'] for row in queue if row['name'].endswith('-build')))
            with self.assertRaises(FileExistsError):
                driver.prepare(args)

    def test_changed_checkpoint_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            driver.prepare(args)
            candidate = Path(driver.read(args.out/'preparation.json')['configurations'][0])
            config = driver.verified_config(candidate)
            checkpoint = Path(config['export'])/'checkpoint/model.safetensors'
            checkpoint.write_bytes(b'another-random-state')
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                driver.verified_config(candidate)

    def test_guard_lease_is_required(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':'7', 'TILEMEGA_DEVICE_INDEX':'7'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'guard'):
                driver.require_lease({'gpu':7})
            os.environ['TILEMEGA_GPU_LOCK_HELD'] = '1'
            driver.require_lease({'gpu':7})
            os.environ['CUDA_VISIBLE_DEVICES'] = '6'
            with self.assertRaisesRegex(ValueError, 'GPU'):
                driver.require_lease({'gpu':7})

    def test_five_fresh_shared_checkpoint_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            driver.prepare(args)
            config_path = Path(driver.read(args.out/'preparation.json')['configurations'][0])
            config = driver.read(config_path)
            folder = Path(config['directory'])
            driver.write(folder/'plan.so.identity.json', {'artifact_id':'fixed-artifact'})
            for index in range(5):
                gate = folder/f'numeric-{index}.json'
                driver.write(gate, dict(passed=True, artifact_id='fixed-artifact',
                             input_tensors_sha256=config['export_files']['inputs_b1.safetensors'],
                             archive_sha256=config['export_files']['exported_program.pt2'],
                             cases=[{'mode':1}, {'mode':2}]))
                driver.write(folder/f'fresh-{index}.json', dict(child_pid=100+index, gate=str(gate)))
            driver.aggregate(config_path)
            self.assertEqual(driver.read(folder/'gate-5fresh.json')['fresh_passes'], 5)
            row = driver.read(folder/'fresh-4.json')
            row['child_pid'] = 100
            driver.write(folder/'fresh-4.json', row)
            with self.assertRaisesRegex(ValueError, 'distinct'):
                driver.aggregate(config_path)
            row['child_pid'] = 104
            driver.write(folder/'fresh-4.json', row)
            gate = driver.read(row['gate'])
            gate['archive_sha256'] = 'other-checkpoint'
            driver.write(row['gate'], gate)
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                driver.aggregate(config_path)


if __name__ == '__main__':
    unittest.main()
