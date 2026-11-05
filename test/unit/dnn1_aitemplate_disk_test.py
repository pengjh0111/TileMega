"""CPU-only storage observation fixtures; retained cache content is immutable."""
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import dnn1_aitemplate_collect as controller


class DiskTest(unittest.TestCase):
    def test_reassignment_admission_requires_current_empty_and_unreserved_card(self):
        clean = dict(owners={}, used_mib=0, free_mib=80000)
        with tempfile.TemporaryDirectory() as folder:
            epoch = Path(folder)
            with patch.object(controller, 'main_reservations', return_value=[]), \
                    patch.object(controller.base, 'held', return_value=False), \
                    patch.object(controller, 'resource_snapshot', return_value=clean):
                self.assertTrue(controller.available_card(epoch, 4))
                for dirty in (dict(clean, used_mib=4), dict(clean, owners={'123': 1}),
                              dict(clean, free_mib=100), dict(clean, used_mib=-1)):
                    with patch.object(controller, 'resource_snapshot', return_value=dirty):
                        self.assertFalse(controller.available_card(epoch, 4))
                with patch.object(controller, 'main_reservations', return_value=['reserved']):
                    self.assertFalse(controller.available_card(epoch, 4))
                with patch.object(controller.base, 'held', return_value=True):
                    self.assertFalse(controller.available_card(epoch, 4))

    def test_gpu_telemetry_timeout_skips_card_and_preserves_error(self):
        with tempfile.TemporaryDirectory() as folder:
            epoch = Path(folder)
            with patch.object(controller.device, 'gpu', side_effect=subprocess.TimeoutExpired('nvidia-smi', 15)):
                self.assertIsNone(controller.resource_snapshot(epoch, 4))
            error = json.loads((epoch / 'resource_errors.jsonl').read_text())
            self.assertEqual(error['gpu'], 4)
            self.assertIn('skip this card', error['disposition'])

    def test_cache_measurement_preserves_files_and_records_available_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            epoch = root / 'epoch'
            epoch.mkdir()
            cache = root / 'ait/engines'
            cache.mkdir(parents=True)
            library = cache / 'engine.so'
            library.write_bytes(b'keep-engine')
            with patch.object(controller, 'ROOT', root), patch.object(controller.runner, 'AIT_ROOT', root / 'ait'):
                value = controller.disk_snapshot(epoch, 'before-cell', 'resnet18', 1)
            self.assertEqual(library.read_bytes(), b'keep-engine')
            self.assertGreaterEqual(value['owned_caches'][str(cache)]['allocated_bytes'], 11)
            self.assertGreater(value['filesystem_free_bytes'], 0)
            self.assertFalse(value['deleted_files'])
            self.assertEqual(json.loads((epoch / 'disk_usage.jsonl').read_text()), value)

    def test_size_probe_timeout_records_error_without_losing_collected_cell(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'ait/engines').mkdir(parents=True)
            epoch = root / 'epoch'
            epoch.mkdir()
            with patch.object(controller, 'ROOT', root), patch.object(controller.runner, 'AIT_ROOT', root / 'ait'), \
                    patch.object(controller.subprocess, 'run', side_effect=subprocess.TimeoutExpired('du', 30)):
                value = controller.disk_snapshot(epoch, 'after-cell', 'resnet18', 1)
            self.assertIn('error', value['owned_caches'][str(root / 'ait/engines')])
            self.assertGreater(value['filesystem_free_bytes'], 0)


if __name__ == '__main__':
    unittest.main()
