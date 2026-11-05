"""CPU-only storage observation fixtures; retained cache content is immutable."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import dnn1_aitemplate_collect as controller


class DiskTest(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
