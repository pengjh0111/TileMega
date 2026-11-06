"""CPU-only source isolation checks for the new preflight entry."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import dnn1_settling_background_python as entry


class EntryTest(unittest.TestCase):
    def test_render_preserves_shared_files_and_exact_cpu_validated_source(self):
        before = {name: entry.sha(entry.ROOT / name) for name in entry.SOURCES}
        with tempfile.TemporaryDirectory() as folder:
            source, manifest = entry.render_guard(folder)
            self.assertEqual(entry.sha(source), entry.PATCHED_SHA)
            self.assertEqual(json.loads(manifest.read_text())['dependencies'], before)
            with self.assertRaises(FileExistsError):
                entry.render_guard(folder)
        self.assertEqual(before, {name: entry.sha(entry.ROOT / name) for name in entry.SOURCES})

    def test_changed_dependency_fails_before_rendering_or_launch(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(entry.SOURCES,
                {'scripts/dnn1_preflight_settle.py': 'invalid-sha'}):
            with self.assertRaisesRegex(ValueError, 'dependency changed'):
                entry.render_guard(folder)
            self.assertFalse((Path(folder) / 'preflight_source').exists())


if __name__ == '__main__':
    unittest.main()
