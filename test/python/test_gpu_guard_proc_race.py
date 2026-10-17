# SPDX-License-Identifier: BSD-3-Clause
"""A process disappearing during discovery must not kill the GPU dispatcher."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'python'))
from tilemega.bench import gpu_guard


class ProcRaceTest(unittest.TestCase):
    def test_disappearing_leaf_preserves_live_descendants_and_session(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            entries = [root/str(pid) for pid in (10, 11, 12, 99)]
            for entry in entries:
                entry.mkdir()
            (root/'10/stat').write_text('10 (parent) S 1 10 10')
            (root/'11/stat').write_text('11 (child) S 10 10 10')
            (root/'12/stat').write_text('12 (grandchild) S 11 10 10')
            original = Path.read_text

            def read(path, *args, **kwargs):
                if path == root/'99/stat':
                    raise ProcessLookupError(3, 'process exited', str(path))
                return original(path, *args, **kwargs)

            with patch.object(Path, 'iterdir', return_value=iter(entries)):
                with patch.object(Path, 'read_text', read):
                    self.assertEqual(gpu_guard.descendants(10), {10, 11, 12})
            with patch.object(Path, 'iterdir', return_value=iter(entries)):
                with patch.object(Path, 'read_text', read):
                    self.assertEqual(gpu_guard.session_members(10), {10, 11, 12})

    def test_discovery_does_not_stat_or_glob_each_process(self):
        paths = [Path('/proc/10'), Path('/proc/11'), Path('/proc/self')]
        with patch.object(Path, 'iterdir', return_value=iter(paths)):
            with patch.object(Path, 'stat', side_effect=ProcessLookupError()):
                with patch.object(Path, 'glob', side_effect=ProcessLookupError()):
                    self.assertEqual(list(gpu_guard.proc_files('/proc', 'stat')),
                                     [Path('/proc/10/stat'), Path('/proc/11/stat')])


if __name__ == '__main__':
    unittest.main()
