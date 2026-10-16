"""Storage failure before launch must leave the pending attempt untouched."""
import errno
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tilemega.bench.scheduler import prepare_attempt

class DiskAdmission(unittest.TestCase):
    def test_no_space_does_not_invent_an_attempt(self):
        record = dict(status='pending', attempts=0)
        with patch.object(Path, 'mkdir', side_effect=OSError(errno.ENOSPC, 'full')):
            self.assertIsNone(prepare_attempt(Path('/out/gpu7'), 'job', record))
        self.assertEqual(record, dict(status='pending', attempts=0))

    def test_real_directory_allocates_exactly_one_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = dict(status='pending', attempts=0)
            folder = prepare_attempt(Path(tmp)/'gpu7', 'job', record)
            self.assertTrue(folder.is_dir())
            self.assertEqual(record['attempts'], 1)

    def test_permissions_are_not_misreported_as_no_space(self):
        record = dict(status='pending', attempts=0)
        with patch.object(Path, 'mkdir', side_effect=OSError(errno.EACCES, 'denied')):
            with self.assertRaises(OSError):
                prepare_attempt(Path('/out/gpu7'), 'job', record)
        self.assertEqual(record['attempts'], 0)

if __name__ == '__main__':
    unittest.main()
