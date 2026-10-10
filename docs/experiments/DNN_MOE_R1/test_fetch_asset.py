#!/usr/bin/env python3
"""Completed checkpoint files must be verified without HTTP range requests."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('fetch_asset.py')


class ExistingAssetTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.asset = Path(self.folder.name) / 'config.json'
        self.payload = b'{"hidden_size":2048}\n'
        self.asset.write_bytes(self.payload)
        self.sha = hashlib.sha256(self.payload).hexdigest()
        self.blob = hashlib.sha1(b'blob ' + str(len(self.payload)).encode() +
                                b'\0' + self.payload).hexdigest()

    def run_asset(self, *extra):
        return subprocess.run([sys.executable, str(SCRIPT), '--url',
                               'http://127.0.0.1:1/unreachable', '--out', str(self.asset),
                               '--bytes', str(len(self.payload)), *extra],
                              capture_output=True, text=True, timeout=10)

    def test_complete_lfs_file_skips_network(self):
        result = self.run_asset('--sha256', self.sha)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['attempts'], 0)

    def test_complete_git_file_skips_network_and_checks_blob_header(self):
        result = self.run_asset('--git-blob', self.blob)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['sha256'], self.sha)
        self.assertNotEqual(hashlib.sha1(self.payload).hexdigest(), self.blob)

    def test_wrong_complete_file_is_retained_without_a_success_record(self):
        result = self.run_asset('--git-blob', '0' * 40)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.asset.read_bytes(), self.payload)
        self.assertFalse(Path(str(self.asset) + '.download.json').exists())

    def test_reuse_rejects_mutation_and_stricter_identity(self):
        self.assertEqual(self.run_asset('--sha256', self.sha).returncode, 0)
        self.assertEqual(self.run_asset('--git-blob', self.blob).returncode, 0)
        self.assertNotEqual(self.run_asset('--git-blob', '0' * 40).returncode, 0)
        self.asset.write_bytes(self.payload.replace(b'2048', b'2049'))
        self.assertNotEqual(self.run_asset('--sha256', self.sha).returncode, 0)


if __name__ == '__main__':
    unittest.main()
