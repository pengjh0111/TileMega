#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from fetch_assets_parallel import read_jobs

HERE = Path(__file__).resolve().parent


class ParallelAssetTests(unittest.TestCase):
    def test_all_existing_assets_have_independent_verified_identities(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); queue = root / 'queue'; queue.mkdir()
            jobs = []
            for i in range(7):
                path = root / f'asset-{i}'; payload = f'asset {i}'.encode(); path.write_bytes(payload)
                jobs.append(dict(name=f'asset_{i}', gpu=False, command=[sys.executable,
                    str(HERE / 'fetch_asset.py'), '--url', 'http://127.0.0.1:1/unreachable',
                    '--out', str(path), '--bytes', str(len(payload)), '--sha256',
                    hashlib.sha256(payload).hexdigest()]))
            (queue / 'queue_assets.json').write_text(json.dumps(jobs))
            run = subprocess.run([sys.executable, str(HERE / 'fetch_assets_parallel.py'),
                '--queue-dir', str(queue), '--out', str(root / 'results'), '--workers', '4'],
                capture_output=True, text=True, timeout=15)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads((root / 'results/result.json').read_text())
            self.assertTrue(result['passed'])
            self.assertEqual({r['name'] for r in result['assets']}, {j['name'] for j in jobs})
            self.assertTrue(all(r['identity']['attempts'] == 0 for r in result['assets']))

    def test_duplicate_writers_and_gpu_commands_are_rejected_before_start(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job = dict(name='one', command=[sys.executable, str(HERE / 'fetch_asset.py'),
                                           '--out', str(root / 'same')])
            path = root / 'queue_assets.json'; path.write_text(json.dumps([job, dict(job, name='two')]))
            with self.assertRaises(ValueError): read_jobs(root)
            path.write_text(json.dumps([dict(job, gpu=True)]))
            with self.assertRaises(ValueError): read_jobs(root)


if __name__ == '__main__':
    unittest.main()
