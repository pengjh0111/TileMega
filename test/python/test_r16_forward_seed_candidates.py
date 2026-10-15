# SPDX-License-Identifier: BSD-3-Clause
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from run_r16_forward_seed_candidates import KINDS, completed_receipt, resource_gate, scheduler_queue_path, seed_shape, seed_tuple


class SeedGeometryTest(unittest.TestCase):
    def test_candidates_retain_their_explicit_geometry(self):
        wide, narrow = seed_shape(128), seed_shape(64)
        self.assertEqual(seed_tuple(wide), 'M64N128K64S2')
        self.assertEqual(seed_tuple(narrow), 'M64N64K64S2')
        self.assertEqual(wide['tile_n'], 128)
        self.assertEqual(narrow['split_k'], 1)
        for value in (0, 32, 65, 256):
            with self.assertRaisesRegex(ValueError, 'predeclared'):
                seed_shape(value)


class ActualSeedResourcesTest(unittest.TestCase):
    def setUp(self):
        self.target = dict(resources=dict(num_sms=132), sm_major=9, sm_minor=0)
        self.actual = dict(num_sms=132, major=9, minor=0, kernels=[
            dict(name=kind, registers=128, resident=1, threads=128,
                 max_threads=128, dynamic_shared=70000) for kind in KINDS])

    def test_actual_final_executor_resources_are_required(self):
        self.assertEqual(resource_gate(self.actual, self.target, 70000), 1)
        for field, value in [('resident', 0), ('registers', 0), ('max_threads', 64),
                             ('dynamic_shared', 49152), ('threads', 160)]:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.actual)
                changed['kernels'][2][field] = value
                with self.assertRaisesRegex(ValueError, 'declared placement'):
                    resource_gate(changed, self.target, 70000)

    def test_device_and_complete_kernel_set_are_checked(self):
        changed = copy.deepcopy(self.actual)
        changed['major'] = 8
        with self.assertRaisesRegex(ValueError, 'bound target'):
            resource_gate(changed, self.target, 70000)
        for change in ('missing', 'duplicate'):
            changed = copy.deepcopy(self.actual)
            if change == 'missing':
                changed['kernels'].pop()
            else:
                changed['kernels'][-1]['name'] = KINDS[0]
            with self.assertRaisesRegex(ValueError, 'kernel set'):
                resource_gate(changed, self.target, 70000)


class CompletedReceiptTest(unittest.TestCase):
    def test_published_queue_matches_the_scheduler_discovery_glob(self):
        self.assertEqual(scheduler_queue_path('/repo', 1, 'candidate'),
                         Path('/repo/runs/r16/queue/gpu1/queue_candidate.json'))
        self.assertTrue(scheduler_queue_path('/repo', 1, 'candidate').match('queue_*.json'))

    def test_compiler_schema_requires_reverified_head_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            file = source / 'fixture.cpp'
            file.write_text('int fixture = 1;\n')
            subprocess.run(['git', 'init', '-q', str(source)], check=True)
            subprocess.run(['git', '-C', str(source), 'add', 'fixture.cpp'], check=True)
            subprocess.run(['git', '-C', str(source), '-c', 'user.name=Fixture',
                '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture'], check=True)
            head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
            compiler = root / 'compiler'
            compiler.write_bytes(b'compiler fixture')
            receipt = root / 'receipt.json'
            value = dict(status='completed', compiler=str(compiler),
                compiler_sha256=hashlib.sha256(compiler.read_bytes()).hexdigest(),
                source_dir=str(source), snapshot_head=head,
                source_files={'fixture.cpp': hashlib.sha256(file.read_bytes()).hexdigest()})
            receipt.write_text(json.dumps(value))
            actual = completed_receipt(receipt)
            self.assertEqual(actual['cli'], str(compiler))
            self.assertEqual(actual['source_files_reverified'], 1)
            file.write_text('int fixture = 2;\n')
            with self.assertRaisesRegex(ValueError, 'captured source changed'):
                completed_receipt(receipt)


if __name__ == '__main__':
    unittest.main()
