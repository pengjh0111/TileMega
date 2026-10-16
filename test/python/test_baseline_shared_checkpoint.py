"""Shared checkpoint claims require live files and one consistent identity."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2] / 'python/tilemega/serving/baseline_common.py'
spec = importlib.util.spec_from_file_location('shared_baseline_common', SOURCE)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


class SharedCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = Path(self.temp.name)
        (self.model / 'config.json').write_text('{}')
        self.manifest = dict(algorithm='test', config_sha256=hashlib.sha256(b'{}').hexdigest(),
                             seed=0, aliases={}, tensors=[], state='live', total_bytes=4,
                             tensor_count=1, shards=[dict(file='weights', bytes=4)])
        identity = {key: self.manifest[key] for key in
                    ('algorithm', 'config_sha256', 'seed', 'aliases', 'tensors')}
        self.manifest['checkpoint_sha256'] = hashlib.sha256(json.dumps(
            identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        (self.model / 'weights').write_bytes(b'test')
        self.save()

    def save(self):
        (self.model / 'random_checkpoint.json').write_text(json.dumps(self.manifest))

    def test_official_loader_retains_shared_identity(self):
        result = common.shared_checkpoint_identity(self.model, 'auto')
        self.assertTrue(result['shared_checkpoint_identity'])
        self.assertEqual(result['checkpoint']['checkpoint_sha256'], self.manifest['checkpoint_sha256'])

    def test_dummy_cannot_claim_shared_weights(self):
        with self.assertRaises(ValueError):
            common.shared_checkpoint_identity(self.model, 'dummy')

    def test_closed_checkpoint_is_rejected(self):
        self.manifest['state'] = 'closed'
        self.save()
        with self.assertRaises(ValueError):
            common.shared_checkpoint_identity(self.model, 'auto')

    def test_stale_memfd_link_is_rejected(self):
        (self.model / 'weights').unlink()
        (self.model / 'weights').symlink_to('/proc/999999999/fd/3')
        with self.assertRaises(ValueError):
            common.shared_checkpoint_identity(self.model, 'auto')

    def test_changed_configuration_is_rejected(self):
        (self.model / 'config.json').write_text('{"changed": true}')
        with self.assertRaises(ValueError):
            common.shared_checkpoint_identity(self.model, 'auto')

    def test_changed_manifest_identity_is_rejected(self):
        self.manifest['seed'] = 1
        self.save()
        with self.assertRaises(ValueError):
            common.shared_checkpoint_identity(self.model, 'auto')

    def test_truncated_shard_is_rejected(self):
        (self.model / 'weights').write_bytes(b'x')
        with self.assertRaises(ValueError):
            common.shared_checkpoint_identity(self.model, 'auto')

    def test_existing_checkpoint_format_is_unchanged(self):
        (self.model / 'random_checkpoint.json').unlink()
        self.assertEqual(common.shared_checkpoint_identity(self.model, 'auto'), {})
        self.assertEqual(common.shared_checkpoint_identity(self.model, 'dummy'), {})


if __name__ == '__main__':
    unittest.main()
