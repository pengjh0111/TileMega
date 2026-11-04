"""Cache transport preserves numerical artifacts and refuses non-equivalent identities."""
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('transport', ROOT/'scripts/dnn1_tvm_cache_transport.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Transport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        library = self.root/'lib.so'; library.write_bytes(b'library')
        self.identity = dict(recipe=dict(schema=6), gpu_uuid='gpuA', compute_capability=[9, 0],
            target_configuration=dict(arch='sm_90a'), libraries={str(library): module.sha(library)},
            tvm_source_identity=dict(modified_sources={}, patches={}))
        self.source = self.root/'source'/module.key(self.identity)
        self.source.mkdir(parents=True)
        (self.source/'model.so').write_bytes(b'compiled-VM')
        (self.source/'cache.lock').touch()
        (self.source/'tune').mkdir(); (self.source/'tune/database.json').write_text('measured trials')
        self.receipt = dict(identity=self.identity, model_sha256=module.sha(self.source/'model.so'),
            diagnostics=dict(tuning=dict(trials=2000)))
        (self.source/'compiled.json').write_text(json.dumps(self.receipt))
        self.target = dict(self.identity, gpu_uuid='gpuB')
        self.parent = self.root/'destination'

    def test_exact_vm_and_database_preserved_with_explicit_two_device_provenance(self):
        before = (self.source/'compiled.json').read_bytes()
        result = module.transport(self.source, self.target, self.parent)
        destination = Path(result['destination'])
        self.assertEqual((destination/'model.so').read_bytes(), (self.source/'model.so').read_bytes())
        self.assertEqual((destination/'tune/database.json').read_bytes(), b'measured trials')
        self.assertEqual((self.source/'compiled.json').read_bytes(), before)
        self.assertEqual(result['tuning_gpu_uuid'], 'gpuA')
        self.assertEqual(result['execution_gpu_uuid'], 'gpuB')
        self.assertFalse(result['retuned'])
        self.assertIn('unvalidated', result['acceptance'])
        receipt = json.loads((destination/'compiled.json').read_text())
        self.assertEqual(receipt['identity'], self.target)
        self.assertEqual(receipt['diagnostics']['cache_identity'], self.target)

    def test_model_recipe_or_library_mismatch_rejected(self):
        for name, value in [('recipe', dict(schema=5)), ('compute_capability', [8, 0]),
                            ('new_flag', True), ('gpu_uuid', 'gpuA')]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                module.transport(self.source, dict(self.target, **{name:value}), self.parent)
        (self.source/'model.so').write_bytes(b'changed')
        with self.assertRaises(ValueError): module.transport(self.source, self.target, self.parent)

    def test_existing_partial_destination_and_active_source_refused(self):
        destination = self.parent/module.key(self.target); destination.mkdir(parents=True)
        with self.assertRaises(FileExistsError): module.transport(self.source, self.target, self.parent)
        destination.rmdir()
        with (self.source/'cache.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError): module.transport(self.source, self.target, self.parent)

    def test_atomic_publish_does_not_replace_racing_empty_directory(self):
        self.parent.mkdir()
        staging = self.parent/'staging'; staging.mkdir(); (staging/'payload').write_text('ours')
        destination = self.parent/'race'; destination.mkdir()
        with self.assertRaises(FileExistsError): module.rename_absent(staging, destination)
        self.assertFalse((destination/'payload').exists())
        self.assertTrue((staging/'payload').exists())


if __name__ == '__main__': unittest.main()
