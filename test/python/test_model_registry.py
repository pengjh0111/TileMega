"""Model pins, shape metadata and fail-before-weights download validation."""
import copy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from tilemega.serving.download_models import fetch, main
from tilemega.serving.model_registry import (
    ALIASES, ROOT, dimensions, load_model, model_ids, verify_config,
)


class ModelRegistry(unittest.TestCase):
    def test_doctor_suggests_only_the_missing_registered_model(self):
        from tilemega.cli import Run
        with tempfile.TemporaryDirectory() as directory:
            run = Run.__new__(Run)
            run.model = Path(directory) / 'missing'
            run.cache = Path(directory) / 'cache'
            run.out = Path(directory)
            run.binary = 'unused'
            run.version = {}
            run.config = {'device': {'index': 0}}
            run.capture = lambda _: dict(name='fixture', arch_tag='sm_89',
                resources={'num_sms': 128}, driver_version=12000)
            for name, expected in [('qwen3_8b', 'qwen3_8b'), ('llama', 'llama3_2_1b'),
                                   ('custom_model', None)]:
                run.name = name
                with patch('tilemega.cli.subprocess.check_output', side_effect=['20,100,100,30', '']), \
                        contextlib.redirect_stdout(io.StringIO()):
                    result = run.doctor()
                if expected:
                    self.assertIn('--model ' + expected, result['download_command'])
                    self.assertNotIn('--model all', result['download_command'])
                else:
                    self.assertNotIn('download_command', result)

    def fixture(self, directory):
        root = Path(directory)
        config = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=6, num_key_value_heads=2, head_dim=8,
                      vocab_size=64, tie_word_embeddings=False)
        raw = (json.dumps(config)+'\n').encode()
        record = dict(id='tiny', repo='example/tiny', revision='a'*40,
                      config_sha256=hashlib.sha256(raw).hexdigest(),
                      tokenizer_family='llama3', tie_word_embeddings=False,
                      dimensions=dimensions(config))
        path = root/'configs/models/tiny.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(record))
        source = root/'downloaded/config.json'
        source.parent.mkdir()
        source.write_bytes(raw)
        return root, record, source

    def test_all_seven_models_and_legacy_aliases(self):
        expected = {'llama3_2_1b', 'llama3_2_3b', 'llama3_1_8b', 'qwen3_0_6b',
                    'qwen3_1_7b', 'qwen3_8b', 'qwen3_30b_a3b'}
        self.assertEqual(set(model_ids()), expected)
        for name in expected:
            record = load_model(name)
            self.assertEqual(record['id'], name)
            self.assertEqual(record['tokenizer_family'], 'llama3' if
                             name.startswith('llama') else 'qwen')
        for alias, name in ALIASES.items():
            self.assertEqual(load_model(alias), load_model(name))
        small = load_model('qwen3_0_6b')['dimensions']
        self.assertEqual(small['hidden_size'], 1024)
        self.assertEqual(small['head_dim']*small['num_attention_heads'], 2048)
        three = load_model('llama3_2_3b')['dimensions']
        self.assertEqual(three['qperkv'], 3)
        self.assertEqual((three['qperkv']+2)*three['head_dim'], 640)
        self.assertTrue(all(640 % tn == 0 for tn in (32, 64, 128)))
        for name in ('llama3_1_8b', 'qwen3_8b'):
            self.assertFalse(load_model(name)['tie_word_embeddings'])

    def test_registry_rejects_unpinned_revision_and_invalid_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root, record, _ = self.fixture(directory)
            record['revision'] = 'main'
            (root/'configs/models/tiny.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'fixed 40-character'):
                load_model('tiny', root)
            with self.assertRaisesRegex(ValueError, 'invalid model id'):
                load_model('../tiny', root)

    def test_config_hash_and_semantics_are_both_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            _, record, source = self.fixture(directory)
            self.assertEqual(verify_config(record, source)['head_dim'], 8)
            source.write_bytes(source.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                verify_config(record, source)
            record['config_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
            wrong = copy.deepcopy(record)
            wrong['dimensions']['hidden_size'] = 64
            with self.assertRaisesRegex(ValueError, 'dimensions differ'):
                verify_config(wrong, source)
            wrong = copy.deepcopy(record)
            wrong['tie_word_embeddings'] = True
            with self.assertRaisesRegex(ValueError, 'tie_word_embeddings differs'):
                verify_config(wrong, source)

    def test_download_keeps_pin_and_records_verified_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root, record, source = self.fixture(directory)
            destination = root/'checkpoint'
            hub = ModuleType('huggingface_hub')
            hub.hf_hub_download = Mock(return_value=str(source))
            def snapshot(**kwargs):
                self.assertEqual(kwargs['revision'], record['revision'])
                (destination/'config.json').write_bytes(source.read_bytes())
                (destination/'model.safetensors').write_bytes(b'weight fixture')
            hub.snapshot_download = Mock(side_effect=snapshot)
            with patch.dict('sys.modules', {'huggingface_hub': hub}):
                result = fetch('tiny', destination, root)
            self.assertEqual(hub.hf_hub_download.call_args.kwargs['revision'], record['revision'])
            self.assertEqual(result['config_sha256'], record['config_sha256'])
            self.assertEqual(result['config_differences'], {})
            self.assertEqual(result['dimensions']['qperkv'], 3)
            self.assertEqual(json.loads((destination/'tilemega_source_manifest.json').read_text()), result)

    def test_bad_config_rejects_before_large_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, source = self.fixture(directory)
            source.write_text('{}')
            hub = ModuleType('huggingface_hub')
            hub.hf_hub_download = Mock(return_value=str(source))
            hub.snapshot_download = Mock()
            with patch.dict('sys.modules', {'huggingface_hub': hub}):
                with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                    fetch('tiny', root/'checkpoint', root)
            hub.snapshot_download.assert_not_called()

    def test_config_only_fetches_just_the_pinned_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root, record, source = self.fixture(directory)
            destination = root/'checkpoint'
            hub = ModuleType('huggingface_hub')
            hub.hf_hub_download = Mock(return_value=str(source))
            hub.snapshot_download = Mock(side_effect=AssertionError('weights must not be requested'))
            with patch.dict('sys.modules', {'huggingface_hub': hub}):
                result = fetch('tiny', destination, root, config_only=True)
            hub.hf_hub_download.assert_called_once_with(
                repo_id=record['repo'], revision=record['revision'],
                filename='config.json', local_dir=destination)
            hub.snapshot_download.assert_not_called()
            self.assertTrue(result['structure_only'])
            self.assertEqual(result['safetensors_sha256'], {})
            self.assertEqual(result['dimensions'], record['dimensions'])
            self.assertFalse((destination/'tilemega_source_manifest.json').exists())

    def test_config_only_reuses_local_config_without_hub_or_weight_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, source = self.fixture(directory)
            destination = source.parent
            manifest = destination/'tilemega_source_manifest.json'
            manifest.write_text('existing weight provenance\n')
            with patch.dict('sys.modules', {'huggingface_hub': None}), \
                    patch('tilemega.serving.download_models.sha256',
                          side_effect=AssertionError('do not hash weight shards')):
                result = fetch('tiny', destination, root, config_only=True)
            self.assertEqual(manifest.read_text(), 'existing weight provenance\n')
            self.assertEqual(json.loads((destination/'tilemega_config_manifest.json').read_text()), result)

    def test_config_only_rejects_mismatched_local_config_without_replacing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, source = self.fixture(directory)
            source.write_text('{}')
            with patch.dict('sys.modules', {'huggingface_hub': None}):
                with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                    fetch('tiny', source.parent, root, config_only=True)
            self.assertEqual(source.read_text(), '{}')
            self.assertFalse((source.parent/'tilemega_config_manifest.json').exists())

    def test_config_only_cli_selects_structure_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, source = self.fixture(directory)
            destination = root/'models/tiny'
            destination.mkdir(parents=True)
            (destination/'config.json').write_bytes(source.read_bytes())
            report = root/'report.json'
            with patch.dict('sys.modules', {'huggingface_hub': None}), patch('builtins.print'):
                main(['--model', 'tiny', '--model-root', str(root/'models'),
                      '--repo-root', str(root), '--report', str(report), '--config-only'])
            self.assertTrue(json.loads(report.read_text())['tiny']['structure_only'])

    def test_nonintegral_query_groups_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'QPerKV'):
            dimensions(dict(hidden_size=64, num_attention_heads=5, num_key_value_heads=2))


if __name__ == '__main__':
    unittest.main()
