"""Random forward artifacts remain separate from dataset accuracy receipts."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tilemega.dnn import export
from tilemega.dnn.cli import model_export

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('dnn_baselines', ROOT / 'python/tilemega/dnn/baselines.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class DnnRandomTests(unittest.TestCase):
    def test_random_export_is_forward_but_not_accuracy_evidence(self):
        module = torch.nn.Sequential(torch.nn.Conv2d(3, 2, 1), torch.nn.AdaptiveAvgPool2d(1)).eval().bfloat16()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(export, 'model', return_value=(module, 'tiny CPU fixture')):
                manifest = export.export('resnet18', root, root / 'fixtures', random_weights=True)
            folder = root / 'resnet18'
            self.assertEqual(manifest['purpose'], 'forward')
            self.assertFalse(manifest['accuracy_eligible'])
            self.assertTrue(manifest['timing_eligible'])
            self.assertTrue((folder / 'checkpoint/model.safetensors').is_file())
            self.assertFalse((folder / 'core_aten.pt2').exists())
            self.assertEqual(model_export({'model': {'name': 'resnet18', 'export': str(folder),
                                                     'random_weights': True}}, root), folder)
            with self.assertRaises(ValueError):
                model_export({'model': {'name': 'resnet18', 'export': str(folder)}}, root)
            program, _ = baseline.load_export(folder)
            args1, kwargs1 = baseline.make_inputs(program, 1, 42)
            args2, kwargs2 = baseline.make_inputs(program, 1, 42)
            self.assertTrue(torch.equal(args1[0], args2[0]))
            expected = baseline.leaves(program.module()(*args1, **kwargs1))
            actual = baseline.leaves(baseline.portable_module(program.module(), dtype=torch.bfloat16)(
                *args2, **kwargs2))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))

    def test_promoting_export_promotes_factory_literals(self):
        class Factory(torch.nn.Module):
            def forward(self, x):
                return x + torch.ones(x.shape, dtype=torch.bfloat16, device='cpu')
        program = torch.export.export(Factory(), (torch.zeros(2, dtype=torch.bfloat16),))
        module = baseline.portable_module(program.module(), dtype=torch.float32)
        output = baseline.leaves(module(torch.zeros(2, dtype=torch.float32)))[0]
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.equal(output, torch.ones(2)))
        factories = [node for node in module.graph.nodes if 'dtype' in node.kwargs]
        self.assertTrue(factories)
        self.assertTrue(all(node.kwargs['dtype'] == torch.float32 for node in factories))

    def test_structure_only_is_rejected_for_baseline_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'exported_program.pt2'
            archive.write_bytes(b'not a forward archive')
            (root / 'manifest.json').write_text(json.dumps(dict(model='tiny',
                accuracy_eligible=False, timing_eligible=False,
                artifacts={'exported_program.pt2': baseline.sha(archive)})))
            with self.assertRaisesRegex(ValueError, 'structure-only'):
                baseline.load_export(root)


if __name__ == '__main__':
    unittest.main()
