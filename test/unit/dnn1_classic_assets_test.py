"""CPU-only checks for classic torchvision export and classification gates."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torchvision.models as torchvision_models
from tilemega.dnn import export, baselines


class ClassicAssetsTests(unittest.TestCase):
    def setUp(self):
        available = patch.object(torch.cuda, 'is_available', return_value=False)
        available.start()
        self.addCleanup(available.stop)

    def test_random_models_select_unweighted_torchvision_factories(self):
        for name in ('resnet34', 'vgg16', 'alexnet'):
            with self.subTest(model=name):
                fixture = torch.nn.Linear(3, 2)
                with patch.object(torchvision_models, name, return_value=fixture) as factory:
                    module, source = export.model(name, random_weights=True)
                factory.assert_called_once_with(weights=None)
                self.assertFalse(module.training)
                self.assertEqual(module.weight.dtype, torch.bfloat16)
                self.assertIn('torchvision/' + name + '/', source)
        self.assertTrue({'resnet18', 'mbv1', 'mbv2', 'bert', 'nafnet'} <= set(export.MODELS))

    def test_classification_top1_is_required_even_with_high_cosine(self):
        reference = [torch.tensor([[1., 1.0001]])]
        actual = [torch.tensor([[1.0001, 1.]])]
        for name in ('resnet34', 'vgg16', 'alexnet'):
            with self.subTest(model=name):
                metrics = baselines.numerical_metrics(actual, reference, name)
                self.assertGreater(metrics[0]['cosine_min_vs_fp32'], .9999)
                self.assertEqual(metrics[0]['top1_agreement_vs_fp32'], 0.)
                with self.assertRaises(baselines.BaselineFailure):
                    baselines.correctness_gate(name, metrics)

    def test_new_names_roundtrip_the_existing_export_and_input_protocol(self):
        module = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1),
                                     torch.nn.AdaptiveAvgPool2d(1),
                                     torch.nn.Flatten(), torch.nn.Linear(4, 1000)).eval().bfloat16()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('resnet34', 'vgg16', 'alexnet'):
                with self.subTest(model=name), patch.object(export, 'model', return_value=(module, 'CPU fixture')):
                    manifest = export.export(name, root, root / 'fixtures', random_weights=True)
                    self.assertTrue(manifest['timing_eligible'])
                    self.assertFalse(manifest['accuracy_eligible'])
                    program, loaded = baselines.load_export(root / name)
                    self.assertEqual(loaded['model'], name)
                    for batch in (1, 2, 4, 8, 16):
                        args, kwargs = baselines.make_inputs(program, batch, 20261007)
                        with torch.inference_mode():
                            actual = baselines.leaves(program.module()(*args, **kwargs))[0]
                            expected = module(*args, **kwargs)
                        self.assertEqual(actual.shape, (batch, 1000))
                        self.assertEqual(actual.dtype, torch.bfloat16)
                        self.assertTrue(torch.equal(actual, expected))


if __name__ == '__main__':
    unittest.main()
