"""CPU checks for DNN1 protocol admission and explicit compiler controls."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(os.environ.get('TILEMEGA_DNN1_TEST_ROOT', Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / 'python/tilemega/serving'))
path = Path(os.environ.get('TILEMEGA_DNN1_BASELINES', ROOT / 'python/tilemega/dnn/baselines.py'))
spec = importlib.util.spec_from_file_location('dnn1_test_baselines', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ProtocolTest(unittest.TestCase):
    def test_classification_requires_both_cosine_and_top1(self):
        good = dict(cosine_vs_fp32=.9999, top1_agreement_vs_fp32=1.0)
        self.assertTrue(module.correctness_gate('resnet18', [good]))
        for bad in [dict(good, cosine_vs_fp32=.99989), dict(good, top1_agreement_vs_fp32=0.0)]:
            with self.assertRaises(module.BaselineFailure):
                module.correctness_gate('resnet18', [bad])

    def test_bert_and_restoration_have_their_own_limits(self):
        self.assertTrue(module.correctness_gate('bert', [dict(cosine_vs_fp32=.999)]))
        self.assertTrue(module.correctness_gate('nafnet', [dict(psnr_vs_fp32_db=40.)]))
        for name, row in [('bert', dict(cosine_vs_fp32=.99899)),
                          ('nafnet', dict(psnr_vs_fp32_db=39.99))]:
            with self.assertRaises(module.BaselineFailure):
                module.correctness_gate(name, [row])

    def test_empty_or_nonfinite_cosine_cannot_pass(self):
        for rows in [[], [dict(cosine_vs_fp32=float('nan'), top1_agreement_vs_fp32=1.)]]:
            with self.assertRaises(module.BaselineFailure):
                module.correctness_gate('mbv2', rows)

    def test_freezing_is_applied_before_compile_and_fullgraph_is_explicit(self):
        observed = []
        config = SimpleNamespace(freezing=False)
        def compile_model(source, *, mode, fullgraph):
            observed.append((config.freezing, mode, fullgraph))
            return lambda *a, **kw: 'compiled-result'
        fake = SimpleNamespace(device=lambda *a: 'cuda:0', cuda=SimpleNamespace(current_device=lambda: 0),
                               bfloat16='bf16', _inductor=SimpleNamespace(config=config), compile=compile_model)
        with patch.dict(sys.modules, torch=fake), \
             patch.object(module, 'portable_module', side_effect=lambda source, **kw: source), \
             patch.object(module, 'convert_inputs', side_effect=lambda args, kwargs, **kw: (args, kwargs)):
            for freezing, fullgraph in [(False, True), (True, True), (False, False)]:
                metadata = {}
                run = module.torch_runner(object(), (), {}, 'max-autotune', metadata,
                                          freezing=freezing, fullgraph=fullgraph)
                self.assertEqual(observed[-1], (freezing, 'max-autotune', fullgraph))
                self.assertEqual(metadata['freezing'], freezing)
                self.assertEqual(metadata['fullgraph'], fullgraph)
                self.assertEqual(run(), 'compiled-result')

    def test_error_receipt_keeps_original_compile_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            module._run_context.clear()
            module._run_context.update(out=Path(tmp), phase='compile', options={})
            with patch.object(module, 'main', side_effect=ValueError('original fullgraph error')):
                with self.assertRaisesRegex(ValueError, 'original fullgraph error'):
                    module.entrypoint()
            data = json.loads((Path(tmp) / 'failure.json').read_text())
            self.assertEqual(data['failure_kind'], 'compile')
            self.assertEqual(data['error'], 'original fullgraph error')
            self.assertIn('ValueError', data['traceback'])

    def test_cuda_selfcheck_failure_kind_survives_outer_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            module._run_context.clear()
            module._run_context.update(out=Path(tmp), phase='compile', options={})
            error = module.BaselineFailure('CUDA disabled', 'tvm_cuda_build_or_selfcheck')
            with patch.object(module, 'main', side_effect=error):
                with self.assertRaises(module.BaselineFailure):
                    module.entrypoint()
            data = json.loads((Path(tmp) / 'failure.json').read_text())
            self.assertEqual(data['failure_kind'], 'tvm_cuda_build_or_selfcheck')


if __name__ == '__main__':
    unittest.main()
