# SPDX-License-Identifier: BSD-3-Clause
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[2]/'scripts/diagnose_r16_request_reuse.py'
SPEC = importlib.util.spec_from_file_location('reuse_diagnostic', PATH)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class RequestReuseTest(unittest.TestCase):
    def test_first_difference_keeps_batch_attribution(self):
        self.assertEqual(diagnostic.differences([[1, 2], [3, 4]], [[1, 2], [3, 5]]), [None, 1])
        with self.assertRaises(ValueError):
            diagnostic.differences([[1, 2]], [[1]])

    def test_gpu_and_frozen_dependency_rejection_precedes_runtime_import(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory);candidate = folder/'candidate.json'
            candidate.write_text(json.dumps(dict(gpu=7)))
            spec = folder/'spec.json'
            spec.write_text(json.dumps(dict(diagnostic_sha256=diagnostic.sha(PATH),
                candidate=str(candidate), files={str(candidate): diagnostic.sha(candidate)})))
            env = dict(CUDA_VISIBLE_DEVICES='7', TILEMEGA_DEVICE_INDEX='7',
                TILEMEGA_GPU_LOCK='/tmp/tilemega_gpu7.lock', TILEMEGA_GPU_LOCK_HELD='1')
            with patch.dict(os.environ, env, clear=True):
                diagnostic.validate_spec(spec)
                for key, value in [('CUDA_VISIBLE_DEVICES', '6'), ('TILEMEGA_DEVICE_INDEX', '0'),
                                   ('TILEMEGA_GPU_LOCK_HELD', '0'), ('TILEMEGA_GPU_LOCK', '/tmp/other')]:
                    with self.subTest(key=key), patch.dict(os.environ, {key: value}):
                        with self.assertRaisesRegex(ValueError, 'guarded physical GPU'):
                            diagnostic.validate_spec(spec)
                candidate.write_text('{}')
                with self.assertRaisesRegex(ValueError, 'frozen dependency changed'):
                    diagnostic.validate_spec(spec)

    def test_functional_cards_require_explicit_matching_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);candidate=folder/'candidate.json';spec=folder/'spec.json'
            for index in (5, 6, 0):
                candidate.write_text(json.dumps(dict(gpu=index)))
                value=dict(diagnostic_sha256=diagnostic.sha(PATH),candidate=str(candidate),
                    files={str(candidate):diagnostic.sha(candidate)})
                env=dict(CUDA_VISIBLE_DEVICES=str(index),TILEMEGA_DEVICE_INDEX=str(index),
                    TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{index}.lock',TILEMEGA_GPU_LOCK_HELD='1')
                with self.subTest(index=index),patch.dict(os.environ,env,clear=True):
                    spec.write_text(json.dumps(value))
                    with self.assertRaises(ValueError):diagnostic.validate_spec(spec)
                    value['allowed_gpu_indices']=[5,6,7];spec.write_text(json.dumps(value))
                    if index in (5,6):diagnostic.validate_spec(spec)
                    else:
                        with self.assertRaises(ValueError):diagnostic.validate_spec(spec)


if __name__ == '__main__':
    unittest.main()
