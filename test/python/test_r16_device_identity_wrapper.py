"""Physical identity metadata must preserve timings and active device limits."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / 'scripts/run_r16_with_device_identity.py'
spec = importlib.util.spec_from_file_location('identity_wrapper', SOURCE)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


class DeviceIdentityWrapper(unittest.TestCase):
    def test_scope_is_checked_before_any_device_access(self):
        with self.assertRaises(ValueError):
            wrapper.require_guard(dict(allowed_gpu_indices=[7]), 6)

    def test_latest_external_scope_cannot_be_bypassed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'scope.json'
            path.write_text(json.dumps(dict(allowed_gpu_indices=[7])))
            with self.assertRaises(ValueError):
                wrapper.require_guard(dict(allowed_gpu_indices=list(range(8)),
                                           gpu_allowlist_file=str(path)), 0)

    def test_guard_variables_are_mandatory(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            wrapper.require_guard(dict(allowed_gpu_indices=[7]), 7)

    def test_append_only_identity_preserves_raw_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            receipt = out/'device.json'
            receipt.write_text(json.dumps(dict(cuda_uuid='abc', nvml_uuid='GPU-abc')))
            target = out/'measurements/B1/measurements.json'
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps(dict(e2e_seconds=1.25, generation_runs=[dict(wall_seconds=1.25)])))
            self.assertEqual(len(wrapper.annotate(out, receipt)), 1)
            value = json.loads(target.read_text())
            self.assertEqual(value['e2e_seconds'], 1.25)
            self.assertEqual(value['generation_runs'], [dict(wall_seconds=1.25)])
            self.assertEqual(value['physical_device_identity_sha256'], wrapper.sha(receipt))
            wrapper.annotate(out, receipt)
            receipt.write_text(json.dumps(dict(cuda_uuid='other')))
            with self.assertRaises(ValueError):
                wrapper.annotate(out, receipt)

    def test_successful_timing_requires_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            receipt = out/'device.json'
            receipt.write_text('{}')
            with self.assertRaises(ValueError):
                wrapper.annotate(out, receipt)
            self.assertEqual(wrapper.annotate(out, receipt, allow_empty=True), [])


if __name__ == '__main__':
    unittest.main()
