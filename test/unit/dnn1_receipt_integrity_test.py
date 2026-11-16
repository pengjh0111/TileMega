"""Official metadata may be appended only when original bytes are recoverable."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import dnn1_receipt_integrity as integrity


class IntegrityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'measurements.json'
        self.wrapper = self.root / 'wrapper.py'; self.wrapper.write_text('pinned fixture')
        self.identity_file = self.root / 'physical-device-identity.json'
        self.identity = dict(gpu_index=5, cuda_uuid='fixture-uuid', nvml_uuid='GPU-fixture-uuid',
            cuda_pci_bus_id='0000:87:00.0', numa_node=1,
            wrapper_sha256=integrity.sha(self.wrapper), outer_guard_holds_percard_lock=True,
            probe_exit_code=0, guard_policy_sha256=integrity.sha(integrity.base.POLICY))
        self.write(self.identity_file, self.identity)
        self.receipt = dict(gpu_index=5, median_ms=2.5, samples_ms=[1., 2., 3.],
            configuration={'tvm_diag': {'cache_identity': {'gpu_uuid': 'fixture-uuid'}}})
        self.write(self.path, self.receipt)
        self.process = dict(receipt=str(self.path), receipt_sha256=integrity.sha(self.path))
        self.annotate()

    def write(self, path, value):
        path.write_text(json.dumps(value, indent=2)+'\n')

    def annotate(self):
        self.receipt.update(physical_device_identity=self.identity,
            physical_device_identity_file=str(self.identity_file),
            physical_device_identity_sha256=integrity.sha(self.identity_file))
        self.write(self.path, self.receipt)

    def check(self):
        return integrity.verify(self.process, self.identity_file, self.wrapper, 5)

    def test_only_exact_official_annotation_is_accepted(self):
        proof = self.check()
        self.assertEqual(proof['recovered_original_sha256'], self.process['receipt_sha256'])
        self.assertNotEqual(proof['current_sha256'], proof['original_sha256'])

    def test_changed_timing_cannot_be_hidden_by_annotation(self):
        self.receipt['median_ms'] = 1.2; self.annotate()
        with self.assertRaisesRegex(ValueError, 'original byte SHA'): self.check()

    def test_numerically_equal_but_differently_serialized_original_is_rejected(self):
        original = {k:v for k,v in self.receipt.items() if k not in integrity.FIELDS}
        self.process['receipt_sha256'] = __import__('hashlib').sha256(json.dumps(original).encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, 'original byte SHA'): self.check()

    def test_extra_unofficial_field_is_rejected(self):
        self.receipt['unofficial'] = True; self.annotate()
        with self.assertRaisesRegex(ValueError, 'original byte SHA'): self.check()

    def test_changed_identity_bytes_are_rejected(self):
        self.identity_file.write_text(self.identity_file.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'contents/hash'): self.check()

    def test_other_wrapper_source_is_rejected(self):
        self.wrapper.write_text('different wrapper')
        with self.assertRaisesRegex(ValueError, 'wrapper SHA'): self.check()

    def test_other_guard_identity_path_is_rejected(self):
        other = self.root / 'other.json'; self.write(other, self.identity)
        self.receipt['physical_device_identity_file'] = str(other)
        self.write(self.path, self.receipt)
        with self.assertRaisesRegex(ValueError, 'outside'): self.check()

    def test_changed_vm_gpu_uuid_is_rejected(self):
        self.receipt['configuration']['tvm_diag']['cache_identity']['gpu_uuid'] = 'other'
        self.annotate()
        with self.assertRaisesRegex(ValueError, 'original byte SHA'): self.check()


if __name__ == '__main__':
    unittest.main()
