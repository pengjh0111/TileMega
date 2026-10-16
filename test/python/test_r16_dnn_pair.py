"""Use the fixed noise criterion, including tails, for a scoped recommendation."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from analyze_r16_dnn_pair import comparison, matching_devices


class Pair(unittest.TestCase):
    def rows(self, center, tail):
        return [dict(median_ms=center+d, p90_ms=tail+d) for d in (0,.01,-.01)]

    def test_faster_median_and_tail_are_eligible(self):
        value = comparison(self.rows(6,6.1), self.rows(5.7,5.8))
        self.assertTrue(value['eligible_in_measured_grid'])

    def test_noise_does_not_promote_and_tail_regression_blocks(self):
        a = self.rows(6,6.1)
        self.assertFalse(comparison(a,self.rows(5.99,6.09))['eligible_in_measured_grid'])
        self.assertFalse(comparison(a,self.rows(5.7,6.4))['eligible_in_measured_grid'])

    def test_index_alone_cannot_establish_same_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'measurement.json').write_text('{}')
            with self.assertRaises(ValueError):
                matching_devices(root, [dict(measurement='measurement.json', gpu=1)])

    def test_runtime_uuid_and_epoch_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def row(name, uuid='abc', epoch='boot'):
                identity = dict(cuda_uuid=uuid, nvml_uuid='GPU-'+uuid,
                    gpu_index=1, cuda_pci_bus_id='0000:10:00.0', numa_node=0, boot_id=epoch)
                receipt = root/(name+'-device.json')
                receipt.write_text(json.dumps(identity))
                measurement = root/(name+'.json')
                measurement.write_text(json.dumps(dict(physical_device_identity=identity,
                    physical_device_identity_file=str(receipt),
                    physical_device_identity_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())))
                return dict(measurement=measurement.name, gpu=1)
            a, b = row('a'), row('b')
            self.assertEqual(matching_devices(root, [a,b])['uuid'], 'abc')
            with self.assertRaises(ValueError):
                matching_devices(root, [a,row('c', uuid='other')])
            with self.assertRaises(ValueError):
                matching_devices(root, [a,row('d', epoch='reboot')])
            (root/'b-device.json').write_text('{}')
            with self.assertRaises(ValueError):
                matching_devices(root, [a,b])


if __name__ == '__main__':
    unittest.main()
