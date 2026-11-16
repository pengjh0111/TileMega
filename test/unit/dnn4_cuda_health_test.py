"""CPU regression checks for failed-CUDA and remapped-device recovery gates."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import dnn4_bert_gpu_health as health

UUIDS = ['GPU-a', 'GPU-b', 'GPU-c', 'GPU-d']
EXPECTED = dict(devices={str(i): dict(uuid=uuid, pci=f'0000:{i+15:02x}:00.0')
                         for i, uuid in enumerate(UUIDS)})


class Recovery(unittest.TestCase):
    def test_reindexed_physical_card_is_not_the_original_inventory(self):
        text = '\n'.join(f'{i}, {UUIDS[j]}, 00000000:{j+15:02x}:00.0'
                         for i, j in enumerate((0, 1, 3)))
        self.assertFalse(health.matches(EXPECTED, health.inventory(text)))
        text = '\n'.join(f'{i}, {uuid}, 00000000:{i+15:02x}:00.0'
                         for i, uuid in enumerate(UUIDS))
        self.assertTrue(health.matches(EXPECTED, health.inventory(text)))

    def test_duplicate_identity_is_rejected(self):
        with self.assertRaises(ValueError):
            health.inventory('0, GPU-a, 0000:0f:00.0\n1, GPU-a, 0000:10:00.0')

    def test_wrapper_success_never_hides_driver_error(self):
        rows = [dict(mask=mask, exit_code=0, stderr='', stdout=json.dumps(dict(
                    mask=mask, cuInit=999, cuDeviceGetCount=3, count=-1, pid=pid)))
                for mask, pid in [('0', 101), ('GPU-a', 102)]]
        result = dict(diagnostic_only=True, rows=rows)
        with self.assertRaisesRegex(ValueError, 'has not recovered'):
            health.require_cuda(result, 'GPU-a', 0, dict(code=0))
        good = copy.deepcopy(result)
        for row in good['rows']:
            probe = json.loads(row['stdout'])
            probe.update(cuInit=0, cuDeviceGetCount=0, count=1)
            row['stdout'] = json.dumps(probe)
        health.require_cuda(good, 'GPU-a', 0, dict(code=0))
        good['rows'][1]['mask'] = 'GPU-d'
        with self.assertRaisesRegex(ValueError, 'physical device'):
            health.require_cuda(good, 'GPU-a', 0, dict(code=0))

    def test_one_reused_process_is_not_a_recovery_pair(self):
        rows = [dict(mask=mask, exit_code=0, stderr='', stdout=json.dumps(dict(
                    mask=mask, cuInit=0, cuDeviceGetCount=0, count=1, pid=101)))
                for mask in ('0', 'GPU-a')]
        with self.assertRaisesRegex(ValueError, 'two fresh'):
            health.require_cuda(dict(diagnostic_only=True, rows=rows),
                                'GPU-a', 0, dict(code=0))


if __name__ == '__main__':
    unittest.main()
