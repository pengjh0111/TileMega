#!/usr/bin/env python3
"""CPU-only checks for the remapped seven-card baseline device contract."""
import unittest
from unittest.mock import patch

import dnn1_device as device
import dnn1_idle_floor_python as floor


class ClassicDevicesTest(unittest.TestCase):
    def test_seven_cards_and_retired_identity(self):
        config = device.device_table()
        self.assertEqual(config['allowed_gpu_indices'], list(range(7)))
        self.assertEqual(config['devices']['6']['pci'], '0000:b7:00.0')
        self.assertNotIn(config['retired_gpu_uuid'],
                         [d['uuid'] for d in config['devices'].values()])

    @patch('dnn1_device.subprocess.check_output')
    def test_current_six_is_the_previous_healthy_seven(self, query):
        expected = device.device_table()['devices']['6']
        query.side_effect = [f"{expected['uuid']}, 00000000:B7:00.0, 0, 81079, 81559, 1980, 1593\n", '']
        row = device.gpu(6)
        self.assertEqual(row['gpu_uuid'], expected['uuid'])
        self.assertEqual(row['pci_bus_id'], expected['pci'])
        self.assertFalse(row['power_monitoring'])
        self.assertTrue(device.idle(row, {}, 8192))
        self.assertTrue(floor.idle(row, {}, 8192))
        for call in query.call_args_list:
            self.assertIn('6', call.args[0])
            self.assertNotIn('power', ' '.join(call.args[0]))

    @patch('dnn1_device.subprocess.check_output')
    def test_absent_seven_is_rejected_before_query(self, query):
        with self.assertRaises(ValueError):
            device.gpu(7)
        query.assert_not_called()

    @patch('dnn1_device.subprocess.check_output')
    def test_ordinal_reassignment_cannot_silently_select_another_card(self, query):
        query.return_value = 'GPU-wrong, 00000000:B7:00.0, 0, 81079, 81559, 1980, 1593\n'
        with self.assertRaisesRegex(ValueError, 'ordinal/UUID/PCI'):
            device.gpu(6)
        self.assertEqual(query.call_count, 1)

    def test_only_process_and_memory_occupancy_admit_work(self):
        row = dict(gpu_index=6, owners={}, used_mib=0, free_mib=81079)
        self.assertTrue(device.idle(row, {}, 8192))
        self.assertFalse(device.idle(dict(row, owners={123: 1}), {}, 8192))
        self.assertFalse(floor.idle(dict(row, owners={123: 1}), {}, 8192))
        self.assertFalse(device.idle(dict(row, used_mib=1), {}, 8192))
        self.assertTrue(floor.idle(dict(row, used_mib=1), {}, 8192))
        self.assertFalse(floor.idle(dict(row, used_mib=2), {}, 8192))
        self.assertFalse(floor.idle(dict(row, free_mib=1), {}, 8192))


if __name__ == '__main__':
    unittest.main()
