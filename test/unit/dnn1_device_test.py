"""CPU fixtures enforce the user's process/memory-only monitoring policy."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import dnn1_device as device


class ResourcePolicy(unittest.TestCase):
    def test_standalone_and_package_observers_both_use_the_provider(self):
        device.install()
        import baseline_common
        import tilemega.serving.baseline_common as packaged
        self.assertIs(packaged,baseline_common)
        self.assertIs(packaged._observe_one,device.observe_one)
        import tilemega.bench.gpu_guard as guard
        self.assertIs(guard.gpu,device.gpu)
        self.assertIs(guard.idle,device.idle)

    def test_queries_never_request_power_utilization_temperature_or_throttle(self):
        commands=[]
        def query(command,**kwargs):
            commands.append(command)
            if '--query-compute-apps=pid,used_memory' in command:return ''
            return '0, 81000, 81559, 1830, 2619\n'
        with patch.object(device.subprocess,'check_output',query):
            row=device.gpu(5)
        self.assertTrue(device.idle(row,dict(idle_power_w=0,power_margin_w=0),8192))
        self.assertEqual(row['sm_clock_mhz'],1830)
        for command in commands:
            for forbidden in ('power.','utilization.','temperature.','throttle','clocks_event'):
                self.assertNotIn(forbidden,' '.join(command))
        self.assertFalse(row['power_monitoring'])
        self.assertNotIn('power_w',row)

    def test_processes_memory_and_free_capacity_are_the_only_idle_checks(self):
        base=dict(gpu_index=7,owners={},used_mib=0,free_mib=81000)
        self.assertTrue(device.idle(base,{},80000))
        for change in (dict(owners={123:1}),dict(used_mib=1),dict(free_mib=1),dict(gpu_index=6)):
            self.assertFalse(device.idle(dict(base,**change),{},8192))

    def test_gpu6_is_excluded_before_any_query(self):
        with patch.object(device.subprocess,'check_output') as query:
            with self.assertRaisesRegex(ValueError,'excludes GPU6'):device.gpu(6)
            query.assert_not_called()


if __name__=='__main__':unittest.main()
