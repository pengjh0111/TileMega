#!/usr/bin/env python3
"""CPU checks for strict preflight and foreign-process classification."""
import os,unittest
from unittest.mock import patch
from gpu_guard import external,idle
class GuardTests(unittest.TestCase):
    def test_idle_limits(self):
        p=dict(max_util_pct=5,idle_power_w=21.81,power_margin_w=30,max_hidden_mib=1024)
        r=dict(owners={},utilization_pct=5,power_w=51.81,hidden_mib=1024,free_mib=12288)
        self.assertTrue(idle(r,p,12288))
        for k,v in [('utilization_pct',6),('power_w',52),('hidden_mib',1025),('free_mib',12287)]:self.assertFalse(idle(dict(r,**{k:v}),p,12288))
    def test_own_and_dead(self):
        self.assertEqual(external(dict(owners={os.getpid():100,999999999:100}),{os.getpid()}),set())
if __name__=='__main__':unittest.main()
