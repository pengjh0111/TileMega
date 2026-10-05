#!/usr/bin/env python3
import json,tempfile,unittest
from pathlib import Path
from ledger_r14 import tasks,linear_fit
from phase_a import timed_tokens
class TestPhaseA(unittest.TestCase):
    def test_fit(self):
        self.assertEqual(linear_fit([dict(bytes=1,run_ns=5),dict(bytes=3,run_ns=9)]),dict(slope_ns_per_byte=2,intercept_ns=3))
    def test_task_span(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'tasks.tsv';p.write_text('kind\tbytes\trun_begin\trun_end\nattention\t4\t9\t8\n')
            with self.assertRaises(ValueError):tasks(p)
    def test_exact_full_tokens(self):
        r=dict(runs=[dict(N=1024,warmup=True,tokens=[[0]]),dict(N=1024,warmup=False,tokens=[[1,2]])])
        self.assertEqual(timed_tokens(r),[[1,2]])
        r['runs'].append(dict(r['runs'][1]))
        with self.assertRaises(ValueError):timed_tokens(r)
if __name__=='__main__':unittest.main()
