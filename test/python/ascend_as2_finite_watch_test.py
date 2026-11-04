"""Only CPU child processes; the deadline must also cover new child sessions."""
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from tilemega.bench.as2_finite_process import run,Deadline
from tilemega.bench.as2_queue import RUNS
from tilemega.bench.as2_taskbody_watch import interval_summary,available_devices


class FiniteWatch(unittest.TestCase):
    def test_process_idle_applies_to_both_functional_and_performance_work(self):
        output='''| 0 | Ascend950PR | OK | 55 | 41 |
| | | 0000:D7:00.0 | 95 0 / 0 7000 / 131072 |
| 1 | Ascend950PR | Alarm | 55 | 41 |
| | | 0000:55:00.0 | 100 0 / 0 5000 / 131072 |
| NPU ID | Process id | Process name | Process memory(MB) |
| 0 | 123456 | python | 332 |
+------------------+
'''
        state=dict(exit_code=0,output=output)
        self.assertEqual(set(available_devices(state,16000,'functional')),{1})
        self.assertEqual(set(available_devices(state,16000,'performance')),{1})
        self.assertEqual(available_devices(state,130000,'functional'),{})
        self.assertEqual(available_devices(dict(state,exit_code=1),0,'functional'),{})

    def test_deadline_ends_new_session_descendant_and_preserves_unrelated_child(self):
        with tempfile.TemporaryDirectory(dir=RUNS/'probes') as raw:
            root=Path(raw);pidfile=root/'descendant.pid'
            unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True)
            script=('import subprocess,sys,time; child=subprocess.Popen([sys.executable,"-c",'
                    '"import time;time.sleep(20)"],start_new_session=True);'
                    'open(sys.argv[1],"w").write(str(child.pid));time.sleep(20)')
            try:
                start=time.monotonic()
                with self.assertRaises(Deadline):run([sys.executable,'-c',script,pidfile],root/'log',start+1.5,20)
                self.assertLess(time.monotonic()-start,4)
                self.assertIsNone(unrelated.poll())
                stat=Path('/proc',pidfile.read_text(),'stat')
                self.assertTrue(not stat.exists() or stat.read_text().split(') ',1)[1][0]=='Z')
            finally:unrelated.terminate();unrelated.wait(timeout=2)

    def test_short_success_does_not_wait_until_budget_end(self):
        with tempfile.TemporaryDirectory(dir=RUNS/'probes') as raw:
            self.assertEqual(run([sys.executable,'-c','pass'],Path(raw)/'log',time.monotonic()+5,2),0)

    def test_incomplete_pairs_never_produce_a_performance_pass(self):
        self.assertEqual(interval_summary([])['status'],'incomplete')
        self.assertEqual(interval_summary([dict(pair=0,variant='control')])['pairs_complete'],0)


if __name__=='__main__':unittest.main()
