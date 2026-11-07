"""X1 scheduling waits may change; guard acceptance and timeout clocks may not."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from sm90x1_scheduler import PollingClock,shorten_retry_waits,abandon_preflight,process_start


class SchedulerWaitTest(unittest.TestCase):
    def check_fresh_handoffs(self,active):
        parent=ROOT/'runs/sm90x1/test-scheduler';parent.mkdir(parents=True,exist_ok=True)
        program='''
import json,subprocess,sys,time
from pathlib import Path
out=Path(sys.argv[1]);active=int(sys.argv[2])
guard='import os,sys,time; from pathlib import Path; out=Path(sys.argv[1]); '+(' (out/"pgid").write_text(str(os.getpid())); ' if active else '')+'(out/"ready").touch(); time.sleep(60)'
p=subprocess.Popen([sys.executable,'-c',guard,str(out),'tilemega.bench.gpu_guard'])
while not (out/'ready').exists():time.sleep(.001)
(out/'state.json').write_text(json.dumps(dict(job=dict(status='running',pid=p.pid,out=str(out)))))
p.wait()
'''
        for _ in range(5):
            with tempfile.TemporaryDirectory(dir=parent) as temporary:
                out=Path(temporary);process=subprocess.Popen([sys.executable,'-c',program,temporary,str(int(active))])
                guard=None;token=None
                try:
                    deadline=time.monotonic()+10
                    while not (out/'state.json').exists():
                        if process.poll() is not None or time.monotonic()>deadline:self.fail('fixture did not start')
                        time.sleep(.005)
                    guard=json.loads((out/'state.json').read_text())['job']['pid'];token=process_start(guard)
                    self.assertEqual(abandon_preflight(process,out/'state.json'),not active)
                    if active:
                        self.assertIsNone(process.poll())
                        self.assertEqual(process_start(guard),token)
                    else:self.assertEqual(process.returncode,-signal.SIGTERM)
                finally:
                    if process.poll() is None:process.terminate();process.wait(timeout=10)
                    if guard and process_start(guard)==token:os.kill(guard,signal.SIGTERM)

    def test_five_fresh_preflight_waits_can_handoff(self):
        self.check_fresh_handoffs(False)

    def test_five_fresh_started_children_are_retained(self):
        self.check_fresh_handoffs(True)

    def test_interference_retries_do_not_hold_a_fixed_ten_minute_wait(self):
        state={'guard':dict(status='pending',exit_code=75,ended=100,retry_at=700),
               'done':dict(status='done',exit_code=0,ended=100,retry_at=700),
               'failed':dict(status='pending',exit_code=1,ended=100,retry_at=700)}
        shorten_retry_waits(state,5)
        self.assertEqual(state['guard']['retry_at'],105)
        self.assertEqual(state['done']['retry_at'],700)
        self.assertEqual(state['failed']['retry_at'],700)

    def test_earlier_retry_and_missing_history_are_preserved(self):
        state={'early':dict(status='pending',exit_code=75,ended=100,retry_at=101),
               'new':dict(status='pending',retry_at=0)}
        shorten_retry_waits(state,5)
        self.assertEqual(state['early']['retry_at'],101)
        self.assertEqual(state['new']['retry_at'],0)

    def test_only_scheduler_sleep_is_shortened(self):
        clock=Mock();clock.time.return_value=123;clock.monotonic.return_value=456
        proxy=PollingClock(clock,5)
        proxy.sleep(60);proxy.sleep(.25)
        self.assertEqual([c.args[0] for c in clock.sleep.call_args_list],[5,.25])
        self.assertEqual(proxy.time(),123)
        self.assertEqual(proxy.monotonic(),456)


if __name__=='__main__':unittest.main()
