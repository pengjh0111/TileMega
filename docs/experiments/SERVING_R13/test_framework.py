#!/usr/bin/env python3
"""CPU checks for strict preflight and foreign-process classification."""
import os,select,signal,subprocess,sys,unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from gpu_guard import external,idle,stop
from anchor import measurement_policy
class PolicyTests(unittest.TestCase):
    def test_explicit_policy_never_falls_back_to_another_device(self):
        with tempfile.TemporaryDirectory() as folder:
            policy=Path(folder)/'policy.json';policy.write_text('{}')
            with patch.dict(os.environ,{'TILEMEGA_MEASUREMENT_POLICY':str(policy)}):
                self.assertEqual(measurement_policy(Path(folder)),policy)
            with patch.dict(os.environ,{'TILEMEGA_MEASUREMENT_POLICY':str(policy)+'missing'}):
                with self.assertRaises(RuntimeError):measurement_policy(Path(folder))
class GuardTests(unittest.TestCase):
    def test_idle_limits(self):
        p=dict(max_util_pct=5,idle_power_w=21.81,power_margin_w=30,max_hidden_mib=1024)
        r=dict(owners={},utilization_pct=5,power_w=51.81,hidden_mib=1024,free_mib=12288)
        self.assertTrue(idle(r,p,12288))
        for k,v in [('utilization_pct',6),('power_w',52),('hidden_mib',1025),('free_mib',12287)]:self.assertFalse(idle(dict(r,**{k:v}),p,12288))
    def test_own_and_dead(self):
        self.assertEqual(external(dict(owners={os.getpid():100,999999999:100}),{os.getpid()}),set())
    def test_parent_exit_does_not_leave_term_ignoring_worker(self):
        worker="import os,signal;signal.signal(signal.SIGTERM,signal.SIG_IGN);print(os.getpid(),flush=True);signal.pause()"
        parent="import subprocess,sys,signal;p=subprocess.Popen([sys.executable,'-c',"+repr(worker)+"],stdout=subprocess.PIPE,text=True);print(p.stdout.readline().strip(),flush=True);signal.pause()"
        p=subprocess.Popen([sys.executable,'-c',parent],stdout=subprocess.PIPE,text=True,start_new_session=True)
        child=int(p.stdout.readline().strip());pidfd=os.pidfd_open(child)
        try:
            stop(p)
            self.assertTrue(select.select([pidfd],[],[],2)[0],"worker did not exit")
            # A killed orphan may briefly remain as a zombie; it owns no GPU.
            stat=__import__('pathlib').Path('/proc',str(child),'stat')
            if stat.exists():self.assertEqual(stat.read_text().rsplit(')',1)[1].split()[0],'Z')
            self.assertIsNotNone(p.poll())
        finally:
            try:os.kill(child,signal.SIGKILL)
            except ProcessLookupError:pass
            if p.poll() is None:p.kill();p.wait()
            p.stdout.close();os.close(pidfd)
if __name__=='__main__':unittest.main()
