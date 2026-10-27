"""AS-2 project isolation and owned-process timeout contracts, without NPU work."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import Mock, patch

REPO=Path(__file__).resolve().parents[2]
ROOT=Path('/home/d50064638/junhuip')
WORK=ROOT/'tilemega-as2'
RUNS=REPO/'runs/as2'
spec=importlib.util.spec_from_file_location('as1_cleanup',REPO/'scripts/as1_cleanup.py')
cleanup=importlib.util.module_from_spec(spec);spec.loader.exec_module(cleanup)


class AscendAs2Safety(unittest.TestCase):
    def test_management_driver_libraries_do_not_change_compute_environment(self):
        from tilemega.bench import as1_run as runner
        proc=Mock(pid=12345,returncode=0);proc.communicate.return_value=('healthy query',None)
        identity=dict(pid=12345,start_ticks=99,process_group=12345,session=12345)
        with patch.dict(os.environ,{'LD_LIBRARY_PATH':'/project/cann/lib64'}):
            with patch.object(runner.subprocess,'Popen',return_value=proc) as spawn, patch.object(runner,'process_identity',return_value=identity):
                state=runner.snapshot(60)
            self.assertEqual(os.environ['LD_LIBRARY_PATH'],'/project/cann/lib64')
        env=spawn.call_args.kwargs['env']
        self.assertEqual(env['LD_LIBRARY_PATH'].split(':')[-1],'/project/cann/lib64')
        self.assertEqual(env['LD_LIBRARY_PATH'].split(':')[:-1],state['driver_library_precedence'])
        self.assertEqual(spawn.call_args.args[0],state['command'])
        self.assertEqual(state['command'][-1],'info')
        proc.communicate.assert_called_once_with(timeout=60)
        self.assertEqual(state['query_timeout_s'],60)
        with patch.object(runner.subprocess,'Popen') as spawn:
            for value in (0,-1,61):
                with self.assertRaises(ValueError):runner.snapshot(value)
            spawn.assert_not_called()

    def test_management_query_reaping_is_bounded_without_repeated_spawn(self):
        from tilemega.bench import as1_run as runner
        identity=dict(pid=12345,start_ticks=99,process_group=12345,session=12345)
        proc=Mock(pid=12345);proc.poll.return_value=None
        proc.communicate.side_effect=[subprocess.TimeoutExpired('npu-smi',15),subprocess.TimeoutExpired('npu-smi',1)]
        try:
            with patch.object(runner.subprocess,'Popen',return_value=proc) as spawn, patch.object(runner,'process_identity',return_value=identity):
                state=runner.snapshot(15)
                self.assertTrue(state['query_timed_out']);self.assertTrue(state['query_may_remain_in_kernel_wait'])
                self.assertEqual(state['query_identity'],identity);proc.kill.assert_called_once()
                self.assertEqual([c.kwargs['timeout'] for c in proc.communicate.call_args_list],[15,1])
                pending=runner.snapshot();self.assertTrue(pending['query_may_remain_in_kernel_wait'])
                self.assertEqual(spawn.call_count,1)
                proc.poll.return_value=-9
                proc.communicate.side_effect=[('healthy query',None)];proc.returncode=0
                fresh=runner.snapshot();self.assertEqual(fresh['output'],'healthy query')
                self.assertEqual(spawn.call_count,2)
        finally:
            runner._pending_telemetry.clear()

    def test_management_query_identity_mismatch_is_not_signalled(self):
        from tilemega.bench import as1_run as runner
        identity=dict(pid=12345,start_ticks=99,process_group=12345,session=12345)
        reused={**identity,'start_ticks':100}
        proc=Mock(pid=12345);proc.poll.return_value=None
        proc.communicate.side_effect=[subprocess.TimeoutExpired('npu-smi',15),subprocess.TimeoutExpired('npu-smi',1)]
        try:
            with patch.object(runner.subprocess,'Popen',return_value=proc), patch.object(runner,'process_identity',side_effect=[identity,reused]):
                state=runner.snapshot(15);self.assertTrue(state['query_may_remain_in_kernel_wait'])
                proc.kill.assert_not_called()
        finally:
            runner._pending_telemetry.clear()

    def test_telemetry_failure_is_not_reported_as_hbm_exhaustion(self):
        from tilemega.bench import as1_run as runner
        for snapshot, status, other in [
            (dict(exit_code=187,output='dcmi module initialize failed. ret is -8005'), 'telemetry_unavailable', None),
            (dict(exit_code=0,output='unrecognized table'), 'telemetry_unavailable', None),
            (dict(exit_code=0,output='| 1 | Ascend950PR | OK |\n| 0000:D7:00.0 | 0 | 0 / 0 | 130000 / 131072 |'), 'blocked-resource', False),
            (dict(exit_code=0,output='| 1 | Ascend950PR | Critical |\n| 0000:D7:00.0 | 0 | 0 / 0 | 5000 / 131072 |'), 'blocked-health', False),
        ]:
            out=RUNS/'jobs'/('telemetry_contract_'+str(time.time_ns()))
            argv=['as2_run','--physical-device','1','--required-free-mib','8192','--out',str(out),'--',sys.executable,'-c','pass']
            with patch.object(sys,'argv',argv), patch.object(runner,'snapshot',return_value=snapshot), patch.object(runner.subprocess,'Popen') as spawn:
                self.assertEqual(runner.main(project='as2'),75)
                spawn.assert_not_called()
            receipt=json.loads((out/'job.json').read_text())
            self.assertEqual(receipt['status'],status)
            self.assertIs(receipt['other_processes_present_before'],other)
            self.assertFalse(receipt['device_locked'])
            if status=='telemetry_unavailable':
                self.assertNotEqual(receipt['mapping_evidence'],'verified')
            if status=='blocked-resource':
                self.assertEqual(receipt['pci_bus_id'],'0000:d7:00.0')

    def test_unfinished_owned_receipt_is_observed_without_a_lock(self):
        from tilemega.bench.as2_run import unfinished_owned_jobs
        from tilemega.bench.as1_run import process_identity
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True)
        out=RUNS/'jobs'/('own_submission_contract_'+str(time.time_ns()));out.mkdir()
        receipt=dict(project='as2',status='running',**process_identity(child.pid))
        try:
            (out/'job.json').write_text(json.dumps(receipt))
            self.assertTrue(any(j['pid']==child.pid for j in unfinished_owned_jobs()))
            receipt['start_ticks']+=1;(out/'job.json').write_text(json.dumps(receipt))
            self.assertFalse(any(j['pid']==child.pid for j in unfinished_owned_jobs()))
            self.assertIsNone(child.poll())
            self.assertEqual({p.name for p in out.iterdir()},{'job.json'})
        finally:
            child.terminate();child.wait(timeout=5)
            receipt['status']='passed';(out/'job.json').write_text(json.dumps(receipt))

    def test_as2_cleanup_cannot_reach_as1(self):
        run,work,roots=cleanup.project_paths('as2')
        self.assertEqual(run,RUNS);self.assertEqual(work,WORK)
        for p in [REPO/'runs/as1/manifest.jsonl',ROOT/'tilemega-as1/build/sentinel',ROOT/'unrelated']:
            with self.assertRaises(ValueError):cleanup.plain_path(p,roots)
        report=RUNS/'probes'/('cleanup_dry_'+str(time.time_ns())+'.json')
        before=(RUNS/'manifest.jsonl').read_bytes()
        subprocess.run([sys.executable,str(REPO/'scripts/as2_cleanup.py'),'--report-json',str(report)],check=True,capture_output=True)
        result=json.loads(report.read_text());self.assertEqual(result['mode'],'dry-run')
        self.assertEqual(result['removed_files'],0);self.assertEqual(before,(RUNS/'manifest.jsonl').read_bytes())

    def test_timeout_stops_owned_descendant_and_preserves_sentinel(self):
        sentinel=subprocess.Popen([sys.executable,'-c','import time;time.sleep(40)'],start_new_session=True)
        out=RUNS/'jobs'/('cpu_timeout_contract_'+str(time.time_ns()))
        code="""import os,subprocess,sys,time
child=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(40)'])
print(child.pid,flush=True)
time.sleep(40)
"""
        env=os.environ.copy();env['PYTHONPATH']=str(REPO/'python')
        entry="""import runpy
from unittest.mock import patch
from tilemega.bench import as1_run
with patch.object(as1_run,'snapshot',return_value={'exit_code':0,'output':''}):
    runpy.run_module('tilemega.bench.as2_run',run_name='__main__')
"""
        try:
            result=subprocess.run([sys.executable,'-c',entry,'--physical-device','1',
                '--out',str(out),'--timeout-s','1','--sample-s','1','--',sys.executable,'-c',code],
                env=env,capture_output=True,text=True,timeout=25)
            self.assertEqual(result.returncode,124,result.stderr)
            job=json.loads((out/'job.json').read_text())
            self.assertEqual(job['project'],'as2');self.assertFalse(job['device_locked'])
            self.assertGreater(job['start_ticks'],0);self.assertEqual(job['session'],job['pid'])
            child=int((out/'stdout.log').read_text().strip())
            stat=Path('/proc',str(child),'stat')
            self.assertTrue(not stat.exists() or stat.read_text().split(') ',1)[1].split()[0]=='Z')
            self.assertIsNone(sentinel.poll())
        finally:
            sentinel.terminate();sentinel.wait(timeout=5)


if __name__=='__main__':unittest.main()
