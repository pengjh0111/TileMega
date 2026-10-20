"""AS-2 project isolation and owned-process timeout contracts, without NPU work."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

REPO=Path(__file__).resolve().parents[2]
ROOT=Path('/home/d50064638/junhuip')
WORK=ROOT/'tilemega-as2'
RUNS=REPO/'runs/as2'
spec=importlib.util.spec_from_file_location('as1_cleanup',REPO/'scripts/as1_cleanup.py')
cleanup=importlib.util.module_from_spec(spec);spec.loader.exec_module(cleanup)


class AscendAs2Safety(unittest.TestCase):
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
        try:
            result=subprocess.run([sys.executable,'-m','tilemega.bench.as2_run','--physical-device','1',
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
