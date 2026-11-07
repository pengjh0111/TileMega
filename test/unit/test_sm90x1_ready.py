"""Readiness must depend on one variant, while machine units remain exclusive."""
import fcntl
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from sm90x1_ready import inventory,choose,post_retry
import sm90x1


class ReadyTest(unittest.TestCase):
    def retry_fixture(self,directory,code):
        path=directory/'L/corr/try1/scheduler/state.json';path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(job=dict(status='failed',exit_code=code))))

    def test_post_retries_abnormal_exit_after_controller_resume(self):
        parent=ROOT/'runs/sm90x1/test-ready';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            directory=Path(temporary);self.retry_fixture(directory,247)
            records=json.loads(json.dumps([dict(cell='c',code=1,at=100)]))
            self.assertEqual(post_retry(directory,'c',records,131),(True,'abnormal_process_exit'))
            self.assertEqual(post_retry(directory,'c',records,110),(False,'retry_wait'))
            self.assertEqual(post_retry(directory,'c',records*3,131),(False,'attempt_limit'))

    def test_post_does_not_retry_numerical_or_unknown_failure(self):
        parent=ROOT/'runs/sm90x1/test-ready';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            directory=Path(temporary);self.retry_fixture(directory,1)
            records=[dict(cell='c',code=1,at=100)]
            self.assertEqual(post_retry(directory,'c',records,131),(False,'failure_requires_review'))
            self.assertEqual(post_retry(directory,'new',records,131),(True,'first_attempt'))
            (directory/'L/corr/result.json').write_text(json.dumps({'pass':False}))
            self.assertEqual(post_retry(directory,'c',records,131),(False,'correctness_failed'))

    def test_post_retains_partial_correctness_failure_after_a_process_kill(self):
        parent=ROOT/'runs/sm90x1/test-ready';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            directory=Path(temporary);self.retry_fixture(directory,247)
            out=directory/'raw';measurements=out/'job/measurements';measurements.mkdir(parents=True)
            (measurements/'c2.json').write_text(json.dumps({'pass':False}))
            state=directory/'L/corr/try1/scheduler/state.json'
            state.write_text(json.dumps(dict(job=dict(status='failed',exit_code=247,out=str(out)))))
            self.assertEqual(post_retry(directory,'c',[dict(cell='c',code=1,at=100)],131),
                             (False,'correctness_failed'))

    def test_five_fresh_submitters_reuse_one_trace_receipt(self):
        self.check_fresh_submitters('trace')

    def test_five_fresh_post_submitters_reuse_one_measurement_flow(self):
        self.check_fresh_submitters('post')

    def check_fresh_submitters(self,mode):
        parent=ROOT/'runs/sm90x1/test-ready';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            (Path(temporary)/'cells/cell/phase2/W').mkdir(parents=True)
            program='''
import os,sys,time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,sys.argv[1])
import sm90x1 as base
base.RUN=Path(sys.argv[2])
def fake_unit(a,c,build_id,reference):
    artifact=base.directory(c)/a.tag/build_id
    receipt=artifact/'receipt'
    if receipt.exists():return
    active=artifact/'active'
    fd=os.open(active,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.close(fd)
    time.sleep(.1)
    with (artifact/'executions').open('a') as out:out.write(str(os.getpid())+'\\n')
    receipt.write_text('accepted')
    active.unlink()
base._trace_cell=fake_unit
if sys.argv[3]=='post':
    import sm90x1_ready as ready
    ready.post_locked=lambda a:fake_unit(SimpleNamespace(tag='phase2'),dict(cell='cell'),'W',False)
    ready.post_one(SimpleNamespace(cell='cell',phase=2))
else:base.trace_cell(SimpleNamespace(tag='phase2'),dict(cell='cell'),'W')
'''
            processes=[subprocess.Popen([sys.executable,'-c',program,str(ROOT/'scripts'),temporary,mode],
                                        stdout=subprocess.PIPE,stderr=subprocess.PIPE) for _ in range(5)]
            results=[p.communicate(timeout=15) for p in processes]
            self.assertEqual([p.returncode for p in processes],[0]*5,results)
            artifact=Path(temporary)/'cells/cell/phase2/W'
            self.assertEqual(len((artifact/'executions').read_text().splitlines()),1)
            self.assertEqual((artifact/'receipt').read_text(),'accepted')

    def test_user_policy_overrides_class_and_keeps_frozen_file(self):
        import hashlib
        original=hashlib.sha256(sm90x1.CELLS.read_bytes()).hexdigest()
        cells=sm90x1.cells('all')
        self.assertTrue(all(c['class']=='trial' for c in cells))
        self.assertTrue(all(c['recorded_class']=='machine' for c in cells if c['moe']))
        with patch.object(sm90x1,'EXECUTION_POLICY',ROOT/'runs/sm90x1/nonexistent-policy.json'):
            self.assertTrue(all(c['class']=='machine' for c in sm90x1.cells('all') if c['moe']))
        self.assertEqual(hashlib.sha256(sm90x1.CELLS.read_bytes()).hexdigest(),original)

    def test_unbuilt_sibling_does_not_block_ready_variants(self):
        parent=ROOT/'runs/sm90x1/test-ready';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            run=Path(temporary);cells=[dict(cell=c,**{'class':'trial'}) for c in ['a','b']]
            for c in cells:
                directory=run/'cells'/c['cell']/'phase2';directory.mkdir(parents=True)
                (directory/'build-plan.json').write_text(json.dumps([dict(id='W'),dict(id='v1')]))
                artifact=directory/'W';artifact.mkdir()
                for name in ('binding.json','trace.so','trace.so.identity.json'):(artifact/name).write_text('{}')
            rows=inventory(cells,2,run)
            self.assertEqual([(r['cell'],r['variant']) for r in choose(rows,2,[])],[('a','W'),('b','W')])
            self.assertEqual(sum(r['status']=='waiting_native' for r in rows),2)
            with (run/'cells/a/phase2/W/trace-unit.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                rows=inventory(cells,2,run)
                self.assertEqual([(r['cell'],r['variant']) for r in choose(rows,2,[])],[('b','W')])

    def test_machine_admission_is_single_and_drains_trials(self):
        machine=[dict(cell='m',variant=str(i),status='ready',**{'class':'machine'}) for i in range(3)]
        trial=dict(cell='d',variant='W',status='ready',**{'class':'trial'})
        self.assertEqual(len(choose(machine,4,[])),1)
        self.assertEqual(choose(machine,4,['trial']),[])
        self.assertEqual(choose([trial],4,['machine']),[])
        self.assertEqual(choose(machine+[trial],4,[]),[trial])


if __name__=='__main__':unittest.main()
