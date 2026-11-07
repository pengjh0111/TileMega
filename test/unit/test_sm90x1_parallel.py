"""CPU fixtures for process-level selection dispatch and provenance."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from sm90d1_common import read,write
import sm90x1_parallel as parallel
import sm90x1_hardware as hardware


class ParallelTest(unittest.TestCase):
    def test_explicit_authorization_and_selection_scope(self):
        with patch.object(parallel,'read',return_value=dict(parallel_selection_processes=True,
                                                           cross_gpu_comparisons_allowed=True)):
            self.assertTrue(parallel.enabled(dict(kind='sel')))
            self.assertFalse(parallel.enabled(dict(kind='corr')))
        with patch.object(parallel,'read',return_value=dict(parallel_selection_processes=True)):
            self.assertFalse(parallel.enabled(dict(kind='sel')))

    def test_isolated_job_preserves_fresh_index_and_guard_command(self):
        jobs=[dict(name=f'job-{i}',after=[f'job-{i-1}'],command=['guard','--fresh',str(i)],
                   env=dict(CUDA_VISIBLE_DEVICES='2')) for i in range(15)]
        original=copy.deepcopy(jobs)
        for i in range(15):
            row=parallel.isolate(jobs,f'job-{i}')[0]
            self.assertEqual(row['after'],[])
            self.assertEqual(row['command'],original[i]['command'])
            self.assertEqual(row['env'],original[i]['env'])
        self.assertEqual(jobs,original)
        with self.assertRaises(ValueError):parallel.isolate(jobs,'missing')
        with self.assertRaises(ValueError):parallel.isolate(jobs+jobs,'job-0')

    def test_adoption_preserves_completed_and_active_processes_only(self):
        with tempfile.TemporaryDirectory() as temp:
            unit=Path(temp)
            write(unit/'parallel-adoption.json',dict(source_try='try1',preserved_jobs=['done','active'],
                  scheduler=dict(pid=123,start='identity')))
            write(unit/'try1/spec.json',dict(gpu=1))
            write(unit/'try1/scheduler/state.json',{'done':dict(status='done'),
                  'active':dict(status='running'),'pending':dict(status='pending')})
            with patch.object(parallel,'live_start',return_value='identity'):
                rows,live=parallel.adopted_state(unit)
            self.assertTrue(live);self.assertEqual(set(rows),{'done','active'})
            self.assertEqual(rows['active']['status'],'running')
            write(unit/'try1/scheduler/state.json',{'done':dict(status='done'),'active':dict(status='failed')})
            with self.assertRaises(ValueError):parallel.adopted_state(unit)

    def exercise_coordinator(self):
        with tempfile.TemporaryDirectory() as temp:
            unit=Path(temp)/'unit';template=Path(temp)/'spec.json'
            spec=dict(name='fixture',kind='sel',order=['W','L','Wstar']*5,
                      arms={label:dict(decode_sha256='d',prefill_sha256='p') for label in ('W','L','Wstar')})
            write(template,spec);active=0;maximum=0;lock=threading.Lock();barrier=threading.Barrier(4)
            def process(template,unit,index):
                nonlocal active,maximum
                with lock:active+=1;maximum=max(maximum,active)
                if index<4:barrier.wait(timeout=10)
                path=unit/'parallel/jobs'/str(index)/'try1/spec.json';write(path,spec)
                with lock:active-=1
                return dict(status='done',gpu=index%4,out=str(path.parent))
            def collect(argv,**kwargs):
                if argv[2]!='collect':raise AssertionError(argv)
                rows=[dict(arm=label,gpu=i%4) for i,label in enumerate(spec['order'])]
                write(Path(argv[argv.index('--out')+1]),dict(rows=rows,arms={k:{} for k in spec['arms']}))
            policy=dict(parallel_selection_processes=True,cross_gpu_comparisons_allowed=True,
                        max_parallel_timing_processes=4,authorization='fixture user')
            real_read=read
            def read_policy(path):return policy if Path(path)==parallel.POLICY else real_read(path)
            with patch.object(parallel,'read',side_effect=read_policy), \
                 patch.object(parallel,'one_process',side_effect=process), \
                 patch.object(parallel,'validate_processes') as validate, \
                 patch.object(parallel.subprocess,'check_call',side_effect=collect), \
                 patch.object(parallel.time,'sleep',return_value=None), \
                 patch.object(hardware,'audit_unit',return_value=dict(passed=True)):
                result=parallel.run_unit(template,unit,'trial')
            self.assertEqual(maximum,4)
            self.assertEqual(len(result['rows']),15)
            self.assertEqual(len(read(unit/'parallel/scheduler/state.json')),15)
            self.assertEqual(read(unit/'unit.json')['gpus'],[0,1,2,3])
            self.assertTrue(all(len(arm['gpus'])>=3 for arm in result['arms'].values()))
            validate.assert_called_once()

    def test_device_round_and_binary_provenance_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'out';local=root/'spec.json'
            spec=dict(name='fixture',order=['W'],arms=dict(W=dict(decode_sha256='d',prefill_sha256='p')))
            name=parallel.names(spec)[0]
            state={name:dict(status='done',gpu=0,out=str(out),process_spec=str(local))}
            write(local,dict(spec,gpu=0));process=dict(fresh=0,arm='W',decode_sha256='d',prefill_sha256='p')
            write(out/'job/sm90r3-process.json',process)
            identity=dict(nvml_uuid='GPU-test',cuda_uuid='test')
            write(out/'job/physical-device-identity.json',identity)
            (out/'guard.jsonl').write_text(json.dumps(dict(phase='running',owners={'123':1}))+'\n')
            policy=dict(allowed_timing_gpu_indices=[0],idle_power_mapping_provenance=dict(uuid_by_gpu={'0':'GPU-test'}))
            real_read=read
            def read_policy(path):return policy if Path(path)==parallel.dispatch.POLICY else real_read(path)
            with patch.object(parallel,'read',side_effect=read_policy), \
                 patch.object(parallel.llm,'job_result',return_value=dict(gpu=0)):
                parallel.validate_processes(spec,state)
                for field,bad in [('fresh',1),('arm','L'),('decode_sha256','other')]:
                    write(out/'job/sm90r3-process.json',dict(process,**{field:bad}))
                    with self.subTest(field=field),self.assertRaises(ValueError):parallel.validate_processes(spec,state)
                write(out/'job/sm90r3-process.json',process)
                write(out/'job/physical-device-identity.json',dict(identity,nvml_uuid='GPU-other'))
                with self.assertRaises(ValueError):parallel.validate_processes(spec,state)

    def test_five_fresh_cpu_coordinators_preserve_all_rounds(self):
        code='''import sys
sys.path.insert(0,sys.argv[1])
from test_sm90x1_parallel import ParallelTest
ParallelTest().exercise_coordinator()
print('passed')
'''
        for _ in range(5):
            value=subprocess.run([sys.executable,'-c',code,str(ROOT/'test/unit')],text=True,
                 capture_output=True,timeout=30,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
            self.assertEqual(value.returncode,0,value.stderr)
            self.assertEqual(value.stdout.strip(),'passed')


if __name__=='__main__':unittest.main()
