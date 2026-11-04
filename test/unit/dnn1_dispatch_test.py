"""CPU-only checks for idle hints, accepted cells and parallel isolation."""
import importlib.util
import json
import os
import signal
import subprocess
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'python'))
spec=importlib.util.spec_from_file_location('dnn1_dispatch',ROOT/'scripts/dnn1_dispatch.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
baseline=module.baseline


class Dispatch(unittest.TestCase):
    def test_completed_rejected_payload_requires_full_dead_process_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);payload=root/'raw/dnn1-cell-x/attempt1/job';payload.mkdir(parents=True)
            guard=payload.parent/'guard_result.json';session=payload/'cell-result.json'
            guard.write_text(json.dumps(dict(code=75)))
            session.write_text(json.dumps(dict(status='hard_stop',ended=20)))
            logical=payload/'logical-attempt.json'
            logical.write_text(json.dumps(dict(pid=99999999,returncode=75,ended=19)))
            job=dict(exit_code=75,ended=21)
            with patch('tilemega.bench.gpu_guard.process_start',return_value=None), \
                    patch('tilemega.bench.gpu_guard.session_members',return_value=set()):
                proof=module.completed_rejections(root,job)
                self.assertEqual(proof[0]['exited_processes'][0]['pid'],99999999)
                for field,value in [('exit_code',0),('ended',0)]:
                    self.assertIsNone(module.completed_rejections(root,dict(job,**{field:value})))
                with patch('tilemega.bench.gpu_guard.process_start',return_value='live'):
                    self.assertIsNone(module.completed_rejections(root,job))
                with patch('tilemega.bench.gpu_guard.session_members',return_value={123}):
                    self.assertIsNone(module.completed_rejections(root,job))
                session.write_text(json.dumps(dict(status='running',ended=20)))
                self.assertIsNone(module.completed_rejections(root,job))
                session.write_text(json.dumps(dict(status='hard_stop',ended=20)))
                logical.unlink()
                self.assertIsNone(module.completed_rejections(root,job))

    def test_running_payload_is_never_signalled_for_migration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'scheduler').mkdir()
            (root/'scheduler/state.json').write_text(json.dumps({'dnn1-cell-x':dict(status='running')}))
            with patch.object(os,'killpg',side_effect=AssertionError('active payload signalled')):
                self.assertIsNone(module.migrate_waiter(dict(root=root,key='x',process=None)))

    def test_only_payload_free_pending_worker_is_cancelled(self):
        for has_payload in (False,True):
            with self.subTest(has_payload=has_payload), tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);(root/'scheduler').mkdir()
                (root/'scheduler/state.json').write_text(json.dumps({'dnn1-cell-x':dict(status='pending')}))
                if has_payload:
                    (root/'raw/dnn1-cell-x/attempt1/job').mkdir(parents=True)
                process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
                try:
                    result=module.migrate_waiter(dict(root=root,key='x',process=process))
                    if has_payload:
                        self.assertIsNone(result)
                        self.assertIsNone(process.poll())
                    else:
                        self.assertEqual(result['key'],'x')
                        self.assertIsNotNone(process.poll())
                        self.assertTrue((root/'migration.json').exists())
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid,signal.SIGCONT)
                        os.killpg(process.pid,signal.SIGTERM)
                        process.wait(timeout=5)

    def test_adoption_observes_existing_launcher_without_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'launch.sh').write_text('sleep 30\n')
            process=subprocess.Popen(['bash',str(root/'launch.sh')],start_new_session=True)
            try:
                adopted=module.AdoptedWorker(process.pid,root)
                self.assertEqual(adopted.pid,process.pid)
                self.assertIsNone(adopted.poll())
            finally:
                os.killpg(process.pid,signal.SIGTERM)
                process.wait(timeout=5)

    def test_cpu_ranges_and_per_card_affinities_are_disjoint(self):
        self.assertEqual(module.cpu_list('0-2,8,10-11'),{0,1,2,8,10,11})
        def nodes(path, *args, **kwargs):
            return '0-31' if 'node0' in str(path) else '32-63'
        with patch.object(Path,'read_text',nodes):
            bindings=module.affinities([0,1,2,3,4,5,7],8,range(64))
            self.assertEqual(len(set(cpu for cpus in bindings.values() for cpu in cpus)),56)
            self.assertTrue(set(bindings[3])<=set(range(32)))
            self.assertTrue(set(bindings[4])<=set(range(32,64)))
            with self.assertRaisesRegex(ValueError,'not enough disjoint'):
                module.affinities([0,1,2,3],9,range(64))

    def test_explicit_affinity_rejects_overlap_wrong_node_and_unavailable_cpus(self):
        def nodes(path, *args, **kwargs):
            return '0-31' if 'node0' in str(path) else '32-63'
        with patch.object(Path, 'read_text', nodes):
            self.assertEqual(module.affinities([1, 4], 8, range(64),
                {'1': [8, 9], '4': [40, 41]}), {1: [8, 9], 4: [40, 41]})
            for config in ({'1': [8], '2': [8]}, {'1': [40], '2': [9]},
                           {'1': [8], '2': [80]}, {'1': [8]}):
                with self.assertRaises(ValueError):
                    module.affinities([1, 2], 8, range(64), config)

    def test_physical_lock_and_guard_idle_both_gate_dispatch(self):
        calls=[]
        with patch.object(module.device,'idle',lambda row,policy,needs: calls.append(needs) or row['idle']):
            self.assertFalse(module.available(dict(idle=True),{},8192,True)[0])
            self.assertEqual(calls,[])
            self.assertFalse(module.available(dict(idle=False),{},8192,False)[0])
            self.assertTrue(module.available(dict(idle=True),{},8192,False)[0])
            self.assertEqual(calls,[8192,8192])

    def test_reuse_requires_completed_session_and_guard_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'scheduler').mkdir();attempt=root/'attempt';(attempt/'job').mkdir(parents=True)
            cell=root/'cell.json';cell.write_text(json.dumps(dict(model='resnet18',batch=1)))
            (root/'queue-spec.json').write_text(json.dumps(dict(cells=[str(cell)])))
            (root/'scheduler/state.json').write_text(json.dumps({'dnn1-cell-resnet18_b1':dict(status='done',out=str(attempt))}))
            (attempt/'guard_result.json').write_text(json.dumps(dict(code=0)))
            session=attempt/'job/cell-result.json';session.write_text(json.dumps(dict(status='complete')))
            self.assertEqual(module.accepted(root)[0]['key'],'resnet18_b1')
            (attempt/'guard_result.json').write_text(json.dumps(dict(code=75)))
            self.assertEqual(module.accepted(root),[])
            (attempt/'guard_result.json').write_text(json.dumps(dict(code=0)))
            session.write_text(json.dumps(dict(status='running')))
            self.assertEqual(module.accepted(root),[])

    def test_parallel_session_uses_trial_metadata_and_real_device_lock(self):
        env=dict(TILEMEGA_GPU_LOCK_HELD='1',CUDA_VISIBLE_DEVICES='1',TILEMEGA_DEVICE_INDEX='1',
            TILEMEGA_GPU_LOCK='/tmp/tilemega_gpu1.lock',TILEMEGA_TIMING='1',
            TILEMEGA_NUMA_CONCURRENCY_TRIAL='1')
        cell=dict(gpu=1,parallel=True,cpu_affinity=list(range(8)))
        with patch.dict(os.environ,env,clear=True),patch.object(os,'sched_getaffinity',return_value={0,1}), \
                patch.object(baseline,'held',side_effect=lambda path:'gpu1' in path):
            baseline.require_session(cell)
            with self.assertRaises(baseline.HardStop):
                baseline.require_session(dict(gpu=1))
            with patch.dict(os.environ,{'TILEMEGA_NUMA_CONCURRENCY_VERIFIED':'1'}):
                with self.assertRaises(baseline.HardStop):baseline.require_session(cell)
            with patch.object(os,'sched_getaffinity',return_value={8}):
                with self.assertRaises(baseline.HardStop):baseline.require_session(cell)


if __name__=='__main__':unittest.main()
