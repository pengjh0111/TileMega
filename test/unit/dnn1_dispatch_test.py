"""CPU-only checks for idle hints, accepted cells and parallel isolation."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('dnn1_dispatch',ROOT/'scripts/dnn1_dispatch.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
baseline=module.baseline


class Dispatch(unittest.TestCase):
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
