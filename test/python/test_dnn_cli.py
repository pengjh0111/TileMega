import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tilemega.dnn.cli import read_config


class DnnCliTest(unittest.TestCase):
    def test_bound_batch_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config.json'
            for batches in ([True],[0],[65],[2,2]):
                path.write_text(json.dumps(dict(model=dict(name='bert'),workload=dict(batch=batches))))
                with self.assertRaises(ValueError):
                    read_config(path)
            path.write_text(json.dumps(dict(model=dict(name='bert'),workload=dict(batch=[1,8,32]))))
            config=read_config(path)
            self.assertEqual(config['workload']['batch'],[1,8,32])
            self.assertTrue(config['device']['cache_dir'].endswith('tilemega-dm'))
            invalid=dict(model=dict(name='bert'),features=dict(deferred_ln='1'))
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError,'deferred_ln'):read_config(path)
            invalid['features']=dict(unimplemented=True);path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError,'unsupported DNN features'):read_config(path)

    def test_reduction_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config.json'
            data=dict(model=dict(name='nafnet'),features=dict(global_la='0',
                nonpaged_la=0,paged_la=0,paged_la_splitk=0))
            path.write_text(json.dumps(data))
            features=read_config(path)['features']
            self.assertEqual(features['global_la'],'0')
            self.assertEqual(features['nonpaged_la'],0)
            data['features']['global_la']='1'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'global_la'):read_config(path)

    def test_inherited_gpu_lock(self):
        # A nested correctness CLI must retain exclusivity without waiting on
        # its own scheduler's inherited flock description.
        script='''from tilemega.dnn.cli import gpu_lock
import os, subprocess
with gpu_lock():
    assert subprocess.run(['flock','-n',os.environ['TILEMEGA_GPU_LOCK'],'true']).returncode != 0
print('exclusive')
'''
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'gpu.lock'
            env=dict(os.environ,TILEMEGA_GPU_LOCK=str(path))
            for command in ([sys.executable,'-c',script],
                    ['flock',str(path),sys.executable,'-c',script]):
                result=subprocess.run(command,env=env,text=True,capture_output=True,timeout=10,check=True)
                self.assertEqual(result.stdout.strip(),'exclusive')
                self.assertEqual(subprocess.run(['flock','-n',str(path),'true']).returncode,0)


if __name__=='__main__':
    unittest.main()
