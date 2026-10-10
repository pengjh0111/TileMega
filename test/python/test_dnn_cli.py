import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tilemega.dnn.cli import read_config, model_export, check, report


class DnnCliTest(unittest.TestCase):
    def test_explicit_calibration_reference_preserves_target_resources(self):
        from tilemega.dnn.target import resolve_target
        from tilemega.fingerprint import ROOT
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = json.loads((ROOT/'configs/targets/sm_89.json').read_text())
            original['resources']['num_sms'] = 2
            source = root/'target.json'
            source.write_text(json.dumps(original))
            reference = ROOT/'docs/experiments/DNN_MOE_R1/inputs/regression/llama_B1/prefill/target.json'
            config = dict(target=str(source), solver=dict(calibration_reference=str(reference)))
            resolved = resolve_target(config, root/'resolved')
            target = json.loads(resolved.read_text())
            self.assertEqual(target['resources'], original['resources'])
            self.assertEqual(target['calibration_by_dtype']['bf16']['pipelines'],
                             original['calibration_by_dtype']['bf16']['pipelines'])
            archived = json.loads(reference.read_text())
            for key in ('task_publication', 'task_wait', 'task_source', 'task_source_sha256'):
                self.assertEqual(target['event_calibration_by_dtype']['bf16'][key],
                                 archived['event_calibration_by_dtype']['bf16'][key])
            provenance = json.loads((resolved.parent/'resolved_target.provenance.json').read_text())
            self.assertIn('inferred', provenance['scope'])
            self.assertIn('serving_hop', provenance['fields'])
            with self.assertRaises(FileExistsError):
                resolve_target(config, root/'resolved')
            original['arch_tag'] = 'sm_80'
            source.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, 'architecture differs'):
                resolve_target(config, root/'other')
            self.assertEqual(resolve_target(dict(target=str(source)), root/'plain'), source)

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

    def test_synthetic_smoke_preserves_dataset_evidence_and_checks_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'B2').mkdir()
            plan=dict(batch=2,library='plan.so',export='export',bridge='bridge.json',artifact_id='id')
            official=root/'B2/correctness.json'
            official.write_text(json.dumps(dict(artifact_id='id',passed=False,metrics=dict(cosine=.9))))
            before=official.read_bytes()
            receipt=dict(artifact_id='id',passed=True,scope='smoke; not G-DNN')
            with patch('tilemega.dnn.check_generated.check',return_value=receipt) as execute:
                check({},[plan],root,synthetic_weights=True)
                execute.assert_called_once_with('plan.so','export','bridge.json',2,synthetic_weights=True)
                with self.assertRaises(FileExistsError):
                    check({},[plan],root,synthetic_weights=True)
            self.assertEqual(official.read_bytes(),before)
            report([plan],root)
            row=json.loads((root/'report.json').read_text())['plans'][0]
            self.assertEqual(row['correctness'],'failed')
            self.assertTrue(row['smoke']['passed'])
            receipt['artifact_id']='other'
            (root/'B2/smoke.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,'another artifact'):
                report([plan],root)

    def test_architecture_export_requires_explicit_synthetic_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'manifest.json').write_text(json.dumps(dict(model='mbv1',
                accuracy_eligible=False,artifacts={})))
            config=dict(model=dict(name='mbv1',export=str(root)))
            with self.assertRaisesRegex(ValueError,'pretrained weights'):
                model_export(config,root)
            config['model']['structure_only']=True
            self.assertEqual(model_export(config,root),root)
            with self.assertRaisesRegex(ValueError,'synthetic-weights'):
                check(config,[],root)

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
