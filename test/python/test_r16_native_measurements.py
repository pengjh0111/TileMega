"""Native receipts require binary provenance, shared weights and a passed input gate."""
import hashlib
import json
from pathlib import Path
import shutil
import unittest

import test_r16_measurement_analysis as fixtures

validation=fixtures.validation
summary=fixtures.summary


class NativeMeasurementTest(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.MeasurementAnalysisTest()
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root

    def identity(self, name, phase):
        binary=self.root/'binaries'/f'{name}.so';binary.parent.mkdir(exist_ok=True)
        plan={'phase':phase,'pg':'off'}
        for suffix, contents in (('',b'native-'+name.encode()),('.cu',b'// generated'),
                ('.plan.json',json.dumps(plan).encode()),('.ptxas.log',b'32 registers')):
            Path(str(binary)+suffix).write_bytes(contents)
        identity=dict(schema=1,trace=False,binary_path=str(binary),plan=plan,
            binary_sha256=validation.file_sha(binary),generated_source_sha256=validation.file_sha(str(binary)+'.cu'),
            manifest_sha256=validation.file_sha(str(binary)+'.plan.json'),resources_sha256=validation.file_sha(str(binary)+'.ptxas.log'))
        identity['artifact_id']=validation.json_digest(identity)
        Path(str(binary)+'.identity.json').write_text(json.dumps(identity))
        return binary,identity

    def native_llm_receipt(self):
        path=self.fixture.receipt('solo',6,0,1000)
        value=json.loads(path.read_text())
        canonical=self.root/'runs/r16/models/llama3_2_1b';canonical.mkdir(parents=True)
        shutil.copyfile(self.fixture.model/'config.json',canonical/'config.json')
        checkpoint=path.parents[2]/'checkpoint';checkpoint.mkdir()
        shutil.copyfile(canonical/'config.json',checkpoint/'config.json')
        manifest=dict(algorithm='tilemega-r16-name-seeded-uniform-v1',config_sha256=value['model_config_sha256'],
            seed=0,aliases={},tensors=[dict(name='embedding',shape=[32,16],dtype='BF16',sha256='f'*64)],state='closed')
        payload={key:manifest[key] for key in ('algorithm','config_sha256','seed','aliases','tensors')}
        manifest['checkpoint_sha256']=validation.json_digest(payload)
        (checkpoint/'random_checkpoint.json').write_text(json.dumps(manifest))
        config=dict(value['configuration'],checkpoint_manifest=str(checkpoint/'random_checkpoint.json'),
            checkpoint_sha256=manifest['checkpoint_sha256'],identities={},plan_manifests=[])
        for phase in ('prefill','decode'):
            binary,identity=self.identity(phase,phase)
            config[phase+'_binary']=str(binary);config[phase+'_sha256']=identity['binary_sha256']
            config['identities'][phase]=identity;config['plan_manifests'].append(identity['plan'])
        value.update(system='TM-seed',model=str(checkpoint),configuration=config)
        path.write_text(json.dumps(value))
        old=next(iter(self.fixture.state[6]));job=self.fixture.state[6].pop(old)
        self.fixture.state[6]['m1_tilemega_llama3_2_1b_B16_r0_gpu6']=job
        self.fixture.persist()
        return path

    def test_temporary_checkpoint_normalizes_model_and_keeps_native_arms_separate(self):
        self.native_llm_receipt()
        rows,rejected=summary.collect(self.root)
        self.assertFalse(rejected);self.assertEqual(len(rows),1)
        row=rows[0]
        self.assertEqual(row['model'],'llama3_2_1b')
        self.assertNotEqual(row['model'],'checkpoint')
        self.assertTrue(row['checkpoint_identity_shared'])
        self.assertTrue(row['source_checkpoint_sha256'])
        self.assertEqual(len(row['native_artifact_ids'].split(',')),2)
        self.assertEqual(len(summary.aggregate([row,dict(row,system='TM-H1')])),2)

    def test_native_binary_and_shared_checkpoint_tampering_are_rejected(self):
        path=self.native_llm_receipt()
        value=json.loads(path.read_text())
        binary=Path(value['configuration']['decode_binary']);binary.write_bytes(binary.read_bytes()+b'changed')
        rows,rejected=summary.collect(self.root)
        self.assertFalse(rows);self.assertEqual(len(rejected),1)
        binary.write_bytes(binary.read_bytes()[:-len(b'changed')])
        manifest=Path(value['configuration']['checkpoint_manifest'])
        contents=json.loads(manifest.read_text());contents['seed']+=1;manifest.write_text(json.dumps(contents))
        rows,rejected=summary.collect(self.root)
        self.assertFalse(rows);self.assertEqual(len(rejected),1)
        self.assertIn('shared checkpoint identity',rejected[0]['reason'])

    def test_r15_label_requires_pristine_code_and_the_original_compiler(self):
        compiler=self.root/'runs/r16/build-r15/tools/tilemega'
        compiler.parent.mkdir(parents=True);compiler.write_bytes(b'pristine fixture compiler')
        source=dict(head=validation.R15_HEAD,worktree_diff_sha256=hashlib.sha256(b'').hexdigest(),
            root=str(self.root/'runs/r16/tm-r15'),compiler=str(compiler),
            compiler_sha256=validation.file_sha(compiler))
        validation.pristine_r15({'source':source},self.root)
        for key,value in [('head','newer-head'),('worktree_diff_sha256','dirty'),
                          ('root',str(self.root)),('compiler_sha256','different-compiler')]:
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'pristine baseline'):
                validation.pristine_r15({'source':dict(source,**{key:value})},self.root)

    def test_rf1_off_label_requires_the_exact_single_function_patch(self):
        directory=self.root/'runs/r16/rf1/r15-no-l1-spread'
        compiler=directory/'build/tools/tilemega'
        compiler.parent.mkdir(parents=True);compiler.write_bytes(b'ablation fixture compiler')
        source=dict(head=validation.R15_HEAD,worktree_diff_sha256=validation.RF1_PATCH_SHA256,
            root=str(directory/'source'),compiler=str(compiler),compiler_sha256=validation.file_sha(compiler))
        validation.rf1_ablation({'source':source},self.root)
        for key,value in [('head','9fa448f3a'),('worktree_diff_sha256',hashlib.sha256(b'').hexdigest()),
                          ('root',str(self.root)),('compiler_sha256','different-compiler')]:
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'single-function R15 ablation'):
                validation.rf1_ablation({'source':dict(source,**{key:value})},self.root)

    def test_native_dnn_requires_matching_artifact_gate_and_input_without_export_field(self):
        directory=self.root/'runs/r16/scheduler/raw/dnn-native-test/attempt1'
        out=directory/'measurements';out.mkdir(parents=True)
        own=dict(ownership_verified=False,acceptance='provisional_container_pid_namespace')
        (directory/'guard_result.json').write_text(json.dumps(dict(code=0,**own)))
        samples=[dict(time=100+i,label=label,accepted=True,unresolved_host_pids=[99999999],**own,
            telemetry={'power.draw':'120','clocks.sm':'1800','clocks.mem':'2600','temperature.gpu':'40',
                       'clocks_throttle_reasons.active':'0x0'}) for i,label in enumerate(('before-measurement','after-measurement'))]
        (out/'guard.jsonl').write_text('\n'.join(map(json.dumps,samples))+'\n')
        binary,identity=self.identity('dnn','forward')
        inputs=directory/'inputs.safetensors';inputs.write_bytes(b'identified input fixture')
        gate=directory/'gate.json';gate.write_text(json.dumps(dict(passed=True,synthetic_weights=False,
            artifact_id=identity['artifact_id'],input_tensors_sha256=validation.file_sha(inputs))))
        execution=dict(artifact_id=identity['artifact_id'],trace=False,executor='L2')
        execution['execution_id']=validation.json_digest(execution)
        value=dict(evidence='verified',system='tilemega',model='resnet18',batch=1,precision='bf16',gpu_index=6,
            round_index=0,measurement=dict(warmup=50,iterations=200),median_ms=1,p90_ms=1,samples_ms=[1]*200,
            guard_log=str(out/'guard.jsonl'),artifact_id=identity['artifact_id'],execution_identity=execution,
            library_path=str(binary),identity=identity,input_path=str(inputs),input_sha256=validation.file_sha(inputs),
            numerical_gate_path=str(gate),numerical_gate_sha256=validation.file_sha(gate))
        path=out/'measurements.json';path.write_text(json.dumps(value))
        self.fixture.state[6]['dnn-native-resnet18-r0']=dict(status='done',exit_code=0,out=str(directory.relative_to(self.root)))
        self.fixture.persist()
        rows,rejected=summary.collect_dnn(self.root)
        self.assertEqual(len(rows),1);self.assertFalse(rejected)
        self.assertEqual(rows[0]['native_artifact_ids'],identity['artifact_id'])
        inputs.write_bytes(b'changed and longer fixture')
        rows,rejected=summary.collect_dnn(self.root)
        self.assertFalse(rows);self.assertEqual(len(rejected),1)


if __name__=='__main__':unittest.main()
