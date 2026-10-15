"""Native receipts require binary provenance, shared weights and a passed input gate."""
import hashlib
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

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
        separate = dict(row, round=1, pid=row['pid']+1, native_artifact_ids='different-binary')
        self.assertEqual(len(summary.aggregate([row, separate])), 2)
        with self.assertRaisesRegex(ValueError, 'distinct measured process'):
            summary.aggregate([row, dict(row, round=1)])

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

    def test_named_cluster_and_paged_arms_use_all_native_checks(self):
        path=self.native_llm_receipt()
        value=json.loads(path.read_text())
        for label in ('TM-H3-stage','TM-H3-cluster2','TM-paged-h1h4','TM-H4d-cluster4'):
            value['system']=label
            path.write_text(json.dumps(value))
            rows,rejected=summary.collect(self.root)
            self.assertFalse(rejected)
            self.assertTrue(rows[0]['checkpoint_identity_shared'])
        value['system']='TM-unregistered-arm'
        path.write_text(json.dumps(value))
        rows,rejected=summary.collect(self.root)
        self.assertFalse(rows)
        self.assertTrue(rejected)

    def test_consistently_hashed_manifest_cannot_mislabel_actual_dispatch(self):
        binary, identity=self.identity('dispatch','decode')
        identity['plan']['gemms']=[dict(index=0,split_k=1,tile_m=16,tile_n=128,
            tile_k=64,stages=5,impl='wgmma')]
        identity['defines']=dict(TILEMEGA_GEMM_VARIANT_COUNT='1',TILEMEGA_GEMM_IMPL_0='2',
            TILEMEGA_GEMM_TILE_M='16',TILEMEGA_GEMM_TILE_N='128',
            TILEMEGA_GEMM_TILE_K='64',TILEMEGA_GEMM_STAGES='5')
        Path(str(binary)+'.cu').write_text('constexpr GemmRuntimeDesc kRuntimeGemms0[] = {\n{0,1,16,128,64,5},\n};\n')
        def persist():
            Path(str(binary)+'.plan.json').write_text(json.dumps(identity['plan']))
            identity['generated_source_sha256']=validation.file_sha(str(binary)+'.cu')
            identity['manifest_sha256']=validation.file_sha(str(binary)+'.plan.json')
            identity['artifact_id']=validation.json_digest({k:v for k,v in identity.items() if k!='artifact_id'})
            Path(str(binary)+'.identity.json').write_text(json.dumps(identity))
        persist()
        self.assertEqual(validation.native_identity(binary)['plan']['gemms'][0]['impl'], 'wgmma')
        identity['plan']['gemms'][0]['impl']='mma16'
        persist()
        with self.assertRaisesRegex(ValueError, 'actual GEMM implementation'):
            validation.native_identity(binary)

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

    def test_repaired_ir_label_requires_both_pristine_backends_and_prefill_provenance(self):
        import r16_r15_ir_provenance as ir
        value=json.loads(self.native_llm_receipt().read_text())
        value['system']='TM-R15-IR'
        identities=value['configuration']['identities']
        for identity in identities.values():
            identity['source']={}
        def actual_identity(binary, expected):
            return expected
        with patch.object(validation,'native_identity',side_effect=actual_identity), \
                patch.object(validation,'pristine_r15') as pristine, \
                patch.object(ir,'validate_frontend_ir',side_effect=ValueError('missing IR origin')) as origin:
            with self.assertRaisesRegex(ValueError,'missing IR origin'):
                validation.native_llm(value,self.root)
            origin.side_effect=None
            pristine.reset_mock();origin.reset_mock()
            result=validation.native_llm(value,self.root)
            self.assertTrue(result['checkpoint_identity_shared'])
            self.assertEqual(pristine.call_count,2)
            origin.assert_called_once_with(identities['prefill'],self.root)
            identities['decode']['source']['frontend_ir_provenance']={'label':'TM-R15-IR'}
            with self.assertRaisesRegex(ValueError,'decode must retain'):
                validation.native_llm(value,self.root)

    def test_repaired_ir_cannot_be_silently_labelled_pristine_r15(self):
        value=json.loads(self.native_llm_receipt().read_text())
        identities=value['configuration']['identities']
        for identity in identities.values():
            identity['source']={}
        identities['prefill']['source']['frontend_ir_provenance']={'label':'TM-R15-IR'}
        with patch.object(validation,'native_identity',side_effect=lambda binary, expected:expected), \
                patch.object(validation,'pristine_r15'):
            for label in ('TM-R15','TM-RF1-on'):
                value['system']=label
                with self.subTest(label=label), self.assertRaisesRegex(ValueError,'explicit TM-R15-IR label'):
                    validation.native_llm(value,self.root)

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
        export=directory/'export';(export/'checkpoint').mkdir(parents=True)
        checkpoint=export/'checkpoint/model.safetensors';checkpoint.write_bytes(b'bound source checkpoint')
        checkpoint_sha=validation.file_sha(checkpoint)
        receipt=json.loads(gate.read_text());receipt['source_checkpoint_sha256']=checkpoint_sha
        gate.write_text(json.dumps(receipt))
        value.update(source_checkpoint_sha256=checkpoint_sha,export_path=str(export),
                     numerical_gate_sha256=validation.file_sha(gate));path.write_text(json.dumps(value))
        rows,rejected=summary.collect_dnn(self.root)
        self.assertEqual(len(rows),1);self.assertFalse(rejected)
        checkpoint.write_bytes(b'other checkpoint after numerical check')
        rows,rejected=summary.collect_dnn(self.root)
        self.assertFalse(rows);self.assertIn('source checkpoint',rejected[0]['reason'])
        checkpoint.write_bytes(b'bound source checkpoint')
        inputs.write_bytes(b'changed and longer fixture')
        rows,rejected=summary.collect_dnn(self.root)
        self.assertFalse(rows);self.assertEqual(len(rejected),1)


if __name__=='__main__':unittest.main()
