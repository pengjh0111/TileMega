"""Native binding, full-model coverage and bounded E2E queue contracts."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('r16_hopper_queues',ROOT/'scripts/run_r16_hopper_candidates.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


class HopperCandidates(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)/'repo';self.source=self.root/'runs/r16/source/frozen'
        compile_source=self.source/'tools/commands/compile.cpp'
        compile_source.parent.mkdir(parents=True);compile_source.write_text('flag=="--seed-arch"')
        compiler=self.root/'compiler';compiler.write_text('immutable compiler fixture')
        driver.write(str(self.source)+'.json',{'cli':str(compiler),'cli_sha256':driver.file_sha(compiler),
                     'build_returncode':0,'snapshot_head':'abc'})
        for model in ('llama3_2_1b','qwen3_30b_a3b'):
            driver.write(self.root/'runs/r16/models'/model/'config.json',{'model_type':'qwen3_moe' if model.startswith('qwen') else 'llama',
                'num_hidden_layers':2,'hidden_size':128})
            for phase in ('prefill','decode'):
                driver.write(self.root/'runs/r16/exports'/model/phase/'export.json',{'phase':phase})
        self.args=SimpleNamespace(root=self.root,snapshot=self.source,out=self.root/'runs/r16/candidates/test',
            run_id='test',python=Path(sys.executable),models=['llama3_2_1b'],batches=[1,16],gpu=6,moe_gpu=3,
            pid_mode='strict',cpu_bind_only=False,seed=0)

    def test_eight_candidates_bind_native_caps_and_keep_shared_recipes(self):
        interpreter=self.root/'env/bin/python';interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(sys.executable);self.args.python=interpreter
        driver.prepare(self.args)
        plan=driver.load(self.args.out/'plan.json');jobs=driver.load(self.args.out/'gpu6/queue_prepare_and_pilot.json')
        self.assertFalse(plan['enqueued']);self.assertEqual(len(plan['cells']),2)
        for cell in plan['cells']:
            self.assertEqual(len(cell['candidates']),8)
            self.assertEqual({c['arm'] for c in cell['candidates']},set(driver.ARMS))
            self.assertEqual({(c['mode'],c['loop']) for c in cell['candidates']},{('L2',0),('L1',1)})
            self.assertEqual({c['seed'] for c in cell['candidates']},{0})
            self.assertEqual({c['python'] for c in cell['candidates']},{str(interpreter)})
            self.assertEqual(len({c['layout'] for c in cell['candidates']}),1)
        self.assertEqual(len({j['name'] for j in jobs}),len(jobs))
        for job in jobs:
            self.assertEqual(job['env']['CUDA_VISIBLE_DEVICES'],job['env']['TILEMEGA_DEVICE_INDEX'])
            self.assertEqual(job['env']['PYTHONPATH'],str(self.source/'python'))
            if 'compile' in job['command'] and 'bootstrap' not in job['name']:
                command=job['command'];self.assertEqual(command[command.index('--seed-arch')+1],'native')
                self.assertIn('--shared-weight-layout',command)
            if '_smoke_' in job['name']:
                self.assertFalse(job['timing']);self.assertGreaterEqual(job['needs_free_mib'],24000)
            if '_pilot_' in job['name']:
                self.assertTrue(job['timing']);self.assertIn('--membind=1',job['command'])

    def test_missing_model_does_not_stop_other_cells_and_moe_uses_numa0(self):
        self.args.models=['qwen3_1_7b','llama3_2_1b','qwen3_30b_a3b'];self.args.batches=[1]
        driver.prepare(self.args)
        plan=driver.load(self.args.out/'plan.json');self.assertEqual(len(plan['cells']),2)
        self.assertEqual(plan['rejected'][0]['model'],'qwen3_1_7b')
        jobs=driver.load(self.args.out/'gpu3/queue_prepare_and_pilot.json')
        self.assertTrue(all(j['env']['TILEMEGA_NUMA_NODE']=='0' for j in jobs))
        self.assertTrue(any('--moe-activation-bulk' in j['command'] for j in jobs))

    def test_fourth_snapshot_without_formal_binding_is_rejected(self):
        (self.source/'tools/commands/compile.cpp').write_text('old compiler')
        with self.assertRaisesRegex(ValueError,'predates'):driver.prepare(self.args)

    def test_optional_pipeline_and_qkv_coordinates_keep_control_arms(self):
        self.args.activation_tma=1;self.args.qkv_epilogue=1;self.args.batches=[4]
        source=self.source/'tools/commands/compile.cpp'
        with self.assertRaisesRegex(ValueError,'support'):driver.prepare(self.args)
        source.write_text(source.read_text()+' "--activation-tma" "--qkv-epilogue"')
        driver.prepare(self.args)
        candidates=driver.load(self.args.out/'plan.json')['cells'][0]['candidates']
        for candidate in candidates:
            self.assertEqual(candidate['system'],driver.ARMS[candidate['arm']])
            self.assertEqual(candidate['activation_tma'],int(candidate['arm'] in ('h1','h1h4')))
            self.assertEqual(candidate['qkv_epilogue'],int(candidate['arm']=='h1h4'))
            command=driver.compile_command(candidate,'decode','out.so')
            self.assertEqual('--activation-tma' in command,bool(candidate['activation_tma']))
            self.assertEqual('--qkv-epilogue' in command,bool(candidate['qkv_epilogue']))
            prefill=driver.compile_command(candidate,'prefill','out.so')
            self.assertNotIn('--activation-tma',prefill);self.assertNotIn('--qkv-epilogue',prefill)

    def test_identity_rejects_actual_arch800_missing_gemms_and_body_fallback(self):
        layout=self.root/'layout.json'
        packed=[{'kind':'tile_pages','tile_n':128,'tile_k':64,'source':{'kind':'alias','source':f'weight{i}'}} for i in range(65)]
        driver.write(layout,{'buffers':[{'recipe':recipe} for recipe in packed]})
        candidate=dict(arm='h1',snapshot_source=str(self.source),snapshot_head='abc',compiler_sha256='compiler',
                       gemm_count=65,layout=str(layout))
        gemms=[{'index':i,'impl':'wgmma','tile_n':128,'tile_k':64} for i in range(65)]
        identity=dict(source={'root':str(self.source),'head':'abc','compiler_sha256':'compiler'},arch='sm_90',nvcc_arch='sm_90a',
            caps={'wgmma':True,'tma':True,'bulk_copy':True},defines={'TILEMEGA_ARCH_ID':'900'},trace=False,
            plan={'gemms':copy.deepcopy(gemms),'arch_paths':'auto','pg':'off','shared_weight_layout_sha256':driver.file_sha(layout),
                  'target_spec':{'arch_tag':'sm_90'},
                  'seed_arch':'native','seed_target_arch':'sm_90','buffers':[{'recipe':recipe} for recipe in packed]},
            implementations={'gemms':gemms,'attention':{'impl':'mma16','page_layout':'chunk16'}})
        driver.assert_implementation(identity,candidate,'decode')
        legacy=copy.deepcopy(identity);legacy['arch']='sm_90a'
        driver.assert_implementation(legacy,candidate,'decode')
        dense=copy.deepcopy(identity);del dense['plan']['shared_weight_layout_sha256']
        driver.assert_implementation(dense,candidate,'decode')
        for coordinate,macro in (('activation_tma','TILEMEGA_ACTIVATION_TMA'),('qkv_epilogue','TILEMEGA_QKV_EPILOGUE')):
            enabled=copy.deepcopy(identity);enabled['plan'][coordinate]=1;enabled['defines'][macro]='1'
            requested=dict(candidate,**{coordinate:1})
            driver.assert_implementation(enabled,requested,'decode')
            enabled['defines'][macro]='0'
            with self.assertRaisesRegex(ValueError,coordinate):driver.assert_implementation(enabled,requested,'decode')
        for change in ('arch800','missing_gemm','body_fallback','cap_missing','sm80_macro','packed_name','packed_shape','missing_weight','gemm_shape','target_tag','generic90'):
            broken=copy.deepcopy(identity)
            if change=='arch800':broken['defines']={}
            elif change=='missing_gemm':broken['implementations']['gemms'].pop()
            elif change=='body_fallback':broken['implementations']['gemms'][17]['impl']='mma16'
            elif change=='cap_missing':broken['caps']['wgmma']=False
            elif change=='sm80_macro':broken['defines']['TILEMEGA_ARCH_PATH_SM80']='1'
            elif change=='packed_name':broken['plan']['buffers'][17]['recipe']['source']['source']='wrong'
            elif change=='packed_shape':broken['plan']['buffers'][17]['recipe']['tile_k']=128
            elif change=='missing_weight':broken['plan']['buffers'].pop()
            elif change=='target_tag':broken['plan']['target_spec']['arch_tag']='sm_89'
            elif change=='generic90':broken['nvcc_arch']='sm_90'
            else:broken['plan']['gemms'][17]['tile_k']=128
            with self.subTest(change=change),self.assertRaises(ValueError):
                driver.assert_implementation(broken,candidate,'decode')

    def test_finalists_are_top_three_full_e2e_and_rotate_fresh_rounds(self):
        driver.prepare(self.args);plan=driver.load(self.args.out/'plan.json')
        state={candidate['pilot_job']:{'status':'done'} for cell in plan['cells'] for candidate in cell['candidates']}
        scores={candidate['id']:i+1 for i,candidate in enumerate(plan['cells'][0]['candidates'])}
        def measure(root,state,name,candidate):
            return dict(candidate=candidate['id'],e2e_seconds=scores[candidate['id']],checkpoint_sha256='same',versions={'torch':'2.8'})
        with patch.object(driver,'states',return_value=state),patch.object(driver,'validate_smoke'),patch.object(driver,'measurement',side_effect=measure):
            driver.finalists(SimpleNamespace(plan=self.args.out/'plan.json'))
        stage=driver.load(self.args.out/'selection-stage.json');self.assertTrue(all(c['status']=='finalists_ready' for c in stage['cells']))
        jobs=driver.load(self.args.out/'gpu6/queue_finalists.json');self.assertEqual(len(jobs),18)
        first=[]
        for r in range(3):
            group=[j for j in jobs if f'_B1_final{r}_' in j['name']]
            self.assertEqual(len(group),3);first.append(min(group,key=lambda j:j['priority'])['name'].split(f'_final{r}_')[1])
        self.assertEqual(len(set(first)),3)

    def test_shared_layout_preserves_nested_named_expert_recipes(self):
        recipe={'kind':'expert_interleave','sources':['model.layers.0.experts.1.weight']}
        value=driver.shared_layout({'buffers':[{'dtype':'bf16','recipe':{'kind':'tile_pages','tile_n':32,'tile_k':128,'source':recipe}},
                                              {'dtype':'i32','recipe':{'kind':'alias','source':'route'}}]})
        self.assertEqual(len(value['buffers']),1)
        self.assertEqual(value['buffers'][0]['recipe']['source'],recipe)
        self.assertEqual(value['buffers'][0]['recipe']['tile_n'],128)
        self.assertEqual(value['buffers'][0]['recipe']['tile_k'],64)

    def test_actual_device_loop_cannot_silently_fall_back(self):
        engine=SimpleNamespace(decode_loop=True,decode_mode=1)
        value=driver.execution_row(engine,SimpleNamespace(decode_loop_used=True),1024)
        self.assertTrue(value['device_loop_used'])
        with self.assertRaisesRegex(ValueError,'fell back'):
            driver.execution_row(engine,SimpleNamespace(decode_loop_used=False),1024)
        self.assertFalse(driver.execution_row(engine,SimpleNamespace(decode_loop_used=False),1)['device_loop_used'])


if __name__=='__main__':unittest.main()
