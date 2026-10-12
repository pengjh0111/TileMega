"""Cross-phase memory policy and CLI wiring; no compiler or GPU is invoked."""
import json
from pathlib import Path
import tempfile
from unittest import TestCase, main
from unittest.mock import patch

from tilemega.cli import Run, read_config
from tilemega.moe.deployment import automatic_layout_policy


class MoeDeploymentTest(TestCase):
    def test_large_dense_models_request_compatible_shared_packing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);workload=self.fixture(root,'llama')
            config=json.loads((root/'config.json').read_text())
            config.update(hidden_size=4096,num_hidden_layers=32,num_attention_heads=32,
                          num_key_value_heads=8,head_dim=128)
            (root/'config.json').write_text(json.dumps(config))
            (root/'model.safetensors.index.json').write_text(json.dumps(
                dict(metadata=dict(total_size=16_000_000_000))))
            result=automatic_layout_policy(root,dict(resources=dict(dram_capacity_bytes=48*1024**3)),workload)
            self.assertTrue(result['require_shared_layout'])
            self.assertEqual(result['second_layout_estimate_bytes'],16_000_000_000)
            self.assertIn('norm folding',result['scope'])

    def fixture(self, root, model_type='qwen3_moe'):
        config=dict(model_type=model_type,num_hidden_layers=2,hidden_size=64,
            num_experts=4,moe_intermediate_size=32,num_key_value_heads=1,
            num_attention_heads=4,head_dim=16)
        (root/'config.json').write_text(json.dumps(config))
        (root/'model.safetensors.index.json').write_text(json.dumps(
            dict(metadata=dict(total_size=120000))))
        return dict(batch=[1],prompt_len=64,max_new_tokens=8)

    def test_only_proven_low_capacity_reorders_moe(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); workload=self.fixture(root)
            self.assertFalse(automatic_layout_policy(root,{},workload)['require_shared_layout'])
            large=automatic_layout_policy(root,dict(resources=dict(dram_capacity_bytes=1000000)),workload)
            self.assertFalse(large['require_shared_layout'])
            bound=large['two_layout_lower_bound_bytes']
            for capacity,expected in [(bound,False),(bound-1,True)]:
                self.assertEqual(automatic_layout_policy(root,
                    dict(resources=dict(dram_capacity_bytes=capacity)),workload)['require_shared_layout'],expected)
            config=json.loads((root/'config.json').read_text())
            config['num_local_experts']=config.pop('num_experts')
            (root/'config.json').write_text(json.dumps(config))
            self.assertTrue(automatic_layout_policy(root,
                dict(resources=dict(dram_capacity_bytes=bound-1)),workload)['require_shared_layout'])
            config['num_experts']=config['num_local_experts']+1
            (root/'config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'aliases disagree'):
                automatic_layout_policy(root,dict(resources=dict(dram_capacity_bytes=1)),workload)
            self.fixture(root,'llama')
            self.assertFalse(automatic_layout_policy(root,
                dict(resources=dict(dram_capacity_bytes=1)),workload)['require_shared_layout'])

    def test_selected_decode_artifact_constrains_prefill_and_cache(self):
        for constrained in (False,True):
            with self.subTest(constrained=constrained), tempfile.TemporaryDirectory() as directory:
                root=Path(directory); workload=self.fixture(root)
                path=root/'run.json';path.write_text(json.dumps(dict(model=dict(path=str(root)),
                    workload=workload,features=dict(pg='measure',prefill_pg='measure'))))
                run=Run.__new__(Run);run.config=read_config(path);run.model=root
                run.out=root/'out';run.out.mkdir();run.cache=root/'cache'
                run.binary='/compiler';run.device_key='fixture';run.version=dict(source_sha256='source')
                run.target=root/'target.json'
                target=dict(resources=dict(dram_capacity_bytes=1 if constrained else 1000000))
                run.target.write_text(json.dumps(target));run.calibrate=lambda:target
                run.preflight_gpu=lambda _:None;run.event=lambda *args,**kwargs:None
                order=[];options=[]
                def export(phase):
                    order.append(phase);return phase,root
                run.export=export
                def command(argv,label,**kwargs):
                    if argv[1]!='compile':return
                    args=json.loads(Path(argv[-1]).read_text());options.append(args)
                    binary=Path(args[1]);binary.write_bytes(label.encode())
                    Path(str(binary)+'.plan.json').write_text(json.dumps(dict(
                        phase=args[args.index('--serving')+1],pg=args[args.index('--pg')+1],
                        capacity=72,attention_kv_block=64)))
                run.command=command
                def decode(built,batch,prefill_mode):
                    winner=dict(pg='pages',mode='L2',loop=0,library=str(built['pages']))
                    return winner,[winner]
                run.select_decode=decode
                run.select_prefill=lambda built,batch:('l2','L1')
                def select(candidates,*args,**kwargs):
                    winner=next(c for c in candidates if c['pg']=='pages')
                    winner=dict(winner,library=winner['base_library'])
                    return winner,[winner]
                with patch('tilemega.cli.source_fingerprint',return_value='source'), \
                        patch('tilemega.cli.valid_record',return_value=False), \
                        patch('tilemega.cli.successive_halving',side_effect=select), \
                        patch('tilemega.cli.record_outputs'):
                    result=run.build()
                self.assertEqual(order,['decode','prefill'] if constrained else ['prefill','decode'])
                self.assertEqual(result['1']['serving']['prefill_mode'],'L1')
                selected=Path(result['1']['decode']+'.plan.json')
                prefill=[args for args in options if args[args.index('--serving')+1]=='prefill']
                for args in prefill:
                    self.assertEqual('--shared-weight-layout' in args,constrained)
                    if constrained:
                        self.assertEqual(args[args.index('--shared-weight-layout')+1],str(selected))
                        self.assertEqual(args[args.index('--nonpaged-weight-layout')+1],'tiled')
                if constrained:
                    self.assertTrue((run.out/'layout-policy.json').is_file())

    def test_decode_selection_retains_three_round_cache_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=Run.__new__(Run);run.model=root;run.out=root
            run.config=dict(features=dict(decode_executor='L2',decode_loop=0),
                workload=dict(prompt_len=64,max_new_tokens=1024,batch=[1]),
                solver=dict(exclude_l1_loop=False,candidate_guard_wait_s=0))
            built={pg:root/(pg+'.so') for pg in ('l2','pages')}
            for binary in built.values():binary.write_bytes(binary.name.encode())
            calls=[]
            def command(argv,label,**kwargs):
                self.assertTrue(kwargs['gpu']);calls.append(label)
                options=dict(zip(argv[3::2],argv[4::2]))
                out=Path(options['--out']);out.mkdir(exist_ok=True)
                score=1 if options['--so'].endswith('pages.so') else 2
                (out/'measurements.json').write_text(json.dumps(dict(
                    modes={'L2':dict(by_past={str(p):dict(mean_ms=score,decode_loop_used=False)
                                              for p in (64,575,1000)})})))
            run.command=command
            winner,_=run.select_decode(built,1,None)
            self.assertEqual(winner['pg'],'pages');self.assertEqual(len(calls),6)
            calls.clear();run.select_decode(built,1,None);self.assertEqual(calls,[])
            built['pages'].write_bytes(b'changed')
            run.select_decode(built,1,None);self.assertEqual(len(calls),6)


if __name__=='__main__':main()
