"""CPU-only CLI integration: compiler and GPU samples are explicit test doubles."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from tilemega.cli import Run,read_config
from tilemega.cache import record_outputs
from tilemega.serving.integrated_selection import SelectionBudgetExhausted

class BuildOrchestration(unittest.TestCase):
    def runner(self,root,fail_coverage=False):
        config=root/'config.json';config.write_text(json.dumps(dict(model=dict(path=str(root/'model')),
            device=dict(cache_dir=str(root/'cache')),output=dict(dir=str(root/'run')),
            workload=dict(batch=[1]),features=dict(pg='measure',decode_executor='measure',decode_loop='measure'))))
        (root/'model').mkdir()
        (root/'model/config.json').write_text(json.dumps({'model_type':'llama'}))
        r=Run.__new__(Run);r.config=read_config(config);r.model=root/'model';r.cache=root/'cache'
        r.out=root/'run';r.out.mkdir();r.binary='/compiler-test-double';r.version=dict(source_sha256='test-source')
        r.target=root/'target.json';r.target.write_text('{}');r.device_key='test-device';r.events=[]
        r.calibrate=lambda:{};r.preflight_gpu=lambda out:None;r.event=lambda *a,**kw:None
        r.export=lambda phase:(phase,root);r.calls=[]
        def artifact(lib,manifest):
            lib.parent.mkdir(parents=True,exist_ok=True);lib.write_bytes(str(lib).encode())
            Path(str(lib)+'.plan.json').write_text(json.dumps(manifest))
            for suffix in ('.identity.json','.source.json','.source.json.patch','.classes.tsv'):
                Path(str(lib)+suffix).write_text('{}')
            record_outputs(lib.parent/'record.json',[lib,Path(str(lib)+'.plan.json')])
        def variant(base,values,deadline):
            plan=json.loads(Path(str(base)+'.plan.json').read_text())
            plan.update(attention_kv_block=values['ec'],attention_impl=values['attention_impl'],
                        nonpaged_la=values['nonpaged_la'],paged_attention_load=values['paged_attention_load'])
            lib=Path(base).parent/('v'+str(values['ec'])+values['attention_impl']+str(values['nonpaged_la']))/'plan.so'
            artifact(lib,plan);return lib
        r.attention_variant=variant
        def command(argv,label,**kwargs):
            args=list(map(str,argv));r.calls.append((label,kwargs));self.assertIsNotNone(kwargs.get('deadline'))
            if args[1]=='compile':
                options=json.loads(Path(args[-1]).read_text());lib=Path(options[1])
                value=lambda name:options[options.index(name)+1]
                artifact(lib,dict(pg=value('--pg'),capacity=1088,attention_kv_block=256,
                    attention_impl='mma16',nonpaged_la=0,paged_attention_load='loader',gemms=[]))
                Path(value('--dump-cg')).write_text('test module')
            elif args[1]=='audit':(Path(args[-1])).write_text('{}')
            elif args[1]=='inspect':
                for name in args[-2:]:Path(name).write_text('{}')
            elif '-m' in args and args[args.index('-m')+1]=='tilemega.serving.measure_candidate':
                value=lambda name:args[args.index(name)+1]
                if fail_coverage:raise SelectionBudgetExhausted('test pilot deadline')
                lib=value('--so');plan=json.loads(Path(lib+'.plan.json').read_text())
                mode=value('--mode');loop=int(value('--loop'));out=Path(value('--out'));out.mkdir(parents=True)
                best=plan['pg']=='pages' and plan['attention_kv_block']==64 and plan['attention_impl']=='pvswap' and mode=='L1'
                identity=dict(execution_id=lib+mode+str(loop),trace=False,spill=False)
                values={str(p):dict(mean_ms=(1. if best else 2.)+p/10000,
                    decode_loop_used=bool(loop),execution_identity=identity) for p in (64,575,1000)}
                (out/'measurements.json').write_text(json.dumps(dict(modes={mode:dict(by_past=values)})))
            else:self.fail('unexpected command '+str(args))
        r.command=command;return r

    @patch('tilemega.cli.source_fingerprint',return_value='test-source')
    @patch('tilemega.build.identity.verify',return_value=dict(artifact_id='test-artifact'))
    def test_two_levels_write_confirmed_choice_without_modifying_manifest(self,*_):
        with tempfile.TemporaryDirectory() as tmp:
            r=self.runner(Path(tmp));result=r.build();choice=result['1']['decode_pg_choice']
            self.assertEqual((choice['selected']['pg'],choice['selected']['variant']['ec'],choice['selected']['mode']),('pages',64,'L1'))
            self.assertEqual(len(choice['selected']['samples_ms']),3)
            self.assertEqual(len(choice['candidates']),108)
            self.assertTrue(all(c['measurements'] for c in choice['candidates']))
            side=json.loads(Path(result['1']['decode']+'.serving.json').read_text())
            self.assertEqual(side['selection'],choice['selected'])
            self.assertEqual(len(side['candidates']),108)
            self.assertNotIn('selection',json.loads(Path(result['1']['decode']+'.plan.json').read_text()))
            self.assertTrue((r.out/'plans.json').exists())

    @patch('tilemega.cli.source_fingerprint',return_value='test-source')
    @patch('tilemega.build.identity.verify',return_value=dict(artifact_id='test-artifact'))
    def test_missing_dimension_coverage_cannot_publish_a_choice(self,*_):
        with tempfile.TemporaryDirectory() as tmp:
            r=self.runner(Path(tmp),fail_coverage=True)
            with self.assertRaises(SelectionBudgetExhausted):r.build()
            choice=json.loads((r.out/'decode-choice-B1.json').read_text())
            self.assertEqual(choice['status'],'failed_or_budget_exhausted')
            self.assertNotIn('selected',choice)
            self.assertFalse((r.out/'plans.json').exists())
            self.assertFalse(list(r.cache.rglob('*.serving.json')))

if __name__=='__main__':unittest.main()
