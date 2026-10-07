import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from phase_d_r14 import family_candidate
from tilemega.cli import plan_build_choices,read_config,compiler_features

class FamilyGate(unittest.TestCase):
    def test_resident2_family_keeps_ordinary_plan_and_prefill(self):
        settings=dict(resident2_batches=[16])
        self.assertEqual(plan_build_choices(('l2','pages'),settings,16,'decode'),[
            ('l2','l2',{}),('l2_resident2','l2',dict(attention_buffers=1)),('pages','pages',{})])
        self.assertEqual(plan_build_choices(('l2',),settings,16,'prefill'),[('l2','l2',{})])
        self.assertEqual(plan_build_choices(('l2','pages'),settings,1,'decode'),[('l2','l2',{}),('pages','pages',{})])
    def test_r14_configs_are_valid_and_conditional_flags_default_off(self):
        from make_phase0 import ROOT
        for model in ('llama','qwen3'):
            config=read_config(ROOT/f'configs/e2e/{model}_r14.json')
            self.assertEqual(config['features']['attention_buffers'],2)
            self.assertEqual(config['features']['attention_frontier'],0)
            self.assertEqual(config['features']['ep_direct'],0)
            options=compiler_features(config['features'])
            self.assertNotIn('decode_executor',options)
            self.assertNotIn('decode_loop',options)
            self.assertNotIn('resident2_batches',options)
    def rows(self,d,geometry=32):
        result=[]
        for i,times in enumerate(((1.,3.,4.),(4.,2.,1.))):
            library=str(d/f'{i}.so')
            plan=dict(pg='pages',gemms=[dict(tile_n=32 if i==0 else geometry)],kappa=1,residency=1,attention_kv_block=(64,256)[i],attention_impl='mma16')
            Path(library+'.plan.json').write_text(json.dumps(plan))
            result.append(dict(library=library,mode='L1',loop=0,measurements=[dict(by_past={str(p):dict(mean_ms=t) for p,t in zip((64,575,1000),times)})]))
        return result
    @patch('phase_d_r14.verify',return_value=dict(trace=False,implementations=dict(gemms='same_packing')))
    def test_different_past_winners_need_compatible_geometry(self,_):
        with tempfile.TemporaryDirectory() as tmp:
            rows=self.rows(Path(tmp))
            result=family_candidate(rows)
            self.assertTrue(result['different_past_winner'])
            self.assertGreater(result['family_integral_gain'],.02)
            self.assertEqual(family_candidate(self.rows(Path(tmp),128))['family_integral_gain'],0.)
    @patch('phase_d_r14.verify',return_value=dict(trace=False,implementations=dict(gemms='same_packing')))
    def test_missing_failed_measurements_are_not_winners(self,_):
        self.assertEqual(family_candidate([])['family_integral_gain'],0.)
        with tempfile.TemporaryDirectory() as tmp:
            rows=self.rows(Path(tmp));rows[1]['error']='failed'
            self.assertEqual(family_candidate(rows)['family_integral_gain'],0.)
    @patch('phase_d_r14.verify',return_value=dict(trace=True,implementations=dict(gemms='same_packing')))
    def test_trace_timing_rejected(self,_):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):family_candidate(self.rows(Path(tmp)))
if __name__=='__main__':unittest.main()
