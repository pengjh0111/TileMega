import json,tempfile,unittest
from pathlib import Path
from tilemega.serving.attention_selection import variants,matches,pinned_geometry,compile_options
class AttentionSelection(unittest.TestCase):
    def test_conditional_flags_only_apply_to_selected_decode_batch(self):
        from tilemega.cli import batch_features,read_config
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.json'
            data=dict(model=dict(path='/tmp/model'),features_by_batch={'16':{'attention_buffers':1,'ep_direct':1}})
            path.write_text(json.dumps(data));config=read_config(path)
            self.assertEqual(batch_features(config,16,'decode')['attention_buffers'],1)
            self.assertEqual(batch_features(config,1,'decode')['attention_buffers'],2)
            self.assertEqual(batch_features(config,16,'prefill')['attention_buffers'],2)
            data['features_by_batch']['16']['pg']='pages';path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):read_config(path)
    def test_budget_preserves_measured_finalists(self):
        from tilemega.serving.integrated_selection import successive_halving,SelectionBudgetExhausted
        def measure(candidate,round):
            i=candidate['id']
            if round.startswith('pilot') and i>=3:raise SelectionBudgetExhausted()
            return dict(by_past={str(p):dict(mean_ms=1+i) for p in (64,575,1000)},execution_identity={'artifact_id':i})
        winner,rows=successive_halving([dict(id=i) for i in range(8)],measure)
        self.assertEqual(winner['id'],0)
        self.assertEqual(len(winner['samples_ms']),3)
        self.assertEqual(sum(r['eliminated_round']=='budget_unmeasured' for r in rows),5)
    def test_dimensions(self):
        p=dict(capacity=1088,pg='pages',attention_kv_block=256)
        self.assertEqual(len(variants(p)),12)
        self.assertTrue(any(matches(p,v) for v in variants(p)))
        p['pg']='l2';self.assertEqual(len(variants(p)),24)
    def test_winner_geometry_and_family_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);classes=d/'classes.tsv'
            classes.write_text('class\tgemm\ttile_n\n0\t0\t128\n')
            manifest=dict(kappa=1,residency=1,gemms=[dict(index=0,tile_m=16,tile_n=8,tile_k=128,stages=2,split_k=1,impl='gemv')])
            shapes=pinned_geometry(manifest,classes,d)
            self.assertEqual((shapes[0]['tile_n'],shapes[0]['impl']),(8,'gemv'))
            options=compile_options(['bridge','old.so','--serve-kv-block','128','--pg','l2','--serving-warm-start','old.json'],dict(ec=32,attention_impl='pvswap',nonpaged_la=1,paged_attention_load='loader'),d)
            self.assertNotIn('--serving-warm-start',options)
            self.assertEqual(options[options.index('--serve-kv-block')+1],'32')
            self.assertEqual(options[options.index('--pg')+1],'l2')
            self.assertEqual(json.loads((d/'cases.json').read_text())['cases'][0]['geometries'],shapes)
    def test_reject_partial_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'classes.tsv';f.write_text('class\tgemm\n1\t0\n')
            with self.assertRaises(ValueError):pinned_geometry(dict(gemms=[dict(index=0)]),f,tmp)
if __name__=='__main__':unittest.main()
