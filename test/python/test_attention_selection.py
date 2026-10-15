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
    def test_cluster_attention_domain_is_explicit_and_capacity_aligned(self):
        from tilemega.serving.attention_selection import cluster_merge_variants,cluster_merge_resources
        p=dict(capacity=1088,phase='decode',pg='off',attention_q_per_kv=3,attention_head_dim=128,
               target_spec=dict(caps=dict(wgmma=True,dsmem=True)))
        candidates=cluster_merge_variants(p)
        from tilemega.serving.attention_selection import label
        self.assertTrue(all(label(v).endswith('-cluster'+str(v['cluster_dim']))for v in candidates))
        self.assertEqual([(v['cluster_dim'],v['ec'])for v in candidates],[(2,576),(4,320)])
        self.assertFalse(any(v.get('merge')=='cluster' for v in variants(p,include_hopper=True)))
        for v in candidates:
            self.assertTrue(matches(dict(p,attention_impl='wgmma',attention_kv_block=v['ec'],
                attention_page='uniform',nonpaged_la=True,merge='cluster',cluster_dim=v['cluster_dim']),v))
            self.assertFalse(matches(dict(p,attention_impl='wgmma',attention_kv_block=v['ec'],
                attention_page='uniform',nonpaged_la=True,merge='stage',cluster_dim=v['cluster_dim']),v))
        for invalid in (dict(pg='pages'),dict(phase='prefill'),dict(arch_paths='sm80'),dict(attention_q_per_kv=9)):
            self.assertEqual(cluster_merge_variants(dict(p,**invalid)),[])
        cost=cluster_merge_resources(128,8,4,72704)
        self.assertEqual(cost['mailbox_bytes'],25600)
        self.assertEqual(cost['peer_bytes_per_episode'],12384)
        self.assertIsNone(cost['latency_ns'])

    def test_dimensions(self):
        p=dict(capacity=1088,pg='pages',attention_kv_block=256,attention_q_per_kv=4,attention_head_dim=64)
        self.assertEqual(len(variants(p)),24)
        self.assertTrue(any(matches(p,v) for v in variants(p)))
        p['pg']='l2';self.assertEqual(len(variants(p)),48)
        p['attention_q_per_kv']=8
        self.assertFalse(any(v['attention_impl']=='simt' for v in variants(p)))
    def test_hopper_is_explicit_and_pins_page_layout(self):
        p=dict(capacity=1088,pg='pages',attention_kv_block=256,
               attention_q_per_kv=8,attention_head_dim=128,
               target_spec=dict(caps=dict(wgmma=True)))
        self.assertFalse(any(v['attention_impl']=='wgmma' for v in variants(p)))
        h=[v for v in variants(p,include_hopper=True) if v['attention_impl']=='wgmma']
        self.assertTrue(h)
        self.assertTrue(all(v['page_bytes']==32768 for v in h))
        v=h[0]
        p.update(attention_impl='wgmma',attention_kv_block=v['ec'],pages=dict(page_bytes=16384))
        self.assertFalse(matches(p,v))
        p['pages']['page_bytes']=32768
        self.assertTrue(matches(p,v))
        p['arch_paths']='sm80'
        self.assertFalse(any(v['attention_impl']=='wgmma' for v in variants(p,include_hopper=True)))
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
