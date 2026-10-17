"""CPU contracts for explicit full-model MoE implementation comparisons."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_r16_moe_candidates import assert_implementation

class MoECandidateContract(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        path=Path(self.directory.name)/'layout.json'
        recipes=[{'kind':'tile_pages','tile_n':64,'tile_k':64,'source':{'tensor':i}} for i in range(241)]
        path.write_text(json.dumps({'buffers':[{'recipe':r} for r in recipes]}))
        source={'root':'/frozen/source','head':'frozen-head','compiler_sha256':'compiler-hash'}
        self.c={'arm':'mma','mode':'L2','loop':0,'gemm_count':241,'layout':str(path),'phase_sources':{'decode':source}}
        self.i={'source':source.copy(),'arch':'sm_90','nvcc_arch':'sm_90a','defines':{'TILEMEGA_ARCH_ID':900},'trace':False,
            'caps':{'wgmma':True,'tma':True,'bulk_copy':True},
            'plan':{'target_spec':{'arch_tag':'sm_90'},'seed_arch':'native','seed_target_arch':'sm_90','arch_paths':'auto','pg':'off',
                'attention_noinline':True,'moe_activation_bulk':0,'dm_wgmma_inline':0,'buffers':[{'recipe':r} for r in recipes],
                'gemms':[{'index':i,'impl':'mma16','tile_m':16,'tile_n':64,'tile_k':64} for i in range(241)]},
            'implementations':{'dm_wgmma_inline':False,'gemms':[{'index':i,'impl':'mma16'} for i in range(241)],
                'attention':{'impl':'mma16','page_layout':'chunk16'}}}
    def test_mma_legal(self):
        assert_implementation(self.i,self.c,'decode')
    def wgmma(self):
        self.c['arm']='h1'
        for rows in (self.i['plan']['gemms'],self.i['implementations']['gemms']):
            for row in rows:row['impl']='wgmma'
        self.i['plan'].update(dm_wgmma_inline=1,moe_activation_bulk=1)
        self.i['implementations']['dm_wgmma_inline']=True
        self.i['defines'].update(TILEMEGA_DM_NARROW_INLINE=1,TILEMEGA_DM_WGMMA_INLINE=1)
    def test_wgmma_legal(self):
        self.wgmma();assert_implementation(self.i,self.c,'decode')
    def rejected(self,mutate):
        mutate()
        with self.assertRaises(ValueError):assert_implementation(self.i,self.c,'decode')
    def test_explicit_mma_tn128(self):
        self.c['packing_tile_n']=128
        for g in self.i['plan']['gemms']:g['tile_n']=128
        for b in self.i['plan']['buffers']:b['recipe']['tile_n']=128
        path=Path(self.c['layout']);layout=json.loads(path.read_text())
        for b in layout['buffers']:b['recipe']['tile_n']=128
        path.write_text(json.dumps(layout))
        assert_implementation(self.i,self.c,'decode')
    def test_runtime_manifest_tampering_rejected(self):
        from run_r16_moe_candidates import runtime_contract
        self.c['runtime_source']={'root':'/wrong'}
        self.c['runtime_contract_sha256']='bad'
        with self.assertRaises((ValueError,OSError)):runtime_contract(self.c)
    def test_loop_executor_legal(self):
        self.c.update(mode='L1',loop=1)
        assert_implementation(self.i,self.c,'decode')
    def test_host_L1_not_in_bounded_executor_domain(self):
        self.rejected(lambda:self.c.update(mode='L1',loop=0))
    def test_loop_requires_frozen_numeric_reference(self):
        from run_r16_moe_candidates import validate_loop_reference
        self.c.update(mode='L1',loop=1)
        with self.assertRaises(ValueError):validate_loop_reference(self.c,{})
    def test_loop_launch_count(self):
        from run_r16_moe_candidates import validate_launch_counts
        validate_launch_counts(1024,1,True,0,1,set())
        validate_launch_counts(1,1,False,0,0,set())
        with self.assertRaises(ValueError):validate_launch_counts(1024,1,True,1023,1,{1})
        with self.assertRaises(ValueError):validate_launch_counts(1024,1,True,0,0,set())
    def test_host_launch_count(self):
        from run_r16_moe_candidates import validate_launch_counts
        validate_launch_counts(1024,2,False,1023,0,{2})
        with self.assertRaises(ValueError):validate_launch_counts(1024,2,False,1022,0,{2})
        with self.assertRaises(ValueError):validate_launch_counts(1024,2,False,1023,0,{1})
    def test_no_relabel(self):
        self.rejected(lambda:self.c.update(arm='h1'))
    def test_no_missing_expert(self):
        self.rejected(lambda:self.i['plan']['gemms'].pop())
    def test_no_wrong_packing(self):
        self.rejected(lambda:self.i['plan']['gemms'][0].update(tile_n=128))
    def test_no_other_source(self):
        self.rejected(lambda:self.i['source'].update(head='other'))
    def test_no_sm80_fallback(self):
        self.rejected(lambda:self.i['defines'].update(TILEMEGA_ARCH_PATH_SM80=1))
    def test_mma_rejects_wgmma_only_bulk(self):
        self.rejected(lambda:self.i['plan'].update(moe_activation_bulk=1))
    def test_wgmma_requires_typed_inline(self):
        self.wgmma();self.rejected(lambda:self.i['plan'].update(dm_wgmma_inline=0))
    def test_no_changed_attention(self):
        self.rejected(lambda:self.i['implementations']['attention'].update(impl='wgmma'))
    def test_no_cluster_relabel(self):
        self.rejected(lambda:self.i['plan'].update(cluster_dim=2))

if __name__=='__main__':unittest.main()
