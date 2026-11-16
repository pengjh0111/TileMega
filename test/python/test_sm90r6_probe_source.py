"""Probe operands must follow each materialized geometry's own layout."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from sm90r6_probe_source import compose,template,geometry_source,array


def fixture(alternative=False):
    c=dict(model='dense',expert_fuse=False,attention=dict(chunks=1),residency=1)
    buffers=['  {"tokens",1,2,0ull,1088ull,""},',
             '  {"weight16",1,0,1024ull,0ull,"recipe16"},',
             '  {"hidden",0,0,0ull,64ull,""},']
    if alternative:buffers=[buffers[0],buffers[2],'  {"weight32",1,0,1024ull,0ull,"recipe32"},']
    activation=1 if alternative else 2;weight=2 if alternative else 1
    tn,tk=(32,128) if alternative else (16,64)
    rows=['  {"kEmbedding","",0ull,1,0,0,0,0},']+['  {"kGemm","weight",1024ull,4,0,0,0,0},']*11
    source='''#include <tilemega/Codegen/tasks/Sm90DenseProgram.cuh>
constexpr std::array<tm_buffer_info,3> kNativeBuffers={{
'''+ '\n'.join(buffers)+'''
}};
constexpr std::array<NativeTensorMap,1> kNativeTensorMaps={{
'''+f'  {{{activation}u,3u,{{64ull,1ull,1ull,0ull}},{{128ull,128ull,0ull}},{{64u,8u,{tk//64}u,0u}}}},'+'''
}};
constexpr std::array<std::array<unsigned,1>,2> kNativeLayerSlots={{
'''+f'  {{{weight}u}},\n  {{{weight}u}},'+'''
}};
using Body_0=Embedding<1,64,16,1088,0,1,2>;
'''+f'using Body_1=Gemm<1,64,64,{tn},{tk},2,1,4,backend::ServingEpilogueOp::kStore,16,true,kLayerSlot|0u,kAbsent,{activation},kAbsent,1,0>;\n'+'''
using Body_2=AttentionDummy;
using Body_3=OutputDummy;
using Body_4=GateDummy;
using Body_5=DownDummy;
using Body_6=HeadDummy;
using NativeProgram=Program<1,1088,1,3,1,1024,2,1,false,false,Stage_0,LayerGroup<1,Stage_1>,Stage_6,1>;
constexpr std::array<StageInfo,12> kNativeStages={{
'''+ '\n'.join(rows)+'''
}};
'''
    return source,c


class ProbeSource(unittest.TestCase):
    def test_bare_packed_weights_get_distinct_layout_bindings(self):
        import json,re
        source,c=fixture()
        recipe=dict(kind='tile_pages',tile_n=16,tile_k=64,
                    source=dict(kind='gate_up_interleave',u=16))
        source=source.replace('"weight16",1,0,1024ull,0ull,"recipe16"',
            '"expert.gate_up.packed",1,0,1024ull,0ull,'+json.dumps(json.dumps(recipe)))
        variants=[]
        for tn in (32,64):
            geometry=dict(tile_n=tn,slot_bytes=2*tn*64,slots=2,warps_n=2,warps_k=2,
                          register_stages=1,multicast=False)
            leaf=geometry_source(source,c,'qkv',geometry,16)
            self.assertIn(f'expert.gate_up.packed@t{tn}x64',leaf)
            variants.append((geometry,leaf,c))
        combined,_=compose(source,c,variants,'qkv')
        names={}
        for row in array(combined,'kNativeBuffers')[1]:
            name=json.loads(re.match(r'\s*\{("(?:\\.|[^"\\])*")',row)[1])
            self.assertTrue(name not in names or names[name]==row)
            names[name]=row

    def test_layout_and_map_remapping(self):
        source,c=fixture();other,d=fixture(True)
        text,ledger=compose(source,c,[({'tile_n':16},source,c),({'tile_n':32},other,d)],'qkv')
        self.assertIn('ProbeBody1=Gemm<1,64,64,32,128,2,1,4,backend::ServingEpilogueOp::kStore,16,true,3,kAbsent,2,kAbsent,1,1>',text)
        self.assertIn('std::array<tm_buffer_info,4>',text)
        self.assertIn('std::array<NativeTensorMap,2>',text)
        self.assertEqual([v['stages'] for v in ledger['variants']],[[1,2],[4,5]])
        self.assertEqual(ledger['timed_stages'],4)
        self.assertIn('ProbeInstance<NativeProgram,Body_0,3,false>',text)

    def test_equal_layouts_share_storage(self):
        source,c=fixture()
        text,ledger=compose(source,c,[({},source,c),({},source,c)],'qkv')
        self.assertIn('std::array<tm_buffer_info,3>',text)
        self.assertIn('std::array<NativeTensorMap,1>',text)
        self.assertEqual(ledger['stages'],6)

    def test_missing_family_is_rejected(self):
        source,c=fixture()
        with self.assertRaises(ValueError):compose(source,c,[({},source,c)],'expert_gate_up')

    def test_nested_target_is_rejected(self):
        with self.assertRaises(ValueError):template('Head<Gemm<1,2>,3>')

    def test_head_keeps_targets_unrolled_and_reuses_only_prefix_counters(self):
        source,c=fixture()
        import re
        gemm=re.search(r'using Body_1=(.*);',source)[1]
        source=source.replace('using Body_6=HeadDummy;',
            'using HeadGemm_6='+gemm+';\nusing Body_6=Head<HeadGemm_6,0,1,2,64,1088,true,1,0,false>;')
        text,ledger=compose(source,c,[({},source,c)],'lm_head')
        self.assertEqual(ledger['variants'][0]['stages'],[11,12])
        self.assertIn('for(int layer=0;layer<2;++layer)',text)
        self.assertIn('events,iteration*2+layer,trace,layer)',text)
        self.assertIn('ProbeInstance<NativeProgram,ProbeHead0,11,true>',text)
        self.assertIn('ProbeInstance<NativeProgram,ProbeHead0,12,true>',text)
        self.assertNotIn('ProbeInstance<NativeProgram,Body_1,6,false>',text)


if __name__=='__main__':unittest.main()
