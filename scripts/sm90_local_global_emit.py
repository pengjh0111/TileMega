#!/usr/bin/env python3
"""Model-specific emitter for actual static task queues and native ABI recipes."""
import argparse
import json
from pathlib import Path
import re
import subprocess
from sm90_local_global_local import ROOT, CONFIG, SHAPES, sha, write

TEMPLATE_LOCAL=Path('/root/TileMega-explicit-ae/runs/sm90_explicit_ae/build-v1/A2.so.cu')
TEMPLATE_TM=Path('/root/TileMega-m/runs/sm90r3/solve/FINAL/qwen3_1_7b-B1/decode.so.cu')

def metadata(s):
    local=s['structure']=='S_local';template=TEMPLATE_LOCAL if local else TEMPLATE_TM
    text=template.read_text();marker='} // namespace tilemega::codegen::native_dense' if local else 'using Body_0='
    prefix=text[:text.index(marker)];c=s['coordinates'];gs=c['gemms'];att=c['attention']
    prefix=re.sub(r'constexpr char kNativeCoordinateIdentity\[\]=.*?;\n',
        lambda _: 'constexpr char kNativeCoordinateIdentity[]='+json.dumps(json.dumps(c,sort_keys=True))+';\n',prefix,count=1)
    family_ids={}
    slots=re.search(r'kNativeLayerSlots=\{\{(.*?)\}\};',prefix,re.S).group(1)
    for row in re.findall(r'\{([^{}]+)\}',slots):
        ids=[int(x.strip().rstrip('u')) for x in row.split(',')]
        for i,name in ((0,'qkv'),(5,'o'),(6,'gate_up'),(7,'down')):family_ids[ids[i]]=name
    family_ids[352]='lm_head'
    def replace_buffers(match):
        lines=match.group(1).splitlines();result=[];index=0
        for line in lines:
            if not line.strip():continue
            m=re.fullmatch(r'  \{"(.*?)",(\d+),(\d+),(\d+)ull,(\d+)ull,"(.*)"\},',line)
            if not m:raise ValueError('unrecognized native buffer descriptor: '+line)
            name,role,dtype,constant,batch,recipe=m.groups();recipe=json.loads('"'+recipe+'"')
            if index in family_ids:
                g=gs[family_ids[index]];tn=g['tile_n'];tk=g['slot_bytes']//(2*tn)
                r=json.loads(recipe);assert r['kind']=='tile_pages';r.update(tile_n=tn,tile_k=tk)
                recipe=json.dumps(r,separators=(',',':'));name=re.sub(r'@t\d+x\d+',f'@t{tn}x{tk}',name)
            if index==12:batch=str(2048*att['chunks'])
            if index==13:batch=str(16*att['chunks'])
            if index in (237,238):batch=str(151936//gs['lm_head']['tile_n'])
            result.append(f'  {{{json.dumps(name)},{role},{dtype},{constant}ull,{batch}ull,{json.dumps(recipe)}}},')
            index+=1
        assert index==(355 if local else 353)
        return 'kNativeBuffers={{\n'+'\n'.join(result)+'\n}};'
    prefix=re.sub(r'kNativeBuffers=\{\{(.*?)\}\};',replace_buffers,prefix,flags=re.S)
    maps=[]
    for family,buffer,k in [('qkv',4,2048),('o',11,2048),('down',17,6144),('lm_head',4,2048)]:
        g=gs[family];tk=g['slot_bytes']//(2*g['tile_n'])
        maps.append(f'  {{{buffer}u,3u,{{64ull,1ull,{k//64}ull,0ull}},{{{2*k}ull,128ull,0ull}},{{64u,8u,{tk//64}u,0u}}}},')
    kv=[f'  {{{i}u,4u,{{128ull,1088ull,8ull,28ull}},{{256ull,278528ull,4456448ull}},{{64u,{att["block_keys"]}u,1u,1u}}}},' for i in (7,8)]
    maps=[maps[0],*kv,*maps[1:]]
    if local:
        g=gs['gate_up'];tk=g['slot_bytes']//(2*g['tile_n'])
        maps.append(f'  {{4u,3u,{{64ull,1ull,32ull,0ull}},{{4096ull,128ull,0ull}},{{64u,8u,{tk//64}u,0u}}}},')
    prefix=re.sub(r'constexpr std::array<NativeTensorMap,\d+> kNativeTensorMaps=\{\{.*?\}\};',
        f'constexpr std::array<NativeTensorMap,{len(maps)}> kNativeTensorMaps={{{{\n'+'\n'.join(maps)+'\n}};',prefix,flags=re.S)
    return prefix+'} // namespace tilemega::codegen::native_dense\n',template

def aliases(s):
    local=s['structure']=='S_local';c=s['coordinates'];g=c['gemms'];a=c['attention'];lines=[]
    lines.append('using MapTasks=backend::Sm90DenseExplicitTasks;')
    lines.append('using Op_embedding=Embedding<1,2048,151936,1088,0,3,4,MapTasks>;')
    data=dict(qkv=('kLayerSlot|0u',353 if local else 6,6,'kAbsent',0,'kStore',True),
        o=('kLayerSlot|5u','kAbsent' if local else 4,354 if local else 4,'kAbsent',3,'kStore' if local else 'kResidual',False),
        gate_up=('kLayerSlot|6u',353 if local else 17,354 if local else 17,'kAbsent',6 if local else 0,'kStore' if local else 'kSwiGLU',True),
        down=('kLayerSlot|7u','kAbsent' if local else 4,354 if local else 4,'kAbsent',4,'kStore' if local else 'kResidual',False),
        lm_head=(352,353 if local else 237,237,238,5,'kArgmaxPartial',True))
    for name,(n,k) in SHAPES.items():
        p=g[name];w,res,out,index,tm,op,norm=data[name];tn=p['tile_n'];tk=p['slot_bytes']//(2*tn)
        args=[1,n,k,tn,tk,p['slots'],p['warps_n'],p['warps_k'],'backend::ServingEpilogueOp::'+op,
            8,str(norm).lower(),w,res,out,index,p['register_stages'],tm,'false',str(local and norm).lower(),'MapTasks']
        lines.append(f'using Op_{name}=Gemm<{",".join(map(str,args))}>;')
    args=[1,8,128,2,'true',a['chunks'],a['block_keys'],a['slots'],0,1088,6,'kLayerSlot|1u','kLayerSlot|2u',11,1,2,
          'kLayerSlot|3u','kLayerSlot|4u',12,13,14,1,2,'false']
    lines.append('using AttentionUnprepared=Attention<'+','.join(map(str,args))+',false,false,MapTasks>;')
    lines.append('using Op_attention=Attention<'+','.join(map(str,args))+','+
                 str(local and a['merge']=='stage').lower()+','+str(local).lower()+',MapTasks>;')
    if local:
        for name,family in [('rms_qkv','qkv'),('rms_gate','gate_up'),('rms_head','lm_head')]:
            lines.append(f'using Op_{name}=ExplicitRms<2048,{g[family]["warps_k"]},4,353,MapTasks>;')
        lines += ['using Op_prepare=ExplicitAttentionPrepare<AttentionUnprepared,2,128,MapTasks>;',
            'using Op_merge=ExplicitAttentionMerge<Op_attention,MapTasks>;',
            'using Op_residual_o=ExplicitOutput<false,2048,8,354,4,4,MapTasks>;',
            'using Op_residual_down=ExplicitOutput<false,2048,8,354,4,4,MapTasks>;',
            'using Op_swiglu=ExplicitOutput<true,6144,8,354,kAbsent,17,MapTasks>;']
    lines.append('using Op_head=Head<Op_lm_head,0,3,4,2048,1088,'+str(not local).lower()+'>;')
    return lines

def arrays(plan):
    kinds=list(dict.fromkeys(u['kind'] for u in plan['units']));ops={k:i for i,k in enumerate(kinds)}
    nodes=plan['tasks'];entries=[];waits=[];pubs=[];tracks=[];heads=[];head_ranges=[];unit_ranges={}
    def entry(n,wait=None,coordinate=None):
        w=n['wait_events'] if wait is None else wait;p=n['publish_events']
        row=[ops[n['kind']],n['layer'],n['coordinate'] if coordinate is None else coordinate,len(waits),len(w),len(pubs),len(p)]
        waits.extend(w);pubs.extend(p);entries.append(row)
    for track,q in enumerate(plan['per_track_sequence']):
        begin=len(entries);head=[];head_wait=[]
        for t in q:
            n=nodes[t]
            if n['kind']=='head':
                assert not n['publish_events'];head.append(n['coordinate']);head_wait.extend(n['wait_events']);continue
            unit=n['unit'];key=(unit,track)
            if key not in unit_ranges:unit_ranges[key]=[len(entries),0]
            unit_ranges[key][1]+=1;entry(n)
        head_ranges.append([len(heads),len(head)]);heads.extend(head)
        n=nodes[plan['units'][-1]['tasks'][0]]
        unit_ranges[(n['unit'],track)]=[len(entries),1]
        entry(dict(n,publish_events=[]),sorted(set(head_wait)),coordinate=-1)
        tracks.append([begin,len(entries)-begin])
    # A/E use contiguous per-unit portions of the identical track program.
    if plan['planner']['aggregate']:
        for (unit,track),(begin,count) in unit_ranges.items():
            expected=ops[plan['units'][unit]['kind']]
            assert all(e[0]==expected and e[1]==plan['units'][unit]['layer'] for e in entries[begin:begin+count])
    def array(typ,name,values):
        values=values or [0]
        vals=['{'+','.join(map(str,v))+'}' if isinstance(v,list) else str(v) for v in values]
        return f'__device__ const {typ} {name}[]={{\n'+',\n'.join(vals)+'\n};'
    result=[array('StaticTaskEntry','static_entries',entries),array('int','static_waits',waits),array('int','static_publications',pubs),
        array('unsigned','static_event_counts',[e['count'] for e in plan['events']]),array('StaticTaskRange','static_tracks',tracks),
        array('int','static_head_coordinates',heads),array('StaticTaskRange','static_head_ranges',head_ranges),
        array('StaticTaskRange','static_unit_ranges',[unit_ranges.get((u,t),[0,0]) for u in range(len(plan['units'])) for t in range(132)])]
    result.append(f'''struct ActualStaticPlan {{
  static constexpr unsigned kUnits={len(plan['units'])},kEvents={len(plan['events'])};
  __device__ static StaticTaskEntry const& Entry(int i) {{return static_entries[i];}}
  __device__ static int Wait(int i) {{return static_waits[i];}}
  __device__ static int Publish(int i) {{return static_publications[i];}}
  __device__ static unsigned EventCount(int i) {{return static_event_counts[i];}}
  __device__ static StaticTaskRange Track(int t) {{return static_tracks[t];}}
  __device__ static StaticTaskRange Unit(int u,int t) {{return static_unit_ranges[u*132+t];}}
  __device__ static StaticTaskRange HeadRange(int t) {{return static_head_ranges[t];}}
  __device__ static int const* HeadCoordinates() {{return static_head_coordinates;}}
}};''')
    result.append('using ActualOps=std::tuple<'+','.join('Op_'+k for k in kinds)+'>;')
    return result,dict(entries=len(entries),head_tasks=len(heads),compute_tasks=len(nodes),
        head_finish_calls=132,unique_coverage=sorted(heads)==list(range(len(heads))),kinds=kinds)

def emit(arm,plans,out):
    local=arm in 'ACE';structure=CONFIG/('S_local.json' if local else 'S_TM.json');s=json.loads(structure.read_text())
    assert s['coordinates']['emb']==('stage' if local else 'la')
    assert s['coordinates']['residency']==s['coordinates']['cluster']==1 and not s['coordinates']['weight_stream']
    plan_path=plans/(('A' if arm=='E' else arm)+'.plan.json');plan=json.loads(plan_path.read_text())
    assert plan['structure_sha256']==sha(structure)
    prefix,template=metadata(s);body=aliases(s);program,receipt=arrays(plan)
    lines=[prefix,'#include <tilemega/Codegen/tasks/Sm90DenseStaticPlan.cuh>',
        'namespace tilemega::codegen::native_dense {',*body,*program,
        f'using NativeProgram=StaticTaskProgram<{355 if local else 353},{7 if local else 6},28,8,ActualOps,ActualStaticPlan,{str(not local).lower()}>;',
        f'constexpr std::array<StageInfo,{len(plan["units"])}> kNativeStages={{{{']
    for u in plan['units']:lines.append(f'  {{"static","{u["kind"]}",0,{len(u["tasks"])},0,0,0,0}},')
    lines.extend(['}};','} // namespace tilemega::codegen::native_dense'])
    if arm=='E':lines.extend(['#define TILEMEGA_NATIVE_KERNEL_SEQUENCE 1',
        '#define TILEMEGA_NATIVE_SEQUENCE_PRESERVE_BODY 1',
        '#include <tilemega/Codegen/tasks/Sm90DenseStaticSequence.cuh>',
        'namespace tilemega::codegen::native_dense {using NativeSequence=StaticTaskSequence<NativeProgram>;}'])
    lines.append('#include <tilemega/Codegen/tasks/Sm90DenseRuntime.cuh>')
    path=out/(arm+'.so.cu')
    if path.exists():raise ValueError('do not overwrite generated experiment artifacts')
    path.write_text('\n'.join(lines)+'\n')
    receipt.update(arm=arm,structure_sha256=sha(structure),plan=str(plan_path),plan_sha256=sha(plan_path),
        metadata_template=str(template),metadata_template_sha256=sha(template),source_sha256=sha(path),
        historical_data_reused=False,execution='stage kernels + CUDA Graph' if arm=='E' else 'static task megakernel')
    write(out/(arm+'.emit.json'),receipt)
    write(CONFIG/(arm+'.json'),dict(arm=arm,structure=structure.name,structure_sha256=sha(structure),
        plan_source=str(plan_path),plan_sha256=sha(plan_path),execution=receipt['execution'],
        coordinates=s['coordinates'],body_contract=s['structure'],executor='Sm90DenseStaticPlan.cuh',
        coordinate_role='frozen body/source provenance; group_wait and layer_loop do not drive this executor',
        actual_policy_source='GPU task program: ownership, order, waits and publications in plan_source',
        actual_group_release=not plan['planner']['aggregate'],actual_weight_stream=False,
        actual_iteration='full static per-track program over all 28 layers'))
    print(arm,receipt['compute_tasks'],receipt['entries'],flush=True)

def build(arm,out):
    source=out/(arm+'.so.cu');binary=out/(arm+'.so')
    if binary.exists():raise ValueError('retain prior binary')
    command=['/usr/local/cuda/bin/nvcc','-rdc=true','-O3','-std=c++17','-gencode','arch=compute_90a,code=sm_90a',
        '--expt-relaxed-constexpr','--split-compile=4','-shared','-Xcompiler','-fPIC','-cudart','shared','-Xptxas','-v',
        '-I'+str(ROOT/'include'),'-I'+str(ROOT/'third_party/cutlass/include'),str(source),'-o',str(binary)]
    with (out/(arm+'.build.log')).open('w') as f:code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT).returncode
    write(out/(arm+'.build.json'),dict(command=command,code=code,source_sha256=sha(source),
        binary_sha256=sha(binary) if code==0 else None))
    if code:raise RuntimeError('compile failed: '+arm)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['emit','build'])
    p.add_argument('--arms',default='ABCDE');p.add_argument('--plans',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    for arm in args.arms:
        if args.action=='emit':emit(arm,args.plans.resolve(),out)
        else:build(arm,out)
