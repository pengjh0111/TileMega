"""Relocate two solved phase plans into one named request arena, without rescheduling."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import struct
from tilemega.bench.as2_models import RUNS, controlled

TABLES=('metadata.bin','stages.bin','tasks.bin','waits.bin','offsets.bin','queue.bin','allocations.bin')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def rebind_inputs(inputs,prefill,out):
    inputs,prefill,out=(controlled(p,[RUNS]) for p in (inputs,prefill,out))
    if out.exists():raise FileExistsError(out)
    identity=json.loads((inputs/'inputs.json').read_text())
    original=controlled(Path(identity['plan_directory']),[RUNS])
    if identity['seq']!=64 or identity['past'] or digest(original/'metadata.bin')!=identity['metadata_sha256']:
        raise ValueError('Reusable inputs must identify their actual original prefill Plan')
    old_meta=struct.unpack('<12I3Q',(original/'metadata.bin').read_bytes())
    new_meta=struct.unpack('<12I3Q',(prefill/'metadata.bin').read_bytes())
    # Geometry, pricing and task counts may change; input meaning and layout may not.
    if old_meta[8:11]!=new_meta[8:11] or new_meta[8:11]!=(identity['batch'],64,1088):
        raise ValueError('Rebinding cannot change batch, sequence or KV capacity')
    for plan in (original,prefill):
        imported=json.loads((plan.parent/'import.json').read_text())
        if (imported['phase']!='prefill' or imported['model_id']!=identity['model_id'] or
                imported['config_sha256']!=identity['config_sha256']):
            raise ValueError('Rebinding cannot change model/config/phase identity')
    def required(plan):
        buffers=json.loads((plan/'buffers.json').read_text())['buffers']
        if len({b['name'] for b in buffers})!=len(buffers):raise ValueError('Ambiguous input allocation names')
        binary=list(struct.iter_unpack('<3Q2I',(plan/'allocations.bin').read_bytes()))
        if len(binary)!=len(buffers) or any(b['id']!=i or
                (b['offset_bytes'],b['bytes'],b['allocated_bytes'],int(b.get('required_input',False)),0)!=a
                for i,(b,a) in enumerate(zip(buffers,binary))):
            raise ValueError('Input allocation manifest differs from binary tables')
        return {b['name']:b for b in buffers if b.get('required_input')}
    old,new=required(original),required(prefill)
    records={r['name']:r for r in identity['buffers']}
    if len(records)!=len(identity['buffers']) or old.keys()!=new.keys() or records.keys()!=old.keys():
        raise ValueError('Rebinding cannot add, omit or duplicate required inputs')
    files=[]
    for name,b in new.items():
        if any(old[name][k]!=b[k] for k in ('bytes','dtype','pack_json','external_name')):
            raise ValueError('Rebinding requires identical logical input layout: '+name)
        r=records[name];source=controlled(inputs/r['file'],[RUNS])
        if source.parent!=inputs or r['bytes']!=b['bytes'] or source.stat().st_size!=b['bytes'] or digest(source)!=r['sha256']:
            raise ValueError('Reusable input payload changed: '+name)
        files.append((source,dict(r,buffer=b['id'],file=str(b['id'])+'.bin')))
    out.mkdir()
    for source,r in files:os.link(source,out/r['file'])
    result=dict(identity,metadata_sha256=digest(prefill/'metadata.bin'),plan_directory=str(prefill),
        buffers=[r for _,r in files],rebound_from=dict(inputs=str(inputs),inputs_sha256=digest(inputs/'inputs.json'),
        plan=str(original),metadata_sha256=identity['metadata_sha256']),
        rebinding='verified_named_layout_and_payload_hashes; immutable hardlinks only')
    (out/'inputs.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='rebound_not_executed',inputs=str(out),buffers=len(files),payload_bytes_copied=0)))


def cached_kv_state(stages,buffers,metadata):
    state=[];ordinary=False;used=set();by_offset={b['offset_bytes']:b for b in buffers}
    for s in stages:
        if s[17]!=13:continue
        if s[31] not in (9,12,13):ordinary=True;continue
        heads=(s[24]==128 and s[25] in (2,3,4)) or (s[24]==64 and s[25]==4)
        if metadata[8] not in (1,16) or metadata[10]!=1088 or s[23]!=8 or s[26]!=1088 or not heads:
            raise ValueError('Unregistered derived KV shape')
        names=[];elements=metadata[8]*s[23]*metadata[10]*s[24]
        for i in (1,2,10,11):
            b=by_offset.get(s[i])
            if (b is None or b.get('required_input') or not b['name'].startswith('kv_cache.') or
                    b['dtype']!=('bf16' if i<10 else 'f32') or b['bytes']!=elements*(2 if i<10 else 4) or
                    b['name'] in used):
                raise ValueError('Derived KV planes must be unique full-sized request-reset state')
            names.append(b['name']);used.add(b['name'])
        if names[2:]!=['kv_cache.fp32.'+n for n in names[:2]]:
            raise ValueError('Derived KV is not attached to its logical BF16 state')
        state.append(tuple(names))
    if state and ordinary:raise ValueError('Derived KV cannot be partially enabled in a phase')
    return sorted(state)


from tilemega.bench.as2_precision import REPLAY, STREAMING, CONTRACTS, validate_plan_precision

SOURCE_CAPABILITIES=('TILEMEGA_LLM_SPLIT_K','TILEMEGA_LLM_LOCAL_TAIL_K7','TILEMEGA_LLM_LOCAL_Q_TAIL_K3',
                     'TILEMEGA_LLM_WIDE_QK','TILEMEGA_LLM_FP32_KV_CACHE','TILEMEGA_LLM_SWIGLU_TILE',
                     'TILEMEGA_LLM_SWIGLU_L1_K256','TILEMEGA_LLM_VECTOR_ARGMAX','TILEMEGA_LLM_G14',
                     'TILEMEGA_LLM_STREAMING_ATTENTION','TILEMEGA_LLM_GEMM_SESSION','TILEMEGA_LLM_REG_SWIGLU','TILEMEGA_LLM_FP32_KV_CACHE_B1','TILEMEGA_LLM_VALID_PV','TILEMEGA_LLM_LOCAL_RESIDUAL','TILEMEGA_LLM_LOCAL_RESIDUAL_NO_RAW','TILEMEGA_LLM_QKV_DMA_ACQUIRE','TILEMEGA_LLM_QK_PHASE_DRAIN','TILEMEGA_LLM_BATCHED_RESIDUAL')
SOURCE_GEOMETRY=('TILEMEGA_LLM_VARIANT','TILEMEGA_LLM_TM','TILEMEGA_LLM_TN','TILEMEGA_LLM_TK')
SOURCE_CONTRACT=('TILEMEGA_LLM_PRECISION_CONTRACT',)
PRECISION_CONTRACT=REPLAY


def gemm_variant_geometry(stage):
    if stage[31]==12:
        if ((stage[18],stage[19],stage[20],stage[21])!=(1,8,16,1024) or
                stage[22] not in (2048,3072) or stage[23]>1 or
                any(stage[i]==(1<<64)-1 for i in (0,1,2,3,16))):
            raise ValueError('Unregistered batched BF16 residual shape')
        return 1
    if stage[31]==11:
        if ((stage[18],stage[19],stage[20],stage[21],stage[22])!=(0,32,1,4096,1024) or
                stage[23]>1 or any(stage[i]==(1<<64)-1 for i in (0,1,3))):
            raise ValueError('Unregistered DMA-only QKV shape')
        return 1
    if stage[31] in (9,10):
        if (stage[18],stage[19],stage[20],stage[21])!=(1,8,1,1024) or stage[22] not in (2048,3072) or stage[23]>1 or stage[16]==(1<<64)-1:
            raise ValueError('Unregistered local BF16 residual shape')
        return 1
    if stage[31]==8:
        if stage[18]!=2:raise ValueError('Register SwiGLU requires its paired gate/up output')
        base=list(stage);base[31]=4
        return gemm_variant_geometry(base)
    if stage[31]==7:
        if stage[23]>1 or stage[18]>3 or stage[20] not in (1,64):
            raise ValueError('Unregistered small-M GEMM session')
        base=list(stage);base[31]=4 if stage[18]==2 else 6 if stage[18]==3 else 1
        return gemm_variant_geometry(base)
    if stage[31]==6 and (stage[18]!=3 or not 1<=stage[20]<=16 or stage[21] not in (128256,151936) or
            stage[19]!=(stage[21]+127)//128 or stage[23]>1 or stage[16]==(1<<64)-1):
        raise ValueError('Unregistered vector LM-head argmax')
    if stage[31] in (4,5) and (stage[18]!=2 or stage[20] not in (1,16,64,1024) or
            stage[31]==5 and stage[20]!=16 or
            (stage[21],stage[22]) not in {(6144,1024),(12288,2048),(16384,2048),(16384,3072),(28672,4096)} or
            stage[19]!=((stage[20]+15)//16)*(stage[21]//128) or
            stage[23]>1 or stage[16]==(1<<64)-1):
        raise ValueError('Unregistered strided SwiGLU shape')
    return 1 if stage[31] in (2,3,4,5,6) else stage[31]


def unmaterialized_raw_offsets(stages):
    result=set()
    for i,s in enumerate(stages):
        if s[17]!=0 or s[31]!=10:continue
        gemm_variant_geometry(s)
        if any(s[16] in other[:16] or j!=i and other[16]==s[16] for j,other in enumerate(stages)):
            raise ValueError('Unmaterialized residual raw output has a consumer or aliases another workspace')
        result.add(s[16])
    return result


def source_parts(path):
    path=controlled(path,[RUNS]);text=path.read_text()
    marker='#ifndef TILEMEGA_LLM_VARIANT' if '#ifndef TILEMEGA_LLM_VARIANT' in text else '#include'
    header,body=text.split(marker,1)
    fields={k:int(v) for k,v in re.findall(r'^#define (TILEMEGA_LLM_\w+) (\d+)$',header,re.M)}
    if set(fields)-set(SOURCE_CAPABILITIES+SOURCE_GEOMETRY+SOURCE_CONTRACT) or any(k not in fields for k in SOURCE_GEOMETRY):
        raise ValueError('Unknown generated request capability/geometry')
    fields.setdefault(SOURCE_CONTRACT[0],1)
    if fields[SOURCE_CONTRACT[0]] not in CONTRACTS.values():raise ValueError('Unsupported generated precision contract')
    if bool(fields.get('TILEMEGA_LLM_STREAMING_ATTENTION',0))!=(fields[SOURCE_CONTRACT[0]]>=2):
        raise ValueError('Generated streaming capability and precision contract differ')
    if any(fields.get(k,0) not in (0,1) for k in SOURCE_CAPABILITIES):
        raise ValueError('Unregistered generated request capability')
    if tuple(fields[k] for k in SOURCE_GEOMETRY) not in ((0,32,64,64),(1,16,128,128)):
        raise ValueError('Unregistered generated request geometry')
    return path,fields,marker+body


def compatible_source(required,compiled):
    a,b=source_parts(required),source_parts(compiled)
    if a[2]!=b[2] or any(a[1][k]!=b[1][k] for k in SOURCE_GEOMETRY+SOURCE_CONTRACT):
        raise ValueError('Compiled implementation catalogue changed runtime or geometry')
    if any(a[1].get(k,0)>b[1].get(k,0) for k in SOURCE_CAPABILITIES):
        raise ValueError('Compiled implementation catalogue lacks a required capability')
    return dict(status='compatible',required_source=str(a[0]),compiled_source=str(b[0]),
        required_sha256=digest(a[0]),compiled_sha256=digest(b[0]),
        normalized_runtime_sha256=hashlib.sha256(a[2].encode()).hexdigest(),
        required_capabilities=a[1],compiled_capabilities=b[1])


def request_source(prefill,decode,*additional):
    capabilities=SOURCE_CAPABILITIES;geometry=SOURCE_GEOMETRY+SOURCE_CONTRACT
    if len(additional)>6:raise ValueError('Bounded generated-source catalogue required')
    sources=[]
    for plan in (prefill,decode,*additional):
        sources.append(source_parts(controlled(plan,[RUNS]).parent/'persistent_llm.cpp'))
    if any(sources[0][2]!=s[2] or any(sources[0][1][k]!=s[1][k] for k in geometry) for s in sources[1:]):
        raise ValueError('Request phases require identical generated runtime and geometry')
    fields={k:max(s[1].get(k,0) for s in sources) for k in capabilities}
    fields.update({k:sources[0][1][k] for k in geometry})
    text='// Generated request capability union; phase queues remain unchanged.\n'
    text+=''.join(f'#define {k} {v}\n' for k,v in fields.items())+sources[0][2]
    return text,dict(sources=[dict(path=str(p),sha256=digest(p)) for p,_,_ in sources],capabilities=fields)


def bind(prefill, decode, inputs, out,emit_source=False):
    prefill,decode,inputs,out=(controlled(p,[RUNS]) for p in (prefill,decode,inputs,out))
    if out.exists():raise FileExistsError(out)
    generated_source_text,source_identity=request_source(prefill,decode) if emit_source else (None,None)
    identity=json.loads((inputs/'inputs.json').read_text())
    plans=[];precisions=[]
    for phase,path,seq in [('prefill',prefill,64),('decode',decode,1)]:
        imported=json.loads((path.parent/'import.json').read_text())
        m=list(struct.unpack('<12I3Q',(path/'metadata.bin').read_bytes()))
        if m[0] not in (2,3,4) or m[9]!=seq or m[10]!=1088 or m[8]!=identity['batch']:
            raise ValueError('Incompatible phase ABI')
        if m[11] not in (0,1):raise ValueError('Unknown phase pricing mode')
        if imported['phase']!=phase or imported['model_id']!=identity['model_id'] or imported['config_sha256']!=identity['config_sha256']:
            raise ValueError('Phase model/config mismatch')
        precision=imported.get('precision_contract_id',PRECISION_CONTRACT)
        phase_stages=list(struct.iter_unpack('<17Q16I',(path/'stages.bin').read_bytes()))
        validate_plan_precision(precision,phase_stages)
        if precisions and precision!=precisions[0]:raise ValueError('Request phases use different precision contracts')
        precisions.append(precision)
        if not m[11]:
            bounds=struct.unpack('<4I',(path/'pricing_bounds.bin').read_bytes())
            prices=json.loads((path/'prices.json').read_text())
            if m[0]>=4 and prices.get('runtime_abi')!=m[0]:
                raise ValueError('Empirical prices use a different runtime ABI')
            if bounds[0]!=1 or bounds[1]>bounds[2] or bounds[2]>1086 or bounds[3]!=m[1] or (seq==64 and (bounds[1] or bounds[2])) or not imported.get('priced_solver_executed'):
                raise ValueError('Invalid empirical pricing bounds/provenance')
            if prices.get('precision_contract_id',REPLAY)!=precision:
                raise ValueError('Empirical prices use another precision contract')
            for key,value in dict(model_id=identity['model_id'],config_sha256=identity['config_sha256'],batch=m[8],seq=seq,groups=m[1],phase=phase,past_min=bounds[1],past_max=bounds[2]).items():
                if prices[key]!=value:raise ValueError('Empirical pricing identity mismatch: '+key)
        buffers=json.loads((path/'buffers.json').read_text())['buffers']
        allocations=list(struct.iter_unpack('<3Q2I',(path/'allocations.bin').read_bytes()))
        if len(buffers)!=len(allocations) or len({b['name'] for b in buffers})!=len(buffers):
            raise ValueError('Ambiguous allocation identity')
        for i,(b,a) in enumerate(zip(buffers,allocations)):
            if b['id']!=i or (b['offset_bytes'],b['bytes'],b['allocated_bytes'],int(b.get('required_input',False)),0)!=a:
                raise ValueError('Allocation manifest does not match binary tables')
        plans.append(dict(phase=phase,path=path,metadata=m,buffers=buffers))
    if source_identity and source_identity['capabilities'][SOURCE_CONTRACT[0]]!=CONTRACTS[precisions[0]]:
        raise ValueError('Generated source and Plan precision contract differ')
    variants=[]
    for plan in plans:
        v={gemm_variant_geometry(s) for s in struct.iter_unpack('<17Q16I',(plan['path']/'stages.bin').read_bytes()) if s[17]==0}
        if len(v)!=1 or not v.issubset({0,1}):raise ValueError('One request binary requires a uniform registered geometry')
        variants.append(next(iter(v)))
    if variants[0]!=variants[1]:raise ValueError('Prefill/decode use different compiled GEMM geometries')
    cached_states=[cached_kv_state(list(struct.iter_unpack('<17Q16I',(p['path']/'stages.bin').read_bytes())),
        p['buffers'],p['metadata']) for p in plans]
    if cached_states[0]!=cached_states[1]:
        raise ValueError('Derived KV requires matching prefill/decode producers and state')
    if source_identity and source_identity['capabilities']['TILEMEGA_LLM_VARIANT']!=variants[0]:
        raise ValueError('Generated source geometry differs from actual phase tables')
    if identity['seq']!=64 or identity['past'] or digest(prefill/'metadata.bin')!=identity['metadata_sha256']:
        raise ValueError('Request must start with the actual prefill inputs')
    if plans[0]['metadata'][:3]!=plans[1]['metadata'][:3]:
        raise ValueError('Request phases require the same validated worker topology')
    priced=all(not p['metadata'][11] for p in plans)
    if any(not p['metadata'][11] for p in plans) and not priced:
        raise ValueError('Both phases must declare the same pricing mode')
    union=[];named={}
    for plan in plans:
        for b in plan['buffers']:
            if b['name'] in named:
                u=named[b['name']]
                for key in ('dtype','pack_json','external_name'):
                    if b[key]!=u[key]:raise ValueError('Incompatible shared buffer: '+b['name'])
                if bool(b.get('required_input'))!=bool(u.get('required_input')):
                    raise ValueError('Required input identity changed between phases')
                if (b.get('required_input') or b['name'].startswith('kv_cache.')) and b['bytes']!=u['bytes']:
                    raise ValueError('Weights, tokens and KV must have identical phase extents')
                u['bytes']=max(u['bytes'],b['bytes'])
            else:
                u=dict(b);u['id']=len(union);union.append(u);named[u['name']]=u
    total=0
    for u in union:
        u['offset_bytes']=total;u['allocated_bytes']=((u['bytes']+1023)//512)*512
        total+=u['allocated_bytes']
        if total>64<<30:raise OverflowError('Request arena exceeds checked device budget')
    # Only addresses change. Task identities, semantic edges, waits and FIFO
    # order remain byte-identical to the two actual CTG/Solver outputs.
    out.mkdir();(out/'inputs').mkdir()
    receipt=dict(schema='tilemega.as2.request_binding.v1',**{k:identity[k] for k in
        ('model_id','batch','config_sha256','weight_manifest_sha256','input_sha256')},
        isl=64,capacity=1088,arena_bytes=total,source_inputs=str(inputs),geometry_variant=variants[0],
        groups=plans[0]['metadata'][1],workers=plans[0]['metadata'][2],
        optimization_status='measured_native_rank_eft_shared_gm' if priced else 'unpriced_phase_bootstrap_not_admitted',phases={},buffers=union)
    receipt['precision_contract_id']=precisions[0]
    receipt['derived_fp32_kv_state']=cached_states[0]
    for plan in plans:
        dest=out/plan['phase'];dest.mkdir();m=plan['metadata'];m[12]=total
        remap={b['offset_bytes']:named[b['name']]['offset_bytes'] for b in plan['buffers']}
        remap[(1<<64)-1]=(1<<64)-1
        source=(plan['path']/'stages.bin').read_bytes();stages=[]
        for raw in struct.iter_unpack('<17Q16I',source):
            stage=list(raw)
            for i in range(17):
                if stage[i] not in remap:raise ValueError('Operand is not a named allocation base')
                stage[i]=remap[stage[i]]
            stages.append(struct.pack('<17Q16I',*stage))
        unmaterialized=unmaterialized_raw_offsets(list(struct.iter_unpack('<17Q16I',b''.join(stages))))
        (dest/'metadata.bin').write_bytes(struct.pack('<12I3Q',*m))
        (dest/'stages.bin').write_bytes(b''.join(stages))
        (dest/'allocations.bin').write_bytes(b''.join(struct.pack('<3Q2I',b['offset_bytes'],b['bytes'],b['allocated_bytes'],int(b.get('required_input',False)),0) for b in union))
        # Relocation enlarges shared storage; diagnostics retain the phase's
        # logical extent so inactive prefill tails are never interpreted as decode.
        phase_bytes={b['name']:b['bytes'] for b in plan['buffers']}
        (dest/'active_bytes.bin').write_bytes(b''.join(struct.pack('<Q',phase_bytes.get(b['name'],0)) for b in union))
        for name in ('tasks.bin','waits.bin','offsets.bin','queue.bin'):
            os.link(plan['path']/name,dest/name)
        for name in ('implementations.json',):
            os.link(plan['path']/name,dest/name)
        if priced:
            for name in ('pricing_bounds.bin','prices.json'):os.link(plan['path']/name,dest/name)
        description=json.loads((plan['path']/'buffers.json').read_text())
        description.update(arena_bytes=total,buffers=union,relocated_request_binding=True)
        (dest/'buffers.json').write_text(json.dumps(description,indent=2)+'\n')
        receipt['phases'][plan['phase']]=dict(source_plan=str(plan['path']),
            source_hashes={name:digest(plan['path']/name) for name in TABLES+(('pricing_bounds.bin','prices.json') if priced else ())},
            bound_hashes={name:digest(dest/name) for name in TABLES+(('pricing_bounds.bin','prices.json') if priced else ())})
        receipt['phases'][plan['phase']]['bound_hashes']['active_bytes.bin']=digest(dest/'active_bytes.bin')
        receipt['phases'][plan['phase']]['unmaterialized_private_buffer_ids']=[b['id'] for b in union if b['offset_bytes'] in unmaterialized]
        for name in ('implementations.json',):
            receipt['phases'][plan['phase']]['source_hashes'][name]=digest(plan['path']/name)
            receipt['phases'][plan['phase']]['bound_hashes'][name]=digest(dest/name)
    files={r['name']:r for r in identity['buffers']};receipt['input_buffers']=[]
    for b in union:
        if not b.get('required_input'):continue
        r=files.get(b['name'])
        if r is None or r['bytes']!=b['bytes']:raise ValueError('Missing shared input')
        source=inputs/r['file']
        if source.stat().st_size!=r['bytes'] or digest(source)!=r['sha256']:raise ValueError('Packed input identity changed')
        os.link(source,out/'inputs'/(str(b['id'])+'.bin'))
        receipt['input_buffers'].append(dict(buffer=b['id'],name=b['name'],file=str(b['id'])+'.bin',bytes=r['bytes'],sha256=r['sha256']))
    receipt['tokens_buffer']=named['serving.tokens']['id']
    receipt['kv_buffers']=[b['id'] for b in union if b['name'].startswith('kv_cache.')]
    if generated_source_text is not None:
        (out/'persistent_llm.cpp').write_text(generated_source_text)
        receipt['generated_source']=dict(source_identity,sha256=digest(out/'persistent_llm.cpp'))
    (out/'reset.bin').write_bytes(struct.pack('<2I',receipt['tokens_buffer'],len(receipt['kv_buffers']))+
        struct.pack('<'+'I'*len(receipt['kv_buffers']),*receipt['kv_buffers']))
    (out/'request.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('buffers','phases','input_buffers')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prefill','inputs','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--decode',type=Path)
    p.add_argument('--rebind-inputs',action='store_true',help='Validate and hardlink unchanged input layouts to a new prefill Plan')
    p.add_argument('--emit-source',action='store_true',help='Union registered capabilities of identical generated phase runtimes')
    a=p.parse_args()
    if a.rebind_inputs:
        if a.decode or a.emit_source:p.error('Input rebinding accepts only prefill, inputs and out')
        rebind_inputs(a.inputs,a.prefill,a.out)
    else:
        if not a.decode:p.error('Request binding requires --decode')
        bind(a.prefill,a.decode,a.inputs,a.out,a.emit_source)
