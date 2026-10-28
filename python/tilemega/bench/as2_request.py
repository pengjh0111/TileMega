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


def cached_kv_state(stages,buffers,metadata):
    state=[];ordinary=False;used=set();by_offset={b['offset_bytes']:b for b in buffers}
    for s in stages:
        if s[17]!=13:continue
        if s[31]!=9:ordinary=True;continue
        if metadata[8]!=16 or metadata[10]!=1088 or (s[23],s[24],s[25],s[26])!=(8,128,2,1088):
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


def request_source(prefill,decode):
    capabilities=('TILEMEGA_LLM_SPLIT_K','TILEMEGA_LLM_LOCAL_TAIL_K7','TILEMEGA_LLM_LOCAL_Q_TAIL_K3',
                  'TILEMEGA_LLM_WIDE_QK','TILEMEGA_LLM_FP32_KV_CACHE')
    geometry=('TILEMEGA_LLM_VARIANT','TILEMEGA_LLM_TM','TILEMEGA_LLM_TN','TILEMEGA_LLM_TK')
    sources=[]
    for plan in (prefill,decode):
        path=controlled(plan,[RUNS]).parent/'persistent_llm.cpp';text=path.read_text()
        marker='#ifndef TILEMEGA_LLM_VARIANT' if '#ifndef TILEMEGA_LLM_VARIANT' in text else '#include'
        header,body=text.split(marker,1)
        fields={k:int(v) for k,v in re.findall(r'^#define (TILEMEGA_LLM_\w+) (\d+)$',header,re.M)}
        if set(fields)-set(capabilities+geometry) or any(k not in fields for k in geometry):
            raise ValueError('Unknown generated request capability/geometry')
        if any(fields.get(k,0) not in (0,1) for k in capabilities):
            raise ValueError('Unregistered generated request capability')
        if tuple(fields[k] for k in geometry) not in ((0,32,64,64),(1,16,128,128)):
            raise ValueError('Unregistered generated request geometry')
        sources.append((path,fields,marker+body))
    if sources[0][2]!=sources[1][2] or any(sources[0][1][k]!=sources[1][1][k] for k in geometry):
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
    plans=[]
    for phase,path,seq in [('prefill',prefill,64),('decode',decode,1)]:
        imported=json.loads((path.parent/'import.json').read_text())
        m=list(struct.unpack('<12I3Q',(path/'metadata.bin').read_bytes()))
        if m[0] not in (2,3,4) or m[9]!=seq or m[10]!=1088 or m[8]!=identity['batch']:
            raise ValueError('Incompatible phase ABI')
        if m[11] not in (0,1):raise ValueError('Unknown phase pricing mode')
        if imported['phase']!=phase or imported['model_id']!=identity['model_id'] or imported['config_sha256']!=identity['config_sha256']:
            raise ValueError('Phase model/config mismatch')
        if not m[11]:
            bounds=struct.unpack('<4I',(path/'pricing_bounds.bin').read_bytes())
            prices=json.loads((path/'prices.json').read_text())
            if m[0]>=4 and prices.get('runtime_abi')!=m[0]:
                raise ValueError('Empirical prices use a different runtime ABI')
            if bounds[0]!=1 or bounds[1]!=0 or bounds[2]>1086 or bounds[3]!=m[1] or (seq==64 and bounds[2]) or not imported.get('priced_solver_executed'):
                raise ValueError('Invalid empirical pricing bounds/provenance')
            for key,value in dict(model_id=identity['model_id'],config_sha256=identity['config_sha256'],batch=m[8],seq=seq,groups=m[1],phase=phase,past_min=0,past_max=bounds[2]).items():
                if prices[key]!=value:raise ValueError('Empirical pricing identity mismatch: '+key)
        buffers=json.loads((path/'buffers.json').read_text())['buffers']
        allocations=list(struct.iter_unpack('<3Q2I',(path/'allocations.bin').read_bytes()))
        if len(buffers)!=len(allocations) or len({b['name'] for b in buffers})!=len(buffers):
            raise ValueError('Ambiguous allocation identity')
        for i,(b,a) in enumerate(zip(buffers,allocations)):
            if b['id']!=i or (b['offset_bytes'],b['bytes'],b['allocated_bytes'],int(b.get('required_input',False)),0)!=a:
                raise ValueError('Allocation manifest does not match binary tables')
        plans.append(dict(phase=phase,path=path,metadata=m,buffers=buffers))
    variants=[]
    for plan in plans:
        v={1 if s[31] in (2,3) else s[31] for s in struct.iter_unpack('<17Q16I',(plan['path']/'stages.bin').read_bytes()) if s[17]==0}
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
    for name in ('prefill','decode','inputs','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--emit-source',action='store_true',help='Union registered capabilities of identical generated phase runtimes')
    a=p.parse_args();bind(a.prefill,a.decode,a.inputs,a.out,a.emit_source)
