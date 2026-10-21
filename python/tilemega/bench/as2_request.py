"""Relocate two solved phase plans into one named request arena, without rescheduling."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
from tilemega.bench.as2_models import RUNS, controlled

TABLES=('metadata.bin','stages.bin','tasks.bin','waits.bin','offsets.bin','queue.bin','allocations.bin')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def bind(prefill, decode, inputs, out):
    prefill,decode,inputs,out=(controlled(p,[RUNS]) for p in (prefill,decode,inputs,out))
    if out.exists():raise FileExistsError(out)
    identity=json.loads((inputs/'inputs.json').read_text())
    plans=[]
    for phase,path,seq in [('prefill',prefill,64),('decode',decode,1)]:
        imported=json.loads((path.parent/'import.json').read_text())
        m=list(struct.unpack('<12I3Q',(path/'metadata.bin').read_bytes()))
        if m[0]!=2 or m[9]!=seq or m[10]!=1088 or m[8]!=identity['batch']:
            raise ValueError('Incompatible phase ABI')
        if imported['phase']!=phase or imported['model_id']!=identity['model_id'] or imported['config_sha256']!=identity['config_sha256']:
            raise ValueError('Phase model/config mismatch')
        buffers=json.loads((path/'buffers.json').read_text())['buffers']
        allocations=list(struct.iter_unpack('<3Q2I',(path/'allocations.bin').read_bytes()))
        if len(buffers)!=len(allocations) or len({b['name'] for b in buffers})!=len(buffers):
            raise ValueError('Ambiguous allocation identity')
        for i,(b,a) in enumerate(zip(buffers,allocations)):
            if b['id']!=i or (b['offset_bytes'],b['bytes'],b['allocated_bytes'],int(b.get('required_input',False)),0)!=a:
                raise ValueError('Allocation manifest does not match binary tables')
        plans.append(dict(phase=phase,path=path,metadata=m,buffers=buffers))
    if identity['seq']!=64 or identity['past'] or digest(prefill/'metadata.bin')!=identity['metadata_sha256']:
        raise ValueError('Request must start with the actual prefill inputs')
    if plans[0]['metadata'][1:3]!=plans[1]['metadata'][1:3]:
        raise ValueError('Request phases require the same validated worker topology')
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
        isl=64,capacity=1088,arena_bytes=total,source_inputs=str(inputs),
        optimization_status='unpriced_phase_bootstrap_not_admitted',phases={},buffers=union)
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
        for name in ('tasks.bin','waits.bin','offsets.bin','queue.bin'):
            os.link(plan['path']/name,dest/name)
        for name in ('implementations.json',):
            os.link(plan['path']/name,dest/name)
        description=json.loads((plan['path']/'buffers.json').read_text())
        description.update(arena_bytes=total,buffers=union,relocated_request_binding=True)
        (dest/'buffers.json').write_text(json.dumps(description,indent=2)+'\n')
        receipt['phases'][plan['phase']]=dict(source_plan=str(plan['path']),
            source_hashes={name:digest(plan['path']/name) for name in TABLES},
            bound_hashes={name:digest(dest/name) for name in TABLES})
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
    (out/'reset.bin').write_bytes(struct.pack('<2I',receipt['tokens_buffer'],len(receipt['kv_buffers']))+
        struct.pack('<'+'I'*len(receipt['kv_buffers']),*receipt['kv_buffers']))
    (out/'request.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('buffers','phases','input_buffers')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prefill','decode','inputs','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();bind(a.prefill,a.decode,a.inputs,a.out)
