"""Pack verified serving operands for the native, decode-only AS-2 bootstrap."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import struct
import subprocess
import torch
from tilemega.bench.as2_models import ROOT, REPO, WORK, RUNS, DENSE, controlled


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pack(model_id, plan, out, past, reuse=None, reuse_plan=None):
    plan=controlled(plan,[RUNS]);out=controlled(out,[RUNS])
    if not 0 <= past <= 1086:
        raise ValueError('Decode bootstrap past is bounded by the token output capacity')
    raw=(plan/'metadata.bin').read_bytes()
    if len(raw)!=72:
        raise ValueError('Wrong native ABI')
    version,groups,workers,tasks,stages,waits,refs,stride,batch,seq,capacity,diagnostic,*_=struct.unpack('<12I3Q',raw)
    if version!=2 or seq not in (1,64) or capacity!=1088 or groups not in (2,4) or workers!=groups*3 or (seq==64 and past):
        raise ValueError('Unsupported prefill/decode plan')
    imported=json.loads((plan.parent/'import.json').read_text())
    if imported['model_id']!=model_id or imported['batch']!=batch or imported['past']!=(1087 if seq==1 else 0):
        raise ValueError('Plan/config identity mismatch')
    directory=WORK/'weights'/model_id
    manifest=json.loads((directory/'manifest.json').read_text())
    config_path=REPO/'configs/ascend/as2_models'/(model_id+'.json')
    if manifest['status']!='complete' or digest(config_path)!=manifest['config_sha256']:
        raise ValueError('Require frozen canonical weights and config')
    if imported.get('config_sha256',manifest['config_sha256'])!=manifest['config_sha256']:
        raise ValueError('Exported model config differs from canonical weights')
    records={r['name']:r for r in manifest['tensors']}
    for alias,name in manifest['aliases'].items():
        records[alias]=records[name]
    checked=set()
    def source(name):
        r=records[name];file=directory/r['file']
        if name not in checked:
            if file.stat().st_size!=r['bytes'] or digest(file)!=r['sha256']:
                raise ValueError('Canonical weight mismatch: '+name)
            checked.add(name)
        return r,file
    def tensor(name):
        r,path=source(name)
        return torch.from_file(str(path),shared=False,size=r['bytes']//2,dtype=torch.bfloat16).reshape(r['shape'])
    def data(t):
        return memoryview(t.contiguous().view(torch.uint8).numpy())
    cfg=manifest['config']
    from transformers import AutoConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    hf=AutoConfig.for_model(cfg['model_type'],**{k:v for k,v in cfg.items() if k!='model_type'})
    scaling=cfg.get('rope_scaling') or {}
    rope_type=scaling.get('rope_type',scaling.get('type','default'))
    inv,scale=ROPE_INIT_FUNCTIONS[rope_type](hf,torch.device('cpu'))
    d=cfg['head_dim']
    if inv.numel()!=d//2:
        raise ValueError('Native body requires full-head rotary dimension')
    angle=torch.arange(capacity,dtype=torch.float32)[:,None]*inv.float()[None,:]
    angle=torch.cat((angle,angle),-1)
    cos=(angle.cos()*scale).bfloat16();sin=(angle.sin()*scale).bfloat16()
    input_record=manifest['input_ids'];input_path=directory/input_record['file']
    if digest(input_path)!=input_record['sha256']:
        raise ValueError('Canonical input mismatch')
    ids=torch.from_file(str(input_path),shared=False,size=16*64,dtype=torch.int32).reshape(16,64)[:batch]
    tokens=torch.zeros(batch,capacity,dtype=torch.int32);tokens[:,:64]=ids
    if seq==1:tokens[:,past]=ids[:,-1]
    buffers=json.loads((plan/'buffers.json').read_text())['buffers']
    reusable={}
    if reuse is not None or reuse_plan is not None:
        if reuse is None or reuse_plan is None:raise ValueError('Reuse requires the original plan and input directory')
        reuse=controlled(reuse,[RUNS]);reuse_plan=controlled(reuse_plan,[RUNS])
        old=json.loads((reuse/'inputs.json').read_text())
        if old['model_id']!=model_id or old['config_sha256']!=manifest['config_sha256'] or old['weight_manifest_sha256']!=digest(directory/'manifest.json') or old['metadata_sha256']!=digest(reuse_plan/'metadata.bin'):
            raise ValueError('Reusable packing identity mismatch')
        descriptions={b['name']:b for b in json.loads((reuse_plan/'buffers.json').read_text())['buffers']}
        for record in old['buffers']:
            b=descriptions[record['name']]
            if b['name']=='serving.tokens' and (old['past']!=past or old['batch']!=batch or old.get('seq',1)!=seq):continue
            reusable[(b['name'],b['bytes'],b['dtype'],b['pack_json'])]=(record,reuse/record['file'])
    policy=json.loads((REPO/'configs/bench/as2_disk_policy.json').read_text())
    usage=shutil.disk_usage(ROOT)
    new_bytes=sum(b['bytes'] for b in buffers if b.get('required_input') and b['pack_json'] and json.loads(b['pack_json'])['kind']!='alias')
    owned=[RUNS,*(WORK/name for name in ('build','cache','tmp','weights'))]
    occupied=sum(int(subprocess.check_output(['du','-s','-B1',str(path)],text=True).split()[0]) for path in owned)
    if usage.free/usage.total<policy['stop_below_free_fraction'] or occupied+new_bytes>policy['project_generated_soft_limit_bytes']:
        raise RuntimeError('blocked-resource: private packing exceeds project disk policy')
    out.mkdir(parents=True,exist_ok=False);written=[]
    for b in buffers:
        if not b.get('required_input'):
            continue
        path=out/(str(b['id'])+'.bin');recipe=json.loads(b['pack_json']) if b['pack_json'] else None
        previous=reusable.get((b['name'],b['bytes'],b['dtype'],b['pack_json']))
        if previous:
            record,canonical=previous
            if digest(canonical)!=record['sha256']:raise ValueError('Reusable packed tensor changed')
            os.link(canonical,path)
        elif recipe:
            kind=recipe['kind']
            if kind=='alias':
                record,canonical=source(recipe['source'])
                if record['bytes']!=b['bytes']:
                    raise ValueError('Alias shape mismatch')
                # Hard links retain one canonical storage allocation. Neither
                # side mutates random tensors; candidates never duplicate them.
                os.link(canonical,path)
            elif kind=='qkv_group_interleave':
                q,k,v=(tensor(n) for n in recipe['sources'])
                g,u,d=recipe['hkv'],recipe['qperkv'],recipe['head_dim']
                if q.shape[0]!=g*u*d or k.shape!=v.shape or k.shape[0]!=g*d:
                    raise ValueError('Packed QKV shape mismatch')
                with path.open('xb') as stream:
                    for h in range(g):
                        for t in (q[h*u*d:(h+1)*u*d],k[h*d:(h+1)*d],v[h*d:(h+1)*d]):
                            stream.write(data(t))
                packed=torch.from_file(str(path),shared=False,size=b['bytes']//2,dtype=torch.bfloat16).reshape(g,(u+2)*d,q.shape[1])
                for h in {0,g//2,g-1}:
                    if not torch.equal(packed[h,:u*d],q[h*u*d:(h+1)*u*d]) or not torch.equal(packed[h,u*d:(u+1)*d],k[h*d:(h+1)*d]) or not torch.equal(packed[h,(u+1)*d:],v[h*d:(h+1)*d]):
                        raise ValueError('QKV inverse-layout check failed')
                del packed,q,k,v
            elif kind=='gate_up_interleave':
                gate,up=(tensor(n) for n in recipe['sources']);u=recipe['u']
                if gate.shape!=up.shape or gate.shape[0]%u:
                    raise ValueError('Gate/up shape mismatch')
                with path.open('xb') as stream:
                    for first in range(0,gate.shape[0],u):
                        stream.write(data(gate[first:first+u]));stream.write(data(up[first:first+u]))
                packed=torch.from_file(str(path),shared=False,size=b['bytes']//2,dtype=torch.bfloat16).reshape(-1,2*u,gate.shape[1])
                for h in {0,packed.shape[0]//2,packed.shape[0]-1}:
                    if not torch.equal(packed[h,:u],gate[h*u:(h+1)*u]) or not torch.equal(packed[h,u:],up[h*u:(h+1)*u]):
                        raise ValueError('Gate/up inverse-layout check failed')
                del packed,gate,up
            else:
                raise ValueError('Unsupported native packing: '+kind)
        else:
            value={'serving.tokens':tokens,'serving.rope_cos':cos,'serving.rope_sin':sin}.get(b['name'])
            if value is None:
                raise ValueError('Missing required native input: '+b['name'])
            with path.open('xb') as stream:stream.write(data(value))
        if path.stat().st_size!=b['bytes']:
            raise ValueError('Packed operand byte count mismatch')
        written.append(dict(buffer=b['id'],name=b['name'],file=path.name,bytes=path.stat().st_size,sha256=digest(path)))
    receipt=dict(schema='tilemega.ascend.native_inputs.v2',model_id=model_id,batch=batch,past=past,seq=seq,
        decode_only=seq==1,initial_kv='all_zero_diagnostic_state_not_prefill' if seq==1 else 'empty_cache_for_prefill',
        config_sha256=manifest['config_sha256'],weight_manifest_sha256=digest(directory/'manifest.json'),
        input_sha256=input_record['sha256'],metadata_sha256=digest(plan/'metadata.bin'),plan_directory=str(plan),
        buffers=written,rope_type=rope_type,rope_coefficient_source='pinned_transformers_cpu_fp32_then_bf16',
        inverse_layout_checks='first_middle_last_KV_group_or_interleave_group',status='packed_not_executed')
    (out/'inputs.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='buffers'}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=DENSE,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--past',type=int,default=64)
    p.add_argument('--reuse',type=Path);p.add_argument('--reuse-plan',type=Path)
    a=p.parse_args();torch.set_num_threads(1);pack(a.model,a.plan,a.out,a.past,a.reuse,a.reuse_plan)

if __name__=='__main__':main()
