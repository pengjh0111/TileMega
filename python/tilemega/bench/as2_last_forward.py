"""Compare independently observed eager and native last-forward intermediates."""
import argparse
import json
import math
import struct
from pathlib import Path
import torch
from tilemega.bench.as2_models import RUNS,REPO,WORK,controlled
from tilemega.bench.as2_native import digest
from tilemega.bench.as2_compare_native import metrics


def read(path,shape,dtype=torch.bfloat16):
    count=math.prod(shape)
    if path.stat().st_size!=count*torch.empty((),dtype=dtype).element_size():raise ValueError('Wrong observed tensor extent')
    return torch.from_file(str(path),shared=False,size=count,dtype=dtype).reshape(shape)


def compare_history(binding,native,reference,out):
    identity=json.loads((binding/'request.json').read_text());actual=json.loads((native/'result.json').read_text())
    ref=json.loads((reference/'result.json').read_text());capture=json.loads((reference/'last_forward/capture.json').read_text())
    for name in ('model_id','batch','config_sha256','weight_manifest_sha256','input_sha256'):
        if identity[name]!=ref[name]:raise ValueError('KV prefix identity mismatch: '+name)
    if actual.get('request_mode','natural_greedy')!='natural_greedy' or actual['binding_directory']!=str(binding) or actual['osl']<ref['osl'] or actual['repetitions']!=1 or actual['warmup'] or identity['batch']!=1 or not ref.get('diagnostic_capture_last_forward'):
        raise ValueError('Require one B1 natural request and an independent causal-prefix capture')
    a=read(native/'request_0/tokens.bin',(1,actual['osl']),torch.int32)
    b=read(reference/'tokens_0.bin',(1,ref['osl']),torch.int32)
    if (a[:,:ref['osl']]!=b).any():raise ValueError('Different token paths cannot validate cache identity')
    for phase,entry in identity['phases'].items():
        if any(digest(binding/phase/name)!=sha for name,sha in entry['bound_hashes'].items()):raise ValueError('Bound plan changed')
    stages=list(struct.iter_unpack('<17Q16I',(binding/'decode/stages.bin').read_bytes()));attention=[i for i,s in enumerate(stages) if s[17]==13]
    files={v['name']:v for v in capture['files']};rows=[];valid=actual['osl']+63
    for layer,index in enumerate(attention):
        s=stages[index]
        for side in ('k','v'):
            record=files[f'l{layer}.kv_{side}'];gold_path=reference/'last_forward'/record['file']
            if record['shape']!=[1,s[23],capture['past']+1,s[24]] or digest(gold_path)!=record['sha256']:
                raise ValueError('Independent KV capture changed')
            path=native/'request_0/decode'/f'kv_{side}_history_s{index}.bin'
            value=read(path,(1,s[23],valid,s[24]))[:,:,:capture['past']+1,:];gold=read(gold_path,record['shape'])
            different=(value!=gold).nonzero()
            rows.append(dict(layer=layer,side=side,compared_elements=gold.numel(),bitwise_different=len(different),
                first_different_position=int(different[:,2].min()) if len(different) else None))
    result=dict(schema='tilemega.as2.causal_kv_prefix.v1',status='pass' if all(not v['bitwise_different'] for v in rows) else 'fail',
        scope='all saved layers and valid positions of an independent same-token causal prefix; not timing',
        native_outputs=actual['osl'],reference_outputs=ref['osl'],positions_compared=capture['past']+1,
        boundaries_covered=[n for n in (63,64,65,127,128,129,1023,1024) if n<=capture['past']],rows=rows)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


def compare(binding,native,reference,out):
    identity=json.loads((binding/'request.json').read_text());actual=json.loads((native/'result.json').read_text())
    ref=json.loads((reference/'result.json').read_text());capture=json.loads((reference/'last_forward/capture.json').read_text())
    for key in ('model_id','batch','config_sha256','weight_manifest_sha256','input_sha256'):
        if identity[key]!=ref[key]:raise ValueError('Paired identity mismatch: '+key)
    if actual['binding_directory']!=str(binding) or actual['osl']!=ref['osl'] or not ref.get('diagnostic_capture_last_forward') or actual['repetitions']!=1 or actual['warmup']:
        raise ValueError('Require one paired last-forward diagnostic request')
    batch=identity['batch'];outputs=actual['osl'];vocab=None
    a=read(native/'request_0/tokens.bin',(batch,outputs),torch.int32)
    b=read(reference/'tokens_0.bin',(batch,outputs),torch.int32)
    if (a[:,:-1]!=b[:,:-1]).any():raise ValueError('Token paths diverged before the captured forward')
    if capture['past']!=62+outputs or capture['seq']!=1:raise ValueError('Different captured positions')
    for phase,entry in identity['phases'].items():
        if any(digest(binding/phase/name)!=sha for name,sha in entry['bound_hashes'].items()):raise ValueError('Bound plan changed')
    cfg=json.loads((WORK/'weights'/identity['model_id']/'manifest.json').read_text())['config'];layers=cfg['num_hidden_layers']
    stages=list(struct.iter_unpack('<17Q16I',(binding/'decode/stages.bin').read_bytes()))
    buffers=identity['buffers'];named={b['name']:b for b in buffers};by_offset={b['offset_bytes']:b for b in buffers}
    residual=[s for s in stages if s[17]==0 and s[18]==1];gate_up=[s for s in stages if s[17]==0 and s[18]==2]
    attention=[i for i,s in enumerate(stages) if s[17]==13];heads=[s for s in stages if s[17]==0 and s[18]==3]
    if len(residual)!=2*layers or len(gate_up)!=layers or len(heads)!=1 or len(attention)!=layers:
        raise ValueError('Unrecognized dense stage decomposition')
    observed={r['name']:r for r in capture['files']}
    if any(digest(reference/'last_forward'/r['file'])!=r['sha256'] for r in capture['files']):raise ValueError('Independent capture changed')
    policy=json.loads((REPO/'configs/ascend/as2_correctness_policy.json').read_text());rows=[]
    def append(name,path,gold,category='decoder_layer',storage_bytes=None):
        if path.stat().st_size!=(storage_bytes or gold.numel()*2) or path.stat().st_size<gold.numel()*2:
            raise ValueError('Paired stage extents differ: '+name)
        # Binding reserves max(prefill,decode) by name. Only the leading
        # batch*seq=1 rows belong to this captured forward; the tail is inactive.
        value=torch.from_file(str(path),shared=False,size=gold.numel(),dtype=torch.bfloat16).reshape(gold.shape)
        rows.append(dict(name=name,native_file=str(path),bitwise_different=int((value!=gold).sum()),
            compared_elements=gold.numel(),native_storage_bytes=path.stat().st_size,
            **metrics(value,gold,policy[category])))
    folder=native/'request_0/decode'
    for i in range(layers):
        for name in ('norm1','qkv','context','o.raw','norm2','gate_up.raw','act','down.raw'):
            name=f'l{i}.{name}';r=observed[name]
            if r['dtype']!='torch.bfloat16':raise ValueError('Unexpected captured dtype')
            if name.endswith('.o.raw'):buffer=by_offset[residual[2*i][16]]
            elif name.endswith('.down.raw'):buffer=by_offset[residual[2*i+1][16]]
            elif name.endswith('.gate_up.raw'):buffer=by_offset[gate_up[i][16]]
            else:buffer=named[name]
            expected=batch*(cfg['hidden_size'] if name.endswith(('.norm1','.norm2','.o.raw','.down.raw')) else
                cfg['intermediate_size']*2 if name.endswith('.gate_up.raw') else cfg['intermediate_size'] if name.endswith('.act') else
                cfg['num_attention_heads']*cfg['head_dim'] if name.endswith('.context') else
                (cfg['num_attention_heads']+2*cfg['num_key_value_heads'])*cfg['head_dim'])
            if math.prod(r['shape'])!=expected:raise ValueError('Observed shape differs from the model decode contract')
            append(name,folder/f"{buffer['id']}.bin",read(reference/'last_forward'/r['file'],r['shape']),storage_bytes=buffer['bytes'])
    r=observed['final.normalized'];buffer=named['final.normalized']
    append('final.normalized',folder/f"{buffer['id']}.bin",read(reference/'last_forward'/r['file'],r['shape']),storage_bytes=buffer['bytes'])
    vocab=heads[0][21];logit=by_offset[heads[0][16]]
    gold=read(reference/'logits_last_0.bin',(batch,vocab));path=folder/f"{logit['id']}.bin"
    append('model.logits',path,gold,'model_logits');value=read(path,(batch,vocab))
    margins=[]
    for system,tensor,pred in [('native',value,a[:,-1]),('reference',gold,b[:,-1])]:
        for row in range(batch):
            top=tensor[row].float().topk(2);maximum=top.values[0];ties=(tensor[row].float()==maximum).nonzero().flatten()
            margins.append(dict(system=system,batch_row=row,chosen_token=int(pred[row]),max_value=float(maximum),
                top2_margin=float(top.values[0]-top.values[1]),max_tie_count=ties.numel(),lowest_max_index=int(ties.min())))
    for i,stage in enumerate(attention):
        for side in ('k','v'):
            r=observed[f'l{i}.kv_{side}'];data=read(reference/'last_forward'/r['file'],r['shape'])
            append(f'l{i}.kv_{side}.last',folder/f'kv_{side}_s{stage}.bin',data[:,:,-1,:])
            history=folder/f'kv_{side}_history_s{stage}.bin'
            if history.exists():
                append(f'l{i}.kv_{side}.history',history,data)
                locations=(read(history,data.shape)!=data).nonzero()
                rows[-1]['first_different_position']=int(locations[:,2].min()) if locations.numel() else None
                rows[-1]['first_coordinates']=locations[:16].tolist()
    result=dict(schema='tilemega.as2.last_forward_difference.v1',status='diagnostic_complete',
        scope='actual independent same-token-prefix forward; no correctness/performance admission',
        model_id=identity['model_id'],outputs=outputs,past=capture['past'],same_token_prefix=True,
        prefix_outputs_compared=outputs-1,rows=rows,logit_margins=margins,
        first_non_bitwise=next((r['name'] for r in rows if r['bitwise_different']),None),
        all_observed_bitwise=all(not r['bitwise_different'] for r in rows),reference=str(reference),native=str(native))
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binding','native','reference','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--compare-kv-prefix',action='store_true')
    a=p.parse_args();torch.set_num_threads(1)
    for name in ('binding','native','reference','out'):setattr(a,name,controlled(getattr(a,name),[RUNS]))
    if a.out.exists():raise FileExistsError(a.out)
    (compare_history if a.compare_kv_prefix else compare)(a.binding,a.native,a.reference,a.out)


if __name__=='__main__':main()
