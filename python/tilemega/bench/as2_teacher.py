"""Canonical teacher inputs and independent full-model teacher-forced checks."""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import torch
from tilemega.bench.as2_models import RUNS, WORK, REPO, DENSE, SEED, controlled
from tilemega.bench.as2_native import digest
from tilemega.bench.as2_compare_native import metrics


def checkpoints(steps):
    return sorted({0,steps,*[n for n in (1,2,64,65,66,192,193,448,449) if n<=steps]})


def read(path,shape,dtype=torch.bfloat16):
    count=__import__('math').prod(shape)
    if path.stat().st_size!=count*torch.empty((),dtype=dtype).element_size():raise ValueError('Teacher tensor extent mismatch')
    return torch.from_file(str(path),shared=False,size=count,dtype=dtype).reshape(shape)


def save(path,tensor):
    path.write_bytes(tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes())


def prepare(a):
    if not torch.__version__.startswith('2.7.1'):raise ValueError('Frozen teacher generator requires torch2.7.1')
    manifest_path=WORK/'weights'/a.model/'manifest.json';m=json.loads(manifest_path.read_text())
    if m['status']!='complete':raise ValueError('Require complete canonical weights')
    seed=int.from_bytes(hashlib.sha256(f'as2-teacher-v1:{a.model}:{SEED}'.encode()).digest()[:8],'little')&((1<<63)-1)
    values=torch.randint(m['config']['vocab_size'],(16,512),generator=torch.Generator().manual_seed(seed),dtype=torch.int32)
    a.out.mkdir();save(a.out/'tokens.bin',values)
    (a.out/'header.bin').write_bytes(struct.pack('<5I',1,16,512,1088,m['config']['vocab_size']))
    result=dict(schema='tilemega.as2.teacher_trace.v1',model_id=a.model,seed=SEED,
        generator='as2-teacher-v1-name-sha256-torch2.7.1',shape=[16,512],dtype='int32',b1='first row of B16',
        config_sha256=m['config_sha256'],weight_manifest_sha256=digest(manifest_path),input_sha256=m['input_ids']['sha256'],
        files={p.name:digest(p) for p in a.out.iterdir()})
    (a.out/'trace.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


def validate(trace):
    r=json.loads((trace/'trace.json').read_text())
    if any(digest(trace/name)!=sha for name,sha in r['files'].items()):raise ValueError('Teacher trace changed')
    m=WORK/'weights'/r['model_id']/'manifest.json'
    if digest(m)!=r['weight_manifest_sha256']:raise ValueError('Canonical model identity changed')
    return r


def reference(a):
    from tilemega.bench.as2_eager import load_model
    from transformers.cache_utils import StaticCache
    import torch_npu
    r=validate(a.trace);torch.npu.set_device(0)
    if torch.npu.device_count()!=1:raise ValueError('Use the single-device wrapper')
    model,tokens,m,mhash=load_model(r['model_id'],'sdpa','fused')
    if mhash!=r['weight_manifest_sha256']:raise ValueError('Teacher model mismatch')
    teacher=read(a.trace/'tokens.bin',(16,512),torch.int32)[:a.batch].npu();tokens=tokens[:a.batch].npu()
    positions=torch.arange(1088,dtype=torch.int64,device='npu:0')
    cache=StaticCache(config=model.config,max_batch_size=a.batch,max_cache_len=1088,device=torch.device('npu:0'),dtype=torch.bfloat16)
    predictions=torch.empty((a.batch,a.steps+1),dtype=torch.int32,device='npu:0');points=checkpoints(a.steps)
    a.out.mkdir();(a.out/'teacher_checkpoints').mkdir()
    with torch.inference_mode():
        for step in range(a.steps+1):
            position=63+step if step else 0
            result=model(input_ids=teacher[:,step-1:step] if step else tokens,past_key_values=cache,use_cache=True,
                cache_position=positions[position:position+1] if step else positions[:64],
                position_ids=positions[None,position:position+1] if step else positions[None,:64],logits_to_keep=1,return_dict=True)
            predictions[:,step]=result.logits[:,-1].argmax(-1).to(torch.int32)
            if step in points:save(a.out/'teacher_checkpoints'/f'logits_s{step}.bin',result.logits[:,-1])
        torch.npu.synchronize();save(a.out/'teacher_predictions.bin',predictions)
        for layer,(key,value) in enumerate(zip(cache.key_cache,cache.value_cache)):
            for side,tensor in (('k',key),('v',value)):save(a.out/f'kv_{side}_l{layer}.bin',tensor[:a.batch,:,63+a.steps,:])
    result=dict(schema='tilemega.as2.teacher_reference.v1',system='torch_npu_eager',request_mode='teacher_forced',
        timing_admitted=False,model_id=r['model_id'],batch=a.batch,decoder_steps=a.steps,model_forwards=a.steps+1,
        checkpoints=points,trace_sha256=digest(a.trace/'trace.json'),config_sha256=r['config_sha256'],
        weight_manifest_sha256=mhash,input_sha256=r['input_sha256'],norm='fused',attention='sdpa',
        torch_version=torch.__version__,torch_npu_version=torch_npu.__version__,
        files={str(p.relative_to(a.out)):digest(p) for p in a.out.rglob('*.bin')})
    (a.out/'teacher.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files'}))


def compare(a):
    trace=validate(a.trace);identity=json.loads((a.binding/'request.json').read_text())
    native=json.loads((a.native/'teacher.json').read_text());result=json.loads((a.native/'result.json').read_text())
    ref=json.loads((a.reference/'teacher.json').read_text())
    for name in ('model_id','config_sha256','weight_manifest_sha256','input_sha256'):
        if identity[name]!=ref[name] or trace[name]!=ref[name]:raise ValueError('Teacher pairing mismatch: '+name)
    if any(v['request_mode']!='teacher_forced' for v in (native,result,ref)) or ref['system']!='torch_npu_eager' or native['timing_admitted'] or ref['timing_admitted']:
        raise ValueError('Teacher diagnostics cannot be timing/natural-greedy requests')
    steps=ref['decoder_steps'];batch=ref['batch'];points=checkpoints(steps)
    if native['decoder_steps']!=steps or native['batch']!=batch or points!=native['checkpoints'] or points!=ref['checkpoints'] or ref['trace_sha256']!=digest(a.trace/'trace.json') or native['trace_directory']!=str(a.trace):
        raise ValueError('Teacher extent/checkpoint identity mismatch')
    if any(digest(a.reference/name)!=sha for name,sha in ref['files'].items()):raise ValueError('Independent reference changed')
    if result['binding_directory']!=str(a.binding) or result['osl']!=steps+1:raise ValueError('Wrong native teacher binding')
    for phase,entry in identity['phases'].items():
        if any(digest(a.binding/phase/name)!=sha for name,sha in entry['bound_hashes'].items()):raise ValueError('Teacher plan changed')
    policy=json.loads((REPO/'configs/ascend/as2_correctness_policy.json').read_text());vocab=native['vocab'];rows=[]
    def check(name,xpath,ypath,shape,category):
        x=read(xpath,shape);y=read(ypath,shape)
        rows.append(dict(name=name,bitwise_different=int((x!=y).sum()),**metrics(x,y,policy[category])))
    for step in points:
        file=f'teacher_checkpoints/logits_s{step}.bin'
        check(f'logits_s{step}',a.native/file,a.reference/file,(batch,vocab),'model_logits')
    stages=list(struct.iter_unpack('<17Q16I',(a.binding/'decode/stages.bin').read_bytes()))
    for layer,index in enumerate(i for i,s in enumerate(stages) if s[17]==13):
        s=stages[index]
        for side in ('k','v'):
            check(f'kv_{side}_l{layer}',a.native/f'request_0/decode/kv_{side}_s{index}.bin',a.reference/f'kv_{side}_l{layer}.bin',(batch,s[23],s[24]),'decoder_layer')
    x=read(a.native/'teacher_predictions.bin',(batch,steps+1),torch.int32);y=read(a.reference/'teacher_predictions.bin',(batch,steps+1),torch.int32)
    different=(x!=y);launches=[json.loads(v) for v in (a.native/'launches.jsonl').read_text().splitlines()]
    launch_ok=len(launches)==steps+1==result['host_kernel_launches'] and all(v['forward_index']==i and v['past']==(63+i if i else 0) for i,v in enumerate(launches))
    record=dict(schema='tilemega.as2.teacher_numeric.v1',request_mode='teacher_forced',model_id=trace['model_id'],batch=batch,
        decoder_steps=steps,model_forwards=steps+1,full_teacher512=steps==512,status='pass' if launch_ok and not different.any() and all(r['passed'] for r in rows) else 'fail',
        actual_one_launch_per_forward=launch_ok,predicted_tokens_different=int(different.sum()),first_prediction_difference=different.nonzero()[:16].tolist(),
        timing_admitted=False,formal_greedy_admission=False,trace_sha256=digest(a.trace/'trace.json'),rows=rows)
    a.out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='rows'}))
    return 0 if record['status']=='pass' else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--model',choices=DENSE,required=True);prep.add_argument('--out',type=Path,required=True)
    replay=sub.add_parser('reference')
    for name in ('trace','out'):replay.add_argument('--'+name,type=Path,required=True)
    replay.add_argument('--batch',type=int,choices=(1,16),required=True);replay.add_argument('--steps',type=int,default=512)
    comp=sub.add_parser('compare')
    for name in ('trace','binding','native','reference','out'):comp.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    for name in ('trace','binding','native','reference','out'):
        if hasattr(a,name):setattr(a,name,controlled(getattr(a,name),[RUNS]))
    if a.out.exists():raise FileExistsError(a.out)
    if hasattr(a,'steps') and not 1<=a.steps<=512:raise ValueError('Bounded teacher steps required')
    return {'prepare':prepare,'reference':reference,'compare':compare}[a.command](a)


if __name__=='__main__':raise SystemExit(main())
