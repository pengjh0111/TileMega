"""Check native generated tokens, full final logits and per-layer KV against eager."""
import argparse
import json
from pathlib import Path
import struct
import torch
from tilemega.bench.as2_models import RUNS, REPO, controlled
from tilemega.bench.as2_native import digest
from tilemega.bench.as2_compare_native import metrics


def compare(binding,native,reference,out):
    binding,native,reference,out=(controlled(p,[RUNS]) for p in (binding,native,reference,out))
    if out.exists():raise FileExistsError(out)
    identity=json.loads((binding/'request.json').read_text())
    actual=json.loads((native/'result.json').read_text());ref=json.loads((reference/'result.json').read_text())
    for key in ('model_id','batch','config_sha256','weight_manifest_sha256','input_sha256'):
        if identity[key]!=ref[key]:raise ValueError('Paired request identity mismatch: '+key)
    if actual['binding_directory']!=str(binding) or actual['osl']!=ref['osl'] or actual['isl']!=ref['isl'] or actual['repetitions']!=ref['repetitions'] or actual['warmup']!=ref['warmup']:
        raise ValueError('Incompatible request scopes')
    for phase,entry in identity['phases'].items():
        if any(digest(binding/phase/name)!=sha for name,sha in entry['bound_hashes'].items()):
            raise ValueError('Bound plan changed after request preparation')
    for entry in identity['input_buffers']:
        if digest(binding/'inputs'/entry['file'])!=entry['sha256']:
            raise ValueError('Shared input changed after request preparation')
    stages=list(struct.iter_unpack('<17Q16I',(binding/'decode/stages.bin').read_bytes()))
    buffers={b['offset_bytes']:b for b in identity['buffers']}
    logits=[s for s in stages if s[17]==0 and s[18]==3]
    attention=[i for i,s in enumerate(stages) if s[17]==13]
    if len(logits)!=1:raise ValueError('Require actual full vocabulary projection')
    logit=buffers[logits[0][16]]
    policy=json.loads((REPO/'configs/ascend/as2_correctness_policy.json').read_text())
    def check(x,y,tolerance):
        if x.stat().st_size!=y.stat().st_size:raise ValueError('Unequal paired tensor extent')
        size=x.stat().st_size//2
        return metrics(torch.from_file(str(x),shared=False,size=size,dtype=torch.bfloat16),
            torch.from_file(str(y),shared=False,size=size,dtype=torch.bfloat16),tolerance)
    rows=[]
    for repetition in range(-actual['warmup'],actual['repetitions']):
        path=native/('request_'+str(repetition));x=path/'tokens.bin';y=reference/f'tokens_{repetition}.bin'
        a=torch.from_file(str(x),shared=False,size=actual['batch']*actual['osl'],dtype=torch.int32).reshape(actual['batch'],actual['osl'])
        b=torch.from_file(str(y),shared=False,size=a.numel(),dtype=torch.int32).reshape_as(a)
        mismatch=(a!=b).nonzero();first=mismatch[0].tolist() if mismatch.numel() else None
        row=dict(repetition=repetition,greedy=dict(passed=not mismatch.numel(),failed_elements=int(mismatch.shape[0]),
            first_divergence=first,native_sha256=digest(x),reference_sha256=digest(y)),
            logits=check(path/'decode'/f'{logit["id"]}.bin',reference/f'logits_last_{repetition}.bin',policy['model_logits']),kv=[])
        for layer,stage in enumerate(attention):
            for side in ('k','v'):
                x=path/'decode'/f'kv_{side}_s{stage}.bin';y=reference/f'kv_{side}_l{layer}_r{repetition}.bin'
                if not y.exists():row['kv'].append(dict(layer=layer,side=side,passed=False,status='not_run'))
                else:row['kv'].append(dict(layer=layer,side=side,**check(x,y,policy['decoder_layer'])))
        row['passed']=row['greedy']['passed'] and row['logits']['passed'] and all(k['passed'] for k in row['kv'])
        rows.append(row)
    launch_rows=[json.loads(line) for line in (native/'launches.jsonl').read_text().splitlines()]
    expected=(actual['warmup']+actual['repetitions'])*actual['osl']
    launch_check=len(launch_rows)==expected==actual['host_kernel_launches'] and all(
        r['forward_index']==i%actual['osl'] and r['seq']==(1 if i%actual['osl'] else 64) and
        r['past']==(63+i%actual['osl'] if i%actual['osl'] else 0) for i,r in enumerate(launch_rows))
    result=dict(schema='tilemega.as2.request_numeric.v1',model_id=identity['model_id'],batch=actual['batch'],
        isl=actual['isl'],osl=actual['osl'],status='pass' if launch_check and all(r['passed'] for r in rows) else 'fail',
        scope='full_model_natural_greedy_request',actual_single_forward_launch_count=launch_check,
        reference_system=ref['system'],reference_directory=str(reference),
        tolerance_version=policy['schema'],teacher_forced_512='not_run',performance_gate='not_admitted',rows=rows)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    return 0 if result['status']=='pass' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binding','native','reference','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);raise SystemExit(compare(a.binding,a.native,a.reference,a.out))
