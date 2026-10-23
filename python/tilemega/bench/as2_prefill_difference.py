"""Locate the first composed prefill divergence using independently captured buffers."""
import argparse
import json
from pathlib import Path
import torch
from tilemega.bench.as2_models import RUNS,WORK,REPO,controlled
from tilemega.bench.as2_native import digest
from tilemega.bench.as2_compare_native import metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','native','reference','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--layers',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--kinds',nargs='+',choices=('norm1','qkv','context','norm2','act'),
        default=['norm1','qkv','context','norm2','act'],help='Explicit diagnostic subset; requested missing buffers still fail')
    a=p.parse_args();torch.set_num_threads(1)
    plan,native,reference,out=(controlled(getattr(a,name),[RUNS]) for name in ('plan','native','reference','out'))
    if out.exists():raise FileExistsError(out)
    imported=json.loads((plan.parent/'import.json').read_text());ref=json.loads((reference/'reference.json').read_text())
    manifest=WORK/'weights'/imported['model_id']/'manifest.json'
    if imported['phase']!='prefill' or ref['seq']!=64 or ref['model_id']!=imported['model_id'] or ref['config_sha256']!=imported['config_sha256'] or ref['weight_manifest_sha256']!=digest(manifest):
        raise ValueError('Prefill model/config/weights mismatch')
    buffers={b['name']:b for b in json.loads((plan/'buffers.json').read_text())['buffers']}
    files={f['name']:f for f in ref['files']};rows=[]
    policy=json.loads((REPO/'configs/ascend/as2_correctness_policy.json').read_text())['decoder_layer']
    for layer in a.layers:
        for kind in a.kinds:
            name=f'l{layer}.{kind}';b=buffers[name];f=files[name]
            x=native/f'{b["id"]}.bin';y=reference/f['file']
            if x.stat().st_size!=y.stat().st_size or x.stat().st_size!=b['bytes']:raise ValueError('Diagnostic buffer extent mismatch: '+name)
            xt=torch.from_file(str(x),shared=False,size=b['bytes']//2,dtype=torch.bfloat16)
            yt=torch.from_file(str(y),shared=False,size=b['bytes']//2,dtype=torch.bfloat16)
            rows.append(dict(name=name,bitwise_different=int((xt!=yt).sum()),native_sha256=digest(x),reference_sha256=digest(y),**metrics(xt,yt,policy)))
    result=dict(schema='tilemega.as2.prefill_difference.v1',status='diagnostic_completed_not_model_admission',
        model_id=ref['model_id'],reference_implementation=ref['implementation'],selected_kinds=a.kinds,rows=rows,
        first_nonbitwise=next((r['name'] for r in rows if r['bitwise_different']),None),
        first_policy_failure=next((r['name'] for r in rows if not r['passed']),None))
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
