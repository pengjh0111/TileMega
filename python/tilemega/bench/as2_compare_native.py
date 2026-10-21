"""Compare native decode receipts and intermediates without changing tolerances."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import struct
import torch
from tilemega.bench.as2_models import REPO, RUNS, controlled


def metrics(actual, reference, tolerance):
    x=actual.float();y=reference.float();difference=x-y;error=difference.abs()
    finite=x.isfinite() & y.isfinite();nonfinite=int((~finite).sum())
    rmse=float(difference.square().mean().sqrt());scale=max(float(y.square().mean().sqrt()),1e-12)
    failed=int((~finite | (error>tolerance['atol']+tolerance['rtol']*y.abs())).sum())
    nrmse=rmse/scale
    def number(v):return v if torch.isfinite(torch.tensor(v)) else None
    return dict(max_abs=number(float(error.max())),max_rel=number(float((error/y.abs().clamp_min(1e-12)).max())),
        rmse=number(rmse),nrmse=number(nrmse),nonfinite=nonfinite,failed_elements=failed,
        elements=x.numel(),passed=not nonfinite and not failed and ('nrmse_max' not in tolerance or nrmse<=tolerance['nrmse_max']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','inputs','native','reference','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    for name in ('plan','inputs','native','reference','out'):setattr(a,name,controlled(getattr(a,name),[RUNS]))
    if a.out.exists():raise FileExistsError(a.out)
    policy=json.loads((REPO/'configs/ascend/as2_correctness_policy.json').read_text())
    identity=json.loads((a.inputs/'inputs.json').read_text());reference=json.loads((a.reference/'reference.json').read_text())
    for key in ('model_id','batch','past','initial_kv','config_sha256','weight_manifest_sha256','input_sha256'):
        if identity[key]!=reference[key]:raise ValueError('Paired identity mismatch: '+key)
    buffers=json.loads((a.plan/'buffers.json').read_text())['buffers'];by_name={b['name']:b for b in buffers}
    stages=list(struct.iter_unpack('<17Q16I',(a.plan/'stages.bin').read_bytes()))
    heads=[s for s in stages if s[17]==0 and s[18]==3]
    if len(heads)!=1:raise ValueError('Require one verified vocabulary projection')
    logit=next(b for b in buffers if b['offset_bytes']==heads[0][16])
    attention=[i for i,s in enumerate(stages) if s[17]==13]
    rows=[]
    for r in reference['files']:
        name=r['name'];kind='intermediate';native=None
        if name.startswith('native.gemm.raw.') or name=='model.logits':
            name='model.logits';kind='model_logits';native=a.native/(str(logit['id'])+'.bin')
        elif name.startswith('layer') and '.kv_' in name:
            layer=int(name.split('.')[0][5:]);side=name.rpartition('_')[2]
            native=a.native/f'kv_{side}_s{attention[layer]}.bin';kind='kv'
        elif name in by_name:native=a.native/(str(by_name[name]['id'])+'.bin')
        if native is None or not native.exists():
            rows.append(dict(name=name,status='not_run',reason='native output missing'));continue
        n=r['elements']
        if native.stat().st_size!=n*2:raise ValueError('Output size mismatch: '+name)
        x=torch.from_file(str(native),shared=False,size=n,dtype=torch.bfloat16)
        y=torch.from_file(str(a.reference/r['file']),shared=False,size=n,dtype=torch.bfloat16)
        tolerance=policy['model_logits'] if kind=='model_logits' else policy['decoder_layer']
        m=metrics(x,y,tolerance);rows.append(dict(name=name,kind=kind,status='pass' if m['passed'] else 'fail',**m))
    tokens=a.native/'tokens.bin';token_status='not_run';actual=None
    if tokens.exists():
        state=torch.from_file(str(tokens),shared=False,size=identity['batch']*1088,dtype=torch.int32).reshape(identity['batch'],1088)
        actual=state[:,identity['past']+identity.get('seq',1)].tolist();token_status='pass' if actual==reference['token_ids'] else 'fail'
    receipts='not_run'
    if (a.native/'trace.bin').exists():
        trace=list(struct.iter_unpack('<8Q',(a.native/'trace.bin').read_bytes()))
        tasks=list(struct.iter_unpack('<7I',(a.plan/'tasks.bin').read_bytes()))
        receipts='pass' if len(tasks)==len(trace) and all(t[0]==r[1] and t[1]==r[2] and t[2]==r[0] and r[4]>=r[3] and r[6:8]==stages[t[0]][:2] for t,r in zip(tasks,trace)) else 'fail'
    missing=sum(r['status']=='not_run' for r in rows);failed=sum(r['status']=='fail' for r in rows)
    result=dict(schema='tilemega.as2.native_numeric.v1',model_id=identity['model_id'],batch=identity['batch'],past=identity['past'],
        scope='full_layer_count_phase_diagnostic_not_generation',phase='decode' if identity.get('seq',1)==1 else 'prefill',
        status='pass' if not missing and not failed and token_status==receipts=='pass' else 'fail',
        full_generation_gate='not_started',tolerance_version=policy['schema'],
        greedy=dict(status=token_status,native=actual,reference=reference['token_ids']),launch_task_receipts=receipts,
        checked_outputs=len(rows),missing_outputs=missing,failed_outputs=failed,
        model_logits=next(r for r in rows if r['name']=='model.logits'),outputs=rows)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='outputs'},allow_nan=False))
    return 0 if result['status']=='pass' else 1

if __name__=='__main__':raise SystemExit(main())
