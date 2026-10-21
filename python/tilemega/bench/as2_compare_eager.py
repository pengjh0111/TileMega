"""Admit an eager operator/cache variant against the independent HF eager graph."""
import argparse
import json
from pathlib import Path
import torch

RUNS=Path('/home/d50064638/junhuip/TileMega/runs/as2')
def controlled(path):
    path=path.resolve(strict=True)
    if not path.is_relative_to(RUNS) or path==RUNS:raise ValueError('Use AS-2 evidence paths')
    return path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    reference=controlled(args.reference);candidate=controlled(args.candidate)
    r=json.loads((reference/'result.json').read_text());c=json.loads((candidate/'result.json').read_text())
    for field in ('model_id','batch','isl','osl','capacity','dtype','config_sha256','weight_manifest_sha256','input_sha256'):
        if r[field]!=c[field]:raise ValueError('Baseline identity mismatch: '+field)
    if r['status']!='completed_not_numerically_admitted' or c['status']!='completed_not_numerically_admitted':
        raise ValueError('Both requests must have completed')
    if r['norm']!='hf':raise ValueError('Reference must use the independent HF normalization graph')
    rs=[s for s in r['samples'] if not s['warmup']];cs=[s for s in c['samples'] if not s['warmup']]
    policy_path=RUNS.parents[1]/'configs/ascend/as2_correctness_policy.json'
    policy=json.loads(policy_path.read_text())['model_logits'];checks=[]
    for run in cs:
        ri=rs[0]['repetition'];ci=run['repetition']
        token_equal=run['output_sha256']==rs[0]['output_sha256']
        def logits(folder,index):
            path=folder/('logits_last_'+str(index)+'.bin')
            return torch.from_file(str(path),shared=False,size=path.stat().st_size//2,dtype=torch.bfloat16).float()
        x=logits(candidate,ci);y=logits(reference,ri)
        if x.shape!=y.shape:raise ValueError('Logit shape mismatch')
        delta=(x-y).abs();rmse=delta.square().mean().sqrt().item()
        nrmse=rmse/max(y.square().mean().sqrt().item(),1e-12)
        failed=int((delta>policy['atol']+policy['rtol']*y.abs()).sum())
        nonfinite=int((~torch.isfinite(x)).sum())
        checks.append(dict(repetition=ci,all_greedy_tokens_equal=token_equal,max_abs=delta.max().item(),
            max_rel=(delta/y.abs().clamp_min(1e-12)).max().item(),rmse=rmse,nrmse=nrmse,
            failed_elements=failed,nonfinite=nonfinite,
            pass_=token_equal and not failed and not nonfinite and nrmse<=policy['nrmse_max']))
    result=dict(status='passed' if all(x['pass_'] for x in checks) else 'failed',
        scope='eager variant admission only; not a TileMega correctness or performance gate',
        reference=str(reference),candidate=str(candidate),policy=policy,checks=checks)
    out=args.out.absolute()
    controlled(out.parent)
    with out.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));return 0 if result['status']=='passed' else 1

if __name__=='__main__':raise SystemExit(main())
