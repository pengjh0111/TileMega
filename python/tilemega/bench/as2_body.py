"""Prepare and compare one bounded native BF16 body case, without using its math."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

REPO=Path('/home/d50064638/junhuip/TileMega')
RUNS=REPO/'runs/as2'


def controlled(path):
    path=path.absolute()
    if path.resolve()!=path or path==RUNS or not path.is_relative_to(RUNS):
        raise ValueError('Use a canonical AS-2 runs path')
    return path


def save(path,value):
    path.write_bytes(value.contiguous().view(torch.uint8).numpy().tobytes())


def read(path,dtype,shape):
    count=1
    for d in shape:count*=d
    item_bytes=torch.empty((),dtype=dtype).element_size()
    if path.stat().st_size!=count*item_bytes:raise ValueError('Tensor shape/file mismatch')
    return torch.from_file(str(path),shared=False,size=count,dtype=dtype).reshape(shape)


def prepare(args):
    m,n,k=args.m,args.n,args.k
    if not 1<=m<=64 or not 1<=n<=151936 or not 1<=k<=16384 or (m*k+n*k+m*n)*4>256<<20:
        raise ValueError('Bounded family test exceeds CPU memory policy')
    out=controlled(args.out);out.mkdir(parents=True,exist_ok=False)
    g=torch.Generator().manual_seed(20261016)
    a=torch.randn((m,k),generator=g).bfloat16()
    b=(torch.randn((n,k),generator=g)*.02).bfloat16()
    save(out/'a.bin',a);save(out/'b_nk.bin',b)
    fp32=a.float()@b.float().T
    save(out/'ref_fp32.bin',fp32);save(out/'ref_bf16.bin',fp32.bfloat16())
    receipt=dict(shape_mnk=[m,n,k],seed=20261016,input_dtype='bf16',accumulator='fp32',
        weight_layout='NK',references=['CPU torch FP32 with BF16 inputs','BF16 cast of FP32 graph'],
        torch_version=torch.__version__)
    (out/'config.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


def compare(args):
    source=controlled(args.input);output=controlled(args.output)
    config=json.loads((source/'config.json').read_text())
    shape=config.get('shape',config.get('shape_mnk',[])[:2])
    result=read(output/'c.bin',torch.bfloat16,shape).float()
    policy_path=REPO/'configs/ascend/as2_correctness_policy.json'
    family=config.get('family','gemm')
    policy=json.loads(policy_path.read_text())['body'].get(family)
    if family=='add':policy=dict(atol=0,rtol=0,reason='one BF16 addition matches the BF16 reference exactly')
    if policy is None:raise ValueError('No frozen numerical policy for family')
    checks=[]
    for file,dtype in [('ref_fp32.bin',torch.float32),('ref_bf16.bin',torch.bfloat16)]:
        ref=read(source/file,dtype,shape).float();diff=(result-ref).abs()
        rmse=diff.square().mean().sqrt().item();nonfinite=int((~torch.isfinite(result)).sum())
        failed=int((diff>policy['atol']+policy['rtol']*ref.abs()).sum()) if family!='add' or dtype==torch.bfloat16 else 0
        nrmse=rmse/max(ref.square().mean().sqrt().item(),1e-12)
        checks.append(dict(reference=file,max_abs=diff.max().item(),max_rel=(diff/ref.abs().clamp_min(1e-12)).max().item(),
            rmse=rmse,nrmse=nrmse,nonfinite=nonfinite,
            failed_elements=failed,pass_=not failed and not nonfinite and nrmse<=policy.get('nrmse_max',float('inf'))))
    receipt=dict(status='passed' if all(c['pass_'] for c in checks) else 'failed',shape=shape,family=family,
        correctness_policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(),policy=policy,checks=checks)
    (output/'numeric.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
    return 0 if receipt['status']=='passed' else 1


def prepare_vector(args):
    rows,width=args.rows,args.width
    if not 1<=rows<=64 or not 1<=width<=14336 or (args.kind=='rmsnorm' and width>4096) or (args.kind=='rope' and width not in (64,128)):
        raise ValueError('Unsupported vector family shape')
    out=controlled(args.out);out.mkdir(parents=True,exist_ok=False)
    g=torch.Generator().manual_seed(20261016)
    x=torch.randn((rows,width),generator=g).bfloat16()
    if args.kind=='rmsnorm':
        rhs=(1+torch.randn((width,),generator=g)*.03).bfloat16()
        norm=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)
        fp32=norm.bfloat16().float()*rhs.float()
    elif args.kind=='rope':
        phase=torch.randn((rows,width//2),generator=g)*100
        phase=torch.cat((phase,phase),-1);rhs=phase.cos().bfloat16();third=phase.sin().bfloat16()
        save(out/'third.bin',third)
        rotated=torch.cat((-x[:,width//2:],x[:,:width//2]),-1)
        fp32=(x*rhs).float()+(rotated*third).float()
    else:
        rhs=torch.randn((rows,width),generator=g).bfloat16()
        fp32=torch.nn.functional.silu(x.float()).bfloat16().float()*rhs.float() if args.kind=='swiglu' else x.float()+rhs.float()
    save(out/'input.bin',x);save(out/'rhs.bin',rhs)
    save(out/'ref_fp32.bin',fp32);save(out/'ref_bf16.bin',fp32.bfloat16())
    receipt=dict(family=args.kind,shape=[rows,width],seed=20261016,dtype='bf16',opmath='fp32',
        epsilon=1e-6 if args.kind=='rmsnorm' else None,
        cast_contract='normalized/SILU/rotary products round to BF16 at the pinned HF graph boundaries')
    (out/'config.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


def prepare_attention(args):
    qh,kvh,keys,d=args.qheads,args.kvheads,args.keys,args.head_dim
    if not 1<=kvh<=qh<=32 or qh%kvh or not 1<=keys<=1088 or d not in (64,128):
        raise ValueError('Invalid bounded GQA shape')
    out=controlled(args.out);out.mkdir(parents=True,exist_ok=False)
    g=torch.Generator().manual_seed(20261016)
    q=torch.randn((qh,d),generator=g).bfloat16()
    k=torch.randn((kvh,keys,d),generator=g).bfloat16()
    v=torch.randn((kvh,keys,d),generator=g).bfloat16()
    save(out/'q.bin',q);save(out/'k.bin',k);save(out/'v.bin',v)
    mapping=torch.arange(qh)//(qh//kvh)
    scores=torch.einsum('hd,hkd->hk',q.float(),k[mapping].float())*d**-.5
    result=torch.einsum('hk,hkd->hd',scores.softmax(-1),v[mapping].float())
    save(out/'ref_fp32.bin',result);save(out/'ref_bf16.bin',result.bfloat16())
    receipt=dict(family='attention',shape=[qh,d],qheads=qh,kvheads=kvh,keys=keys,
        gqa_ratio=qh//kvh,head_dim=d,seed=20261016,dtype='bf16',opmath='fp32',
        reference='independent CPU FP32 QK/softmax/PV on BF16 inputs; all keys valid')
    (out/'config.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


def main():
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare-gemm')
    for name in ('m','n','k'):prep.add_argument('--'+name,type=int,required=True)
    prep.add_argument('--out',type=Path,required=True)
    check=sub.add_parser('compare-gemm');check.add_argument('--input',type=Path,required=True);check.add_argument('--output',type=Path,required=True)
    vec=sub.add_parser('prepare-vector');vec.add_argument('--kind',choices=('rmsnorm','swiglu','add','rope'),required=True)
    vec.add_argument('--rows',type=int,required=True);vec.add_argument('--width',type=int,required=True);vec.add_argument('--out',type=Path,required=True)
    vcheck=sub.add_parser('compare-vector');vcheck.add_argument('--input',type=Path,required=True);vcheck.add_argument('--output',type=Path,required=True)
    attn=sub.add_parser('prepare-attention')
    for name in ('qheads','kvheads','keys','head-dim'):attn.add_argument('--'+name,type=int,required=True)
    attn.add_argument('--out',type=Path,required=True)
    acheck=sub.add_parser('compare-attention');acheck.add_argument('--input',type=Path,required=True);acheck.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.command=='prepare-gemm':return prepare(args)
    if args.command=='prepare-vector':return prepare_vector(args)
    if args.command=='prepare-attention':return prepare_attention(args)
    return compare(args)


if __name__=='__main__':raise SystemExit(main())
