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
    for name,expected in config.get('input_sha256',{}).items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected:raise ValueError('Observed Body input/reference changed')
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
            failed_elements=failed,bitwise_different=int((result!=ref).sum()),pass_=not failed and not nonfinite and nrmse<=policy.get('nrmse_max',float('inf'))))
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


def prepare_observed_norm(args):
    from tilemega.bench.as2_models import WORK
    source=controlled(args.reference);out=controlled(args.out)
    metadata=source/'reference.json';data_root=source
    if metadata.exists():
        r=json.loads(metadata.read_text())
        if args.norm!='norm2':raise ValueError('This phase reference captured only Norm2 inputs')
        inp=next(v for v in r['diagnostic_norm_inputs'] if v['layer']==args.layer)
        expected=next(v for v in r['files'] if v['name']==f'l{args.layer}.norm2')
    else:
        metadata=source/'result.json';r=json.loads(metadata.read_text())
        if not r.get('diagnostic_capture_last_forward') or r['system']!='torch_npu_eager' or r['norm']!='fused' or r['attention']!='sdpa':
            raise ValueError('Require an independent eager last-forward capture')
        r['implementation']='pinned_transformers_sdpa_fused_norm';data_root=source/'last_forward'
        records=json.loads((data_root/'capture.json').read_text())['files']
        input_name=args.norm[0]+'.raw' if args.norm in ('q_norm','k_norm') else args.norm+'.input'
        inp=next(v for v in records if v['name']==f'l{args.layer}.{input_name}')
        expected=next(v for v in records if v['name']==f'l{args.layer}.{args.norm}')
    if r['implementation']!='pinned_transformers_sdpa_fused_norm' or not 1<=args.rows<=64:
        raise ValueError('Require an independent fused-NPU reference and bounded rows')
    if any(hashlib.sha256((data_root/v['file']).read_bytes()).hexdigest()!=v['sha256'] for v in (inp,expected)):
        raise ValueError('Observed normalization operands changed')
    domain_shape=expected['shape'] if args.norm in ('q_norm','k_norm') else inp['shape']
    width=domain_shape[-1];total=__import__('math').prod(domain_shape[:-1])
    if args.rows>total or width not in (128,1024,2048,3072,4096):raise ValueError('Invalid full-row norm extent')
    weights=WORK/'weights'/r['model_id'];manifest=json.loads((weights/'manifest.json').read_text())
    if hashlib.sha256((weights/'manifest.json').read_bytes()).hexdigest()!=r['weight_manifest_sha256']:
        raise ValueError('Independent reference/canonical weight identity mismatch')
    module='self_attn.'+args.norm if args.norm in ('q_norm','k_norm') else 'input_layernorm' if args.norm=='norm1' else 'post_attention_layernorm'
    row=next(v for v in manifest['tensors'] if v['name']==f'model.layers.{args.layer}.{module}.weight')
    path=weights/row['file']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Norm scale identity changed')
    x=read(data_root/inp['file'],torch.bfloat16,(total,width))[:args.rows].contiguous()
    gamma=read(path,torch.bfloat16,(width,));gold=read(data_root/expected['file'],torch.bfloat16,(total,width))[:args.rows]
    eps=manifest['config']['rms_norm_eps'];y=x.float();fp32=(y*(y.square().mean(-1,keepdim=True)+eps).rsqrt()).bfloat16().float()*gamma.float()
    out.mkdir();save(out/'input.bin',x);save(out/'rhs.bin',gamma);save(out/'ref_fp32.bin',fp32);save(out/'ref_bf16.bin',gold)
    record=dict(family='rmsnorm',shape=[args.rows,width],dtype='bf16',epsilon=eps,source_reference=str(source),layer=args.layer,
        source_reference_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),norm=args.norm,
        original_shape=domain_shape if args.rows==total else [args.rows,width],
        references=dict(fp32='CPU FP32 same operands with explicit HF normalized BF16 cast',bf16='observed independent fused NPU output on these same operands'),
        input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()})
    (out/'config.json').write_text(json.dumps(record,indent=2)+'\n')


def prepare_attention(args):
    qh,kvh,keys,d=args.qheads,args.kvheads,args.keys,args.head_dim
    batch,rows=args.batch,args.query_rows;capacity=args.capacity or keys
    if (not 1<=kvh<=qh<=32 or qh%kvh or not 1<=keys<=capacity<=1088 or d not in (64,128) or
            batch not in (1,16) or rows not in (1,16) or keys<rows):
        raise ValueError('Invalid bounded GQA shape')
    out=controlled(args.out);out.mkdir(parents=True,exist_ok=False)
    g=torch.Generator().manual_seed(20261016)
    q=torch.randn((batch,qh,rows,d),generator=g).bfloat16()
    k=torch.full((batch,kvh,capacity,d),float('nan'),dtype=torch.bfloat16)
    v=torch.full_like(k,float('nan'))
    k[:,:,:keys]=torch.randn((batch,kvh,keys,d),generator=g).bfloat16()
    v[:,:,:keys]=torch.randn((batch,kvh,keys,d),generator=g).bfloat16()
    save(out/'q.bin',q);save(out/'k.bin',k);save(out/'v.bin',v)
    mapping=torch.arange(qh)//(qh//kvh)
    scores=torch.einsum('bhqd,bhkd->bhqk',q.float(),k[:,mapping,:keys].float())*d**-.5
    causal=torch.arange(keys)[None,:]<=keys-rows+torch.arange(rows)[:,None]
    scores=scores.masked_fill(~causal,float('-inf'))
    result=torch.einsum('bhqk,bhkd->bhqd',scores.softmax(-1),v[:,mapping,:keys].float())
    save(out/'ref_fp32.bin',result);save(out/'ref_bf16.bin',result.bfloat16())
    shape=[batch,qh,rows,d] if rows!=1 else [batch,qh,d] if batch!=1 else [qh,d]
    receipt=dict(family='attention',shape=shape,qheads=qh,kvheads=kvh,keys=keys,batch=batch,query_rows=rows,
        capacity=capacity,unused_kv='NaN poison; excluded from independent reference and device valid domain',
        gqa_ratio=qh//kvh,head_dim=d,seed=20261016,dtype='bf16',opmath='fp32',
        reference='independent CPU FP32 causal QK/softmax/PV on BF16 inputs; effective keys only',
        input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()})
    (out/'config.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


def prepare_fai_tile(args):
    rows,d,keys=args.rows,args.head_dim,args.keys
    if not 1<=rows<=32 or d not in (64,128) or not 1<=keys<=1088:
        raise ValueError('Invalid bounded CATLASS49 TaskBody shape')
    out=controlled(args.out);out.mkdir(parents=True,exist_ok=False)
    g=torch.Generator().manual_seed(20261016)
    q=torch.zeros((2,32,d),dtype=torch.bfloat16)
    q[:,:rows]=torch.randn((2,rows,d),generator=g).bfloat16()
    k=torch.full((2,1088,d),float('nan'),dtype=torch.bfloat16);v=torch.full_like(k,float('nan'))
    k[:,:keys]=torch.randn((2,keys,d),generator=g).bfloat16()
    v[:,:keys]=torch.randn((2,keys,d),generator=g).bfloat16()
    scores=torch.bmm(q.float(),k[:,:keys].float().transpose(1,2))*d**-.5
    result=torch.bmm(scores.softmax(-1),v[:,:keys].float())
    for name,value in [('q.bin',q),('k.bin',k),('v.bin',v),('ref_fp32.bin',result),('ref_bf16.bin',result.bfloat16())]:save(out/name,value)
    policy_path=REPO/'configs/ascend/as2_correctness_policy.json'
    receipt=dict(schema='tilemega.as2.fai_tile_inputs.v1',family='attention',logical_rows=rows,physical_rows=32,
        head_dim=d,keys=keys,capacity=1088,tasks=2,epochs=4,seed=20261016,dtype='bf16',
        reference='Independent CPU FP32 attention on two distinct BF16 Q/KV domains; no causal/prefill claim',
        unused_kv='NaN poison',padded_query_rows='zero input, physical output retained and checked',
        precision_contract_id='catlass49_bf16_qkv_p_fp32_online_statistics_v1',
        correctness_policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()})
    (out/'config.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


def compare_fai_tile(args):
    from tilemega.bench.as2_compare_native import metrics
    source=controlled(args.input);output=controlled(args.output);job=controlled(args.job)
    config=json.loads((source/'config.json').read_text());receipt=json.loads((job/'job.json').read_text())
    if (config.get('schema')!='tilemega.as2.fai_tile_inputs.v1' or config.get('physical_rows')!=32 or
            config.get('tasks')!=2 or config.get('epochs')!=4 or config.get('capacity')!=1088):
        raise ValueError('Unrecognized physical FAI component contract')
    if (receipt.get('project')!='as2' or receipt.get('status')!='passed' or receipt.get('exit_code')!=0 or
            receipt.get('cpu_test_fixture') or
            not receipt.get('pid') or Path(receipt['command'][0]).name!='as2_fai_tile' or
            receipt['command'][1:]!=list(map(str,[config['logical_rows'],config['head_dim'],config['keys'],source,output]))):
        raise ValueError('Require the completed matching single-card component job')
    for name,sha in config['input_sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=sha:raise ValueError('FAI input/reference identity changed')
    policy_path=REPO/'configs/ascend/as2_correctness_policy.json'
    if hashlib.sha256(policy_path.read_bytes()).hexdigest()!=config['correctness_policy_sha256']:
        raise ValueError('Frozen FAI numerical policy changed')
    policy=json.loads(policy_path.read_text())['body']['attention'];d=config['head_dim'];rows=config['logical_rows']
    raw=(output/'output.bin').read_bytes();size=4*2*32*d*2
    if len(raw)!=size+512 or raw[size:]!=b'\xa5'*512:raise ValueError('FAI physical extent/guard mismatch')
    control=(output/'control.bin').read_bytes()
    if len(control)!=7*512:raise ValueError('FAI completion extent mismatch')
    counts=[int.from_bytes(control[i*512:i*512+4],'little') for i in range(7)]
    if counts!=[8,8,8,0,8,8,8]:raise ValueError('FAI owner/participant completion mismatch')
    actual=torch.frombuffer(bytearray(raw[:size]),dtype=torch.bfloat16).reshape(4,2,32,d)
    fp32=read(source/'ref_fp32.bin',torch.float32,[2,32,d]);bf16=read(source/'ref_bf16.bin',torch.bfloat16,[2,32,d])
    checks=[]
    for epoch in range(4):
        for name,ref in [('fp32',fp32),('bf16',bf16)]:
            checks.append(dict(epoch=epoch,reference=name,logical=metrics(actual[epoch,:,:rows],ref[:,:rows],policy),
                physical=metrics(actual[epoch],ref,policy)))
    repeated_bitwise=all(torch.equal(actual[0].view(torch.int16),v.view(torch.int16)) for v in actual[1:])
    result=dict(status='pass_component_numeric' if repeated_bitwise and all(c[k]['passed'] for c in checks for k in ('logical','physical')) else 'fail',
        scope='one component process; no C3/model/TPOT admission',job=str(job),pid=receipt['pid'],start_ticks=receipt['start_ticks'],
        physical_device_id=receipt['physical_device_id'],binary_sha256=receipt['executable_sha256'],
        precision_contract_id=config['precision_contract_id'],checks=checks,policy=policy,
        repeated_epochs_bitwise=repeated_bitwise,completion_counts=counts,
        correctness_policy_sha256=config['correctness_policy_sha256'])
    (output/'numeric.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    return 0 if result['status']=='pass_component_numeric' else 1


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
    obs=sub.add_parser('prepare-observed-norm');obs.add_argument('--reference',type=Path,required=True);obs.add_argument('--out',type=Path,required=True)
    obs.add_argument('--layer',type=int,required=True);obs.add_argument('--rows',type=int,default=64)
    obs.add_argument('--norm',choices=('norm1','norm2','q_norm','k_norm'),default='norm2')
    attn=sub.add_parser('prepare-attention')
    for name in ('qheads','kvheads','keys','head-dim'):attn.add_argument('--'+name,type=int,required=True)
    attn.add_argument('--batch',type=int,default=1);attn.add_argument('--query-rows',type=int,default=1)
    attn.add_argument('--capacity',type=int)
    attn.add_argument('--out',type=Path,required=True)
    acheck=sub.add_parser('compare-attention');acheck.add_argument('--input',type=Path,required=True);acheck.add_argument('--output',type=Path,required=True)
    fai=sub.add_parser('prepare-fai-tile')
    for name in ('rows','head-dim','keys'):fai.add_argument('--'+name,type=int,required=True)
    fai.add_argument('--out',type=Path,required=True)
    fcheck=sub.add_parser('compare-fai-tile')
    for name in ('input','output','job'):fcheck.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.command=='prepare-gemm':return prepare(args)
    if args.command=='prepare-vector':return prepare_vector(args)
    if args.command=='prepare-observed-norm':return prepare_observed_norm(args)
    if args.command=='prepare-attention':return prepare_attention(args)
    if args.command=='prepare-fai-tile':return prepare_fai_tile(args)
    if args.command=='compare-fai-tile':return compare_fai_tile(args)
    return compare(args)


if __name__=='__main__':raise SystemExit(main())
