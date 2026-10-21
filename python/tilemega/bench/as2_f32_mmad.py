"""Check full-FP32 Cube QK arithmetic against actual eager SDPA operands."""
import argparse
import json
from pathlib import Path
import torch
from tilemega.bench.as2_models import RUNS,controlled
from tilemega.bench.as2_native import digest


def read(path,shape,dtype=torch.float32):
    size=__import__('math').prod(shape)
    if path.stat().st_size!=size*torch.empty((),dtype=dtype).element_size():raise ValueError('Wrong operand extent')
    return torch.from_file(str(path),shared=False,size=size,dtype=dtype).reshape(shape)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--dispatch',type=Path,required=True);prep.add_argument('--out',type=Path,required=True)
    prep.add_argument('--m',type=int,choices=(1,32),default=32)
    pv=sub.add_parser('prepare-pv',help='NPU softmax and independent PV reference from recorded QK')
    for name in ('input','reference','out'):pv.add_argument('--'+name,type=Path,required=True)
    sm=sub.add_parser('prepare-softmax',help='CPU-only pack of already observed independent NPU probabilities')
    for name in ('input','reference','out'):sm.add_argument('--'+name,type=Path,required=True)
    ref=sub.add_parser('reference');ref.add_argument('--input',type=Path,required=True);ref.add_argument('--out',type=Path,required=True)
    compare=sub.add_parser('compare')
    for name in ('input','reference','native','out'):compare.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);out=controlled(a.out,[RUNS])
    if out.exists():raise FileExistsError(out)
    if a.command=='prepare':
        source=controlled(a.dispatch,[RUNS]);receipt=json.loads((source/'dispatch.json').read_text());files={v['name']:v for v in receipt['files']}
        q=read(source/files['query']['file'],files['query']['shape'],torch.bfloat16)
        k=read(source/files['key']['file'],files['key']['shape'],torch.bfloat16)
        if q.shape[-1]!=128 or q.shape[-2]<32 or k.shape[-2]<64:raise ValueError('Require real head128 prefill operands')
        root=128**-.25;av=(q[0,0,:a.m].float()*root).contiguous();bv=(k[0,0,:64].float()*root).contiguous()
        out.mkdir();(out/'a.bin').write_bytes(av.numpy().tobytes());(out/'b_nk.bin').write_bytes(bv.numpy().tobytes())
        (out/'cpu_f32.bin').write_bytes((av@bv.T).numpy().tobytes())
        (out/'cpu_f64.bin').write_bytes((av.double()@bv.double().T).float().numpy().tobytes())
        identity=dict(schema='tilemega.as2.f32_mmad_inputs.v1',dispatch=str(source),dispatch_sha256=digest(source/'dispatch.json'),
            shape_mnk=[a.m,64,128],mode=0,dtype='fp32',weights_or_answers_specialized=False,
            operands={n:digest(out/n) for n in ('a.bin','b_nk.bin')})
        (out/'inputs.json').write_text(json.dumps(identity,indent=2)+'\n');print(json.dumps(identity));return
    source=controlled(a.input,[RUNS]);identity=json.loads((source/'inputs.json').read_text())
    if any(digest(source/n)!=sha for n,sha in identity['operands'].items()):raise ValueError('Operand identity changed')
    m,n,k=identity['shape_mnk'];mode=identity.get('mode',0)
    if a.command=='prepare-softmax':
        reference=controlled(a.reference,[RUNS]);pv_identity=json.loads((reference/'inputs.json').read_text())
        if mode or pv_identity['dispatch']!=identity['dispatch'] or pv_identity['shape_mnk']!=[m,128,1088] or pv_identity['causal_query_positions']!=list(range(m)):
            raise ValueError('Wrong observed QK/softmax provenance')
        qk=controlled(Path(pv_identity['upstream_qk_reference']),[RUNS]);ref=json.loads((qk/'reference.json').read_text())
        if ref['input_identity']!=digest(source/'inputs.json') or digest(reference/'a.bin')!=pv_identity['operands']['a.bin']:
            raise ValueError('Observed independent probabilities or scores changed')
        scores=read(qk/'c.bin',(m,64));out.mkdir();(out/'a.bin').write_bytes(scores.numpy().tobytes())
        full=torch.full((m,1088),torch.finfo(torch.bfloat16).min);full[:,:64]=scores
        full.masked_fill_(torch.arange(1088)[None,:]>torch.arange(m)[:,None],torch.finfo(torch.bfloat16).min)
        (out/'cpu_f32.bin').write_bytes(full.softmax(-1).numpy().tobytes());(out/'cpu_f64.bin').write_bytes(full.double().softmax(-1).float().numpy().tobytes())
        record=dict(schema='tilemega.as2.f32_mmad_inputs.v1',shape_mnk=[m,1088,64],mode=2,upstream_pv=str(reference),
            operands={'a.bin':digest(out/'a.bin')})
        (out/'inputs.json').write_text(json.dumps(record,indent=2)+'\n');(out/'c.bin').write_bytes((reference/'a.bin').read_bytes())
        (out/'reference.json').write_text(json.dumps(dict(input_identity=digest(out/'inputs.json'),hf32=False,
            implementation='independent torch_npu eager FP32 softmax',output_sha256=digest(out/'c.bin')),indent=2)+'\n');return
    if a.command=='prepare-pv':
        import torch_npu
        torch.npu.set_device(0)
        if torch.npu.matmul.allow_hf32:raise ValueError('Require full FP32 math')
        reference=controlled(a.reference,[RUNS]);ref=json.loads((reference/'reference.json').read_text())
        if ref['input_identity']!=digest(source/'inputs.json') or mode or n!=64 or k!=128:raise ValueError('Require the independent head128 QK reference')
        dispatch=Path(identity['dispatch']);records=json.loads((dispatch/'dispatch.json').read_text());v=next(x for x in records['files'] if x['name']=='value')
        values=read(dispatch/v['file'],v['shape'],torch.bfloat16)[0,0].float().contiguous()
        if values.shape!=(1088,128):raise ValueError('Require the actual static KV capacity')
        with torch.inference_mode():
            scores=torch.full((m,1088),torch.finfo(torch.bfloat16).min,dtype=torch.float32,device='npu:0')
            scores[:,:64]=read(reference/'c.bin',(m,64)).to('npu:0')
            mask=torch.arange(1088,device='npu:0')[None,:]>torch.arange(m,device='npu:0')[:,None]
            probability=scores.masked_fill(mask,torch.finfo(torch.bfloat16).min).softmax(-1)
            expected=(probability@values.to('npu:0')).cpu().contiguous();av=probability.cpu().contiguous()
        out.mkdir();(out/'a.bin').write_bytes(av.numpy().tobytes());(out/'b_kn.bin').write_bytes(values.numpy().tobytes())
        (out/'cpu_f32.bin').write_bytes((av@values).numpy().tobytes());(out/'cpu_f64.bin').write_bytes((av.double()@values.double()).float().numpy().tobytes())
        record=dict(schema='tilemega.as2.f32_mmad_inputs.v1',dispatch=identity['dispatch'],shape_mnk=[m,128,1088],mode=1,dtype='fp32',
            valid_kv=64,causal_query_positions=list(range(m)),upstream_qk_reference=str(reference),
            operands={name:digest(out/name) for name in ('a.bin','b_kn.bin')})
        (out/'inputs.json').write_text(json.dumps(record,indent=2)+'\n');(out/'c.bin').write_bytes(expected.numpy().tobytes())
        (out/'reference.json').write_text(json.dumps(dict(input_identity=digest(out/'inputs.json'),hf32=False,
            implementation='independent torch_npu eager FP32 capacity-softmax/PV',output_sha256=digest(out/'c.bin')),indent=2)+'\n');return
    if a.command=='reference':
        if mode==2:raise ValueError('Use the independently recorded softmax probabilities')
        import torch_npu
        torch.npu.set_device(0)
        if torch.npu.matmul.allow_hf32:raise ValueError('Reference must retain full FP32 matmul')
        with torch.inference_mode():
            av=read(source/'a.bin',(m,k)).to('npu:0');bv=read(source/('b_kn.bin' if mode else 'b_nk.bin'),(k,n) if mode else (n,k)).to('npu:0')
            value=(av@(bv if mode else bv.T)).cpu().contiguous()
        torch.npu.synchronize();out.mkdir();(out/'c.bin').write_bytes(value.numpy().tobytes())
        (out/'reference.json').write_text(json.dumps(dict(input_identity=digest(source/'inputs.json'),hf32=False,
            implementation='independent torch_npu eager FP32 matmul',output_sha256=digest(out/'c.bin')),indent=2)+'\n');return
    reference,native=(controlled(v,[RUNS]) for v in (a.reference,a.native))
    ref_identity=json.loads((reference/'reference.json').read_text())
    if ref_identity['input_identity']!=digest(source/'inputs.json'):raise ValueError('Reference/input identity mismatch')
    result=read(native/'c.bin',(m,n));rows=[]
    for name,path in [('npu_fp32',reference/'c.bin'),('cpu_fp32',source/'cpu_f32.bin'),('cpu_fp64_cast',source/'cpu_f64.bin')]:
        expected=read(path,(m,n));diff=(result-expected).abs()
        rows.append(dict(reference=name,elements=result.numel(),bitwise_different=int((result!=expected).sum()),
            bf16_cast_different=int((result.bfloat16()!=expected.bfloat16()).sum()),max_abs=float(diff.max()),
            nrmse=float(diff.square().mean().sqrt()/expected.square().mean().sqrt()),nonfinite=int((~torch.isfinite(result)).sum())))
    receipt=dict(schema='tilemega.as2.f32_mmad_numeric.v1',status='pass' if rows[0]['max_abs']<=1e-5 and not rows[0]['nonfinite'] else 'fail',
        scope='diagnostic FP32 attention component; not integrated LLM Body or model Gate',
        operation=['QK','PV','softmax'][mode],shape_mnk=[m,n,k],hf32=False,absolute_limit=1e-5,rows=rows)
    out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
