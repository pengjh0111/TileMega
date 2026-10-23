"""Replay an observed prefill down projection/residual/Norm independently."""
import argparse
import json
from pathlib import Path
import struct
import torch
import torch_npu
from tilemega.bench.as2_models import RUNS,WORK,DENSE,controlled
from tilemega.bench.as2_native import digest


def read(path,shape):
    n=__import__('math').prod(shape)
    if path.stat().st_size!=n*2:raise ValueError('Observed subgraph extent mismatch')
    return torch.from_file(str(path),shared=False,size=n,dtype=torch.bfloat16).reshape(shape)


def save(path,value):
    path.write_bytes(value.cpu().contiguous().view(torch.uint8).numpy().tobytes())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','native','reference','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    for name in ('plan','native','reference','out'):setattr(a,name,controlled(getattr(a,name),[RUNS]))
    if a.out.exists():raise FileExistsError(a.out)
    ref=json.loads((a.reference/'reference.json').read_text());imported=json.loads((a.plan.parent/'import.json').read_text())
    if ref['model_id'] not in DENSE or ref['seq']!=64 or ref['batch']!=1 or ref['past'] or ref['implementation']!='pinned_transformers_sdpa_fused_norm' or imported['model_id']!=ref['model_id']:
        raise ValueError('Require independent fused-Norm B1 prefill observations')
    root=WORK/'weights'/ref['model_id'];manifest=json.loads((root/'manifest.json').read_text());cfg=manifest['config']
    if digest(root/'manifest.json')!=ref['weight_manifest_sha256'] or imported['config_sha256']!=ref['config_sha256']:
        raise ValueError('Subgraph model/weight identity mismatch')
    files={v['name']:v for v in ref['files']};weights={v['name']:v for v in manifest['tensors']}
    stages=list(struct.iter_unpack('<17Q16I',(a.plan/'stages.bin').read_bytes()))
    buffers={v['offset_bytes']:v for v in json.loads((a.plan/'buffers.json').read_text())['buffers']}
    down=[v for v in stages if v[17]==0 and v[18]==1][1];norm=[v for v in stages if v[17]==1][2]
    if down[2]!=down[3] or norm[0]!=down[3] or sum(s[17]==0 and s[16]==down[16] for s in stages)!=1:
        raise ValueError('Unproven residual alias/raw-workspace lifetime')
    h,f=cfg['hidden_size'],cfg['intermediate_size']
    def observed(name,shape):
        row=files[name];path=a.reference/row['file']
        if digest(path)!=row['sha256']:raise ValueError('Independent observation changed')
        return read(path,shape)
    def parameter(name,shape):
        row=weights[name];path=root/row['file']
        if digest(path)!=row['sha256']:raise ValueError('Canonical weight changed')
        return read(path,shape)
    record=next(v for v in ref['diagnostic_norm_inputs'] if v['layer']==0)
    residual_path=a.reference/record['file']
    if digest(residual_path)!=record['sha256']:raise ValueError('Residual observation changed')
    residual=read(residual_path,(1,64,h));act=observed('l0.act',(1,64,f))
    weight=parameter('model.layers.0.mlp.down_proj.weight',(h,f));gamma=parameter('model.layers.1.input_layernorm.weight',(h,))
    checks=[];a.out.mkdir();torch.set_num_threads(1);torch.npu.set_device(0)
    if torch.npu.device_count()!=1 or torch.npu.matmul.allow_hf32:raise ValueError('Require single-device wrapper and full FP32 math')
    with torch.inference_mode():
        da,dw=act.npu(),weight.npu();raw=torch.nn.functional.linear(da,dw)
        fp32=torch.nn.functional.linear(da.float(),dw.float()).bfloat16()
        hidden=residual.npu()+raw;normalized=torch_npu.npu_rms_norm(hidden,gamma.npu(),cfg['rms_norm_eps'])[0]
        def compare(name,value,path):
            x=value.cpu();y=read(path,x.shape);delta=(x.float()-y.float()).abs()
            checks.append(dict(name=name,bitwise_different=int((x!=y).sum()),max_abs=float(delta.max())))
        compare('act',act,a.native/f"{buffers[down[0]]['id']}.bin")
        compare('down_raw',raw,a.native/f"{buffers[down[16]]['id']}.bin")
        compare('down_fp32_then_bf16',fp32,a.native/f"{buffers[down[16]]['id']}.bin")
        # Residual aliases are overwritten in place by this and later layers.
        # A final arena dump is not an observation of their historical inputs.
        for name in ('residual_before_ffn','next_norm_input'):
            checks.append(dict(name=name,status='unavailable_reused_arena',
                reason='historical lifetime ended before final download; no numerical claim'))
        compare('next_norm_output',normalized,a.native/f"{buffers[norm[2]]['id']}.bin")
        gold=observed('l1.norm1',(1,64,h));checks.append(dict(name='replay_vs_independent_model',bitwise_different=int((normalized.cpu()!=gold).sum())))
        save(a.out/'down_bf16.bin',raw);save(a.out/'down_fp32_then_bf16.bin',fp32)
        case=a.out/'norm_input';case.mkdir();save(case/'input.bin',hidden.reshape(64,h));save(case/'rhs.bin',gamma);save(case/'ref_bf16.bin',normalized.reshape(64,h))
    (case/'config.json').write_text(json.dumps(dict(shape=[64,h],original_shape=[1,64,h],epsilon=cfg['rms_norm_eps'],
        input_sha256={p.name:digest(p) for p in case.iterdir()},source_reference=str(a.reference),scope='independent observed down/residual replay'),indent=2)+'\n')
    result=dict(tilemega_pipeline=False,scope='independent observed subgraph replay; not model/performance admission',model_id=ref['model_id'],
        reference=str(a.reference),native=str(a.native),checks=checks,torch_version=torch.__version__,torch_npu_version=torch_npu.__version__)
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
