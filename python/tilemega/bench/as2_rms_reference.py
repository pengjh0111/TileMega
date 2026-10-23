"""Observe the independent fused RMSNorm output and rstd on saved operands."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch_npu

ROOT = Path('/home/d50064638/junhuip/TileMega/runs/as2')


def read(path, dtype, shape):
    count = __import__('math').prod(shape)
    if path.stat().st_size != count * torch.empty((), dtype=dtype).element_size():
        raise ValueError('Shape/file mismatch')
    return torch.from_file(str(path), shared=False, size=count, dtype=dtype).reshape(shape)


def save(path, tensor):
    path.write_bytes(tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes())


def prepare_capture(source,out,last_only=False):
    from tilemega.bench.as2_native import digest
    from tilemega.bench.as2_models import DENSE
    identity=json.loads((source/'result.json').read_text())
    if identity.get('system')!='torch_npu_eager' or identity.get('model_id') not in DENSE or identity.get('norm')!='fused' or not identity.get('diagnostic_capture_last_forward') or identity['batch']!=1:
        raise ValueError('Require an independent B1 fused-Norm model capture')
    weights=ROOT.parents[2]/'tilemega-as2/weights'/identity['model_id']
    manifest=json.loads((weights/'manifest.json').read_text());cfg=manifest['config']
    if digest(weights/'manifest.json')!=identity['weight_manifest_sha256'] or manifest['config_sha256']!=identity['config_sha256']:
        raise ValueError('Model/weight identity changed')
    folder=source/'last_forward';capture=json.loads((folder/'capture.json').read_text())
    observed={v['name']:v for v in capture['files']};tensors={v['name']:v for v in manifest['tensors']}
    width=cfg['hidden_size'];rows=[];gold=[];names=[];gamma=None
    def operand(name):
        record=observed[name];file=folder/record['file']
        if record['shape']!=[1,1,width] or digest(file)!=record['sha256']:
            raise ValueError('Captured full-row operand changed: '+name)
        return read(file,torch.bfloat16,[width])
    def scale(name):
        nonlocal gamma
        record=tensors[name];file=weights/record['file']
        if record['shape']!=[width] or digest(file)!=record['sha256']:raise ValueError('Norm scale changed')
        value=read(file,torch.bfloat16,[width])
        if gamma is None:gamma=value
        elif not torch.equal(gamma,value):raise ValueError('One probe case requires a shared gamma identity')
    if not last_only:
        for layer in range(min(cfg['num_hidden_layers'],16)):
            for name,param in [('norm1','input_layernorm'),('norm2','post_attention_layernorm')]:
                name=f'l{layer}.{name}';scale(f'model.layers.{layer}.{param}.weight')
                rows.append(operand(name+'.input'));gold.append(operand(name));names.append(name)
    final=cfg['num_hidden_layers']-1;scale('model.norm.weight')
    rows.append((operand(f'l{final}.norm2.input')+operand(f'l{final}.down.raw')).bfloat16())
    gold.append(operand('final.normalized'));names.append('final.normalized')
    out.mkdir();save(out/'input.bin',torch.stack(rows));save(out/'rhs.bin',gamma);save(out/'ref_bf16.bin',torch.stack(gold))
    result=dict(model_id=identity['model_id'],shape=[len(rows),width],original_shape=[1,len(rows),width],
        epsilon=cfg['rms_norm_eps'],source_reference=str(source),capture_sha256=digest(folder/'capture.json'),
        weight_manifest_sha256=identity['weight_manifest_sha256'],names=names,
        input_sha256={p.name:digest(p) for p in out.iterdir()},tilemega_pipeline=False,
        scope='observed real model inputs; final input reconstructed by BF16 residual addition and checked separately by fused reference')
    (out/'config.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


def main():
    p = argparse.ArgumentParser()
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path)
    source.add_argument('--capture', type=Path)
    p.add_argument('--last-only',action='store_true')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    for path in (args.input or args.capture, args.out):
        if path.absolute() != path.resolve() or path == ROOT or not path.is_relative_to(ROOT):
            raise ValueError('Use canonical AS-2 run paths')
    if args.capture:
        if args.out.exists():raise FileExistsError(args.out)
        return prepare_capture(args.capture,args.out,args.last_only)
    if args.last_only:raise ValueError('--last-only applies to capture preparation')
    config = json.loads((args.input / 'config.json').read_text())
    for name, expected in config['input_sha256'].items():
        if hashlib.sha256((args.input / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Changed saved operand: ' + name)
    if args.out.exists():
        raise ValueError('Use a new output path')
    x = read(args.input / 'input.bin', torch.bfloat16, config['shape'])
    gamma = read(args.input / 'rhs.bin', torch.bfloat16, [x.shape[-1]])
    gold = read(args.input / 'ref_bf16.bin', torch.bfloat16, x.shape)
    eps = config['epsilon']
    torch.npu.set_device(0)
    if torch.npu.device_count() != 1:
        raise ValueError('Use the single-device wrapper')
    args.out.mkdir()
    result = dict(input=str(args.input), input_sha256=config['input_sha256'],
                  epsilon=eps, torch_version=torch.__version__, torch_npu_version=torch_npu.__version__,
                  tilemega_pipeline=False, measurement='diagnostic_only', observations=[])
    with torch.inference_mode():
        dx, dg = x.npu(), gamma.npu()
        for label, shape in [('flat', config['shape']), ('original', config['original_shape'])]:
            y, rstd = torch_npu.npu_rms_norm(dx.reshape(shape), dg, eps)
            torch.npu.synchronize()
            cy, cr = y.cpu().reshape(x.shape), rstd.cpu().reshape(-1)
            save(args.out / (label + '_output.bin'), cy)
            save(args.out / (label + '_rstd.bin'), cr)
            # Same observed rstd, independent scalar multiplication, no output substitution.
            from_rstd = (x.float() * cr.reshape(-1, 1) * gamma.float()).bfloat16()
            double = (x.double() * (x.double().square().mean(-1, keepdim=True) + eps).rsqrt() * gamma.double()).bfloat16()
            result['observations'].append(dict(label=label, shape=shape, rstd=cr.tolist(),
                gold_bitwise_different=int((cy != gold).sum()),
                fp32_multiply_rstd_bitwise_different=int((cy != from_rstd).sum()),
                fp64_reference_bitwise_different=int((cy != double).sum()),
                changed_columns=torch.nonzero(cy != gold).tolist()))
        result['cpu_fp32_mean'] = x.float().square().mean(-1).tolist()
        result['cpu_fp64_mean'] = x.double().square().mean(-1).tolist()
        result['cpu_fp32_rstd'] = (x.float().square().mean(-1) + eps).rsqrt().tolist()
        result['cpu_fp64_rstd'] = (x.double().square().mean(-1) + eps).rsqrt().tolist()
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
