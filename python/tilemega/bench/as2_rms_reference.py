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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    for path in (args.input, args.out):
        if path.absolute() != path.resolve() or path == ROOT or not path.is_relative_to(ROOT):
            raise ValueError('Use canonical AS-2 run paths')
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
