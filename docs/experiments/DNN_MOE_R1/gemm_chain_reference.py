#!/usr/bin/env python3
"""Independent PyTorch references for GEMM, mapped chains and split-K."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import torch
import torch.nn.functional as F


def vector(value):
    values = value.flatten().float().tolist()
    return struct.pack('<I', len(values)) + struct.pack('<' + 'f' * len(values), *values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20261007)
    records, cases = [], []
    for mode, (m, n, stride, offset) in enumerate(((33, 61, 2, 3), (17, 128, 1, 2),
                                                  (12, 12, 3, 1), (30, 3, 2, 1))):
        k = 192
        a = (torch.randn(offset + m * stride, k) / 2).bfloat16().float()
        b = (torch.randn(n, k) / 2).bfloat16().float()
        bias = (torch.randn(n) / 4).bfloat16().float()
        scale = torch.randn(n).bfloat16().float()
        x = F.linear(a[offset:offset + m * stride:stride], b)
        residual = torch.randn(m, n).bfloat16().float()
        if mode == 0:
            y = (x + bias).relu().bfloat16().float()
            output = torch.full((3, 3, 15, 64), -123.)
            output[:, 1, 2:13, :n] = y.reshape(3, 11, n)
        elif mode == 1:
            pairs = x.bfloat16().reshape(m, -1, 2, 16)
            gates = F.silu(pairs[:, :, 0]).bfloat16()
            y = (gates * pairs[:, :, 1]).float().reshape(m, n // 2)
            output = y
        elif mode == 2:
            y = ((x + bias).bfloat16() + residual.bfloat16()).float()
            def shuffle(value):
                return F.pixel_shuffle(value.reshape(2, 2, 3, 12).permute(0, 3, 1, 2), 2)
            output = torch.full((2, 4, 6, 8), -123.)
            output[..., :3] = shuffle(y).permute(0, 2, 3, 1)
            r = torch.zeros_like(output)
            r[..., :3] = shuffle(residual).permute(0, 2, 3, 1)
            residual = r
        else:
            y = (x * scale).clamp(0, 6).bfloat16().float()
            output = y.reshape(2, 3, 5, 3).permute(0, 3, 1, 2).contiguous()
        columns = 32 if mode == 1 else 64
        stats = torch.stack([torch.stack((part.sum(-1), part.square().sum(-1)), -1)
                             for part in y.split(columns, -1)], 1)
        records.append(struct.pack('<6I', mode, m, n, k, stride, offset) +
                       b''.join(map(vector, (a, b, bias, scale, residual, output, stats))))
        cases.append(dict(mode=mode, m=m, n=n, k=k, row_stride=stride, row_offset=offset))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(struct.pack('<II', 0x444D4731, len(records)) + b''.join(records))
    args.out.with_suffix('.json').write_text(json.dumps(dict(evidence='verified',
        torch=torch.__version__, seed=20261007, cases=cases,
        sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(),
        reference='FP32 PyTorch linear on BF16 inputs and weights; declared chain rounding'), indent=2) + '\n')


if __name__ == '__main__':
    main()
