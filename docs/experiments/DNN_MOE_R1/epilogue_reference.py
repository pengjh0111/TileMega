#!/usr/bin/env python3
"""Generate scalar epilogue references from PyTorch operations."""
import argparse
import json
from pathlib import Path
import struct
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20261007)
    x = torch.linspace(-8, 8, 64)
    partner = torch.randn(64).bfloat16().float()
    residual = torch.randn(64).bfloat16().float()
    bias, scale = 0.125, 0.75
    rows = []
    for mode in range(16):
        p = dict(bias=bias, scale=scale, residual_scale=1., mean=0., rstd=1.,
                 u=0., v=0., gamma=1., beta=0.)
        value = x.clone()
        if mode == 0:
            reference = F.relu(x + bias).bfloat16().float()
        elif mode == 1:
            reference = F.hardtanh(x * scale, 0, 6).bfloat16().float()
        elif mode in (2, 3):
            reference = F.gelu(x.bfloat16(), approximate='none' if mode == 2 else 'tanh').float()
        elif mode == 4:
            reference = torch.tanh(x.bfloat16()).float()
        elif mode == 5:
            reference = F.silu(x.bfloat16()).float()
        elif mode == 6:
            reference = (x.bfloat16() * partner.bfloat16()).float()
        elif mode == 7:
            reference = (F.silu(x.bfloat16()) * partner.bfloat16()).float()
        elif mode == 8:
            reference = (x.bfloat16() + residual.bfloat16()).float()
        elif mode == 9:
            p['residual_scale'] = .375
            reference = (x * p['residual_scale'] + residual).bfloat16().float()
        elif mode in (10, 11, 12):
            h = torch.randn(37).bfloat16().float()
            gamma = torch.randn(37).bfloat16().float()
            beta = torch.randn(37).bfloat16().float()
            weight = torch.randn(64, 37).bfloat16().float()
            linear_bias = torch.randn(64).bfloat16().float()
            mean = h.mean()
            rstd = torch.rsqrt((h - mean).square().mean() + 1e-5)
            if mode == 10:
                rms = F.rms_norm(h, (37,), gamma, eps=1e-5)
                value = F.linear(h, weight * gamma)
                p['rstd'] = float(torch.rsqrt(h.square().mean() + 1e-5))
                reference = F.linear(rms, weight).bfloat16().float()
            elif mode == 11:
                normalized = F.layer_norm(h, (37,), gamma, beta, eps=1e-5)
                folded = weight * gamma
                value = F.linear(h, folded)
                us = folded.sum(dim=1)
                vs = F.linear(beta, weight, linear_bias)
                p['mean'], p['rstd'] = float(mean), float(rstd)
                reference = F.linear(normalized, weight, linear_bias).bfloat16().float()
            else:
                normalized = F.layer_norm(h, (37,), gamma, beta, eps=1e-5).bfloat16()
                indices = torch.arange(64) % 37
                residual = h[indices]
                p['mean'], p['rstd'] = float(mean), float(rstd)
                reference = (x.bfloat16() + normalized[indices]).float()
        elif mode == 13:
            reference = F.relu((x + bias).bfloat16() + residual.bfloat16()).float()
        elif mode == 14:
            reference = F.gelu((x + bias).bfloat16()).float()
        else:
            reference = ((x + bias).bfloat16() * (partner + bias).bfloat16()).float()
        for i in range(64):
            context = dict(p, residual=float(residual[i]))
            if mode == 11:
                context.update(u=float(us[i]), v=float(vs[i]))
            if mode == 12:
                context.update(gamma=float(gamma[indices[i]]), beta=float(beta[indices[i]]))
            fields = [context[k] for k in ('bias', 'scale', 'residual', 'residual_scale',
                      'mean', 'rstd', 'u', 'v', 'gamma', 'beta')]
            rows.append(struct.pack('<I13f', mode, float(value[i]), float(partner[i]),
                                    float(reference[i]), *fields))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(b''.join(rows))
    args.out.with_suffix('.json').write_text(json.dumps(dict(
        torch=torch.__version__, seed=20261007, records=len(rows), variants=16,
        scope='epilogue value functions; no tile or synchronization claim'), indent=2) + '\n')
    print(f'{len(rows)} references generated', flush=True)


if __name__ == '__main__':
    main()
