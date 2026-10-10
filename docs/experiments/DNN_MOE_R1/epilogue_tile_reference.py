#!/usr/bin/env python3
"""PyTorch tile references, including independent layout and reduction oracles."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import torch
import torch.nn.functional as F


def vector(value):
    values = torch.as_tensor(value).flatten().float().tolist()
    return struct.pack('<I', len(values)) + struct.pack('<' + 'f' * len(values), *values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20261007)
    records = []
    inventory = []
    for mode in range(16):
        m, n, image_rows, factor = 17, 61, 7, 1
        shape = (1, 1, m, n)
        if mode == 0:
            m, n, image_rows = 33, 61, 11
            shape = (3, 1, 11, n)
        if mode in (7, 8):
            n = 128
        if mode == 12:
            m, n, image_rows, factor = 12, 12, 6, 2
            shape = (2, 4, 6, 3)
        if mode == 13:
            m, n, image_rows = 30, 3, 15
            shape = (2, 3, 5, 3)
        out_n = n // 2 if mode in (7, 8) else n
        x = torch.randn(m, n) * 2
        bias = torch.randn(n).bfloat16().float() / 4
        scale = torch.randn(n).bfloat16().float()
        residual = torch.randn(m, out_n).bfloat16().float()
        width, parts = 37, 3
        h = torch.randn(m, width).bfloat16().float()
        gamma = torch.randn(out_n).bfloat16().float()
        beta = torch.randn(out_n).bfloat16().float()
        weight = torch.randn(n, width).bfloat16().float()
        norm_gamma = torch.randn(width).bfloat16().float()
        norm_beta = torch.randn(width).bfloat16().float()
        linear_bias = torch.randn(n).bfloat16().float()
        stats = torch.stack([torch.stack((segment.sum(-1), segment.square().sum(-1)), -1)
                             for segment in torch.tensor_split(h, parts, dim=1)], dim=1)
        us, vs = torch.zeros(n), torch.zeros(n)
        if mode == 0:
            y = F.relu(x + bias).bfloat16().float()
        elif mode == 1:
            y = F.relu((x + bias).bfloat16() + residual.bfloat16()).float()
        elif mode == 2:
            y = F.hardtanh(x * scale, 0, 6).bfloat16().float()
        elif mode == 3:
            y = F.gelu((x + bias).bfloat16()).float()
        elif mode == 4:
            y = F.gelu(x.bfloat16(), approximate='tanh').float()
        elif mode == 5:
            y = torch.tanh(x.bfloat16()).float()
        elif mode == 6:
            y = F.silu(x.bfloat16()).float()
        elif mode in (7, 8):
            halves = x.reshape(m, n // 32, 2, 16)
            gate, up = halves[:, :, 0].flatten(1), halves[:, :, 1].flatten(1)
            if mode == 8:
                gate = F.silu(gate.bfloat16()).float()
            y = (gate.bfloat16() * up.bfloat16()).float()
        elif mode == 9:
            folded = weight * norm_gamma
            x = F.linear(h, folded)
            us, vs = folded.sum(-1), F.linear(norm_beta, weight, linear_bias)
            y = F.linear(F.layer_norm(h, (width,), norm_gamma, norm_beta, eps=1e-5),
                         weight, linear_bias).bfloat16().float()
        elif mode == 10:
            # Here the residual tensor itself is the normalized row input.
            width = n
            residual = torch.randn(m, n).bfloat16().float()
            stats = torch.stack([torch.stack((segment.sum(-1), segment.square().sum(-1)), -1)
                                 for segment in torch.tensor_split(residual, parts, dim=1)], dim=1)
            y = (x.bfloat16() + F.layer_norm(residual, (n,), gamma, beta,
                                            eps=1e-5).bfloat16()).float()
        elif mode == 11:
            x = F.linear(h, weight * norm_gamma)
            y = F.linear(F.rms_norm(h, (width,), norm_gamma, eps=1e-5), weight).bfloat16().float()
        elif mode in (12, 13):
            y = ((x + bias).bfloat16() + residual.bfloat16()).float()
        elif mode == 14:
            y = (x + bias).bfloat16().float()
        else:
            y = x
        scatter = torch.randperm(m)
        physical = y.flatten()
        if mode == 0:
            padded = torch.full((shape[0], shape[1] + 2, shape[2] + 4, 64), -123.)
            padded[:, 1:-1, 2:-2, :n] = y.reshape(shape)
            physical = padded.flatten()
        elif mode == 12:
            before = y.reshape(2, 2, 3, 12).permute(0, 3, 1, 2)
            shuffled = F.pixel_shuffle(before, 2).permute(0, 2, 3, 1).contiguous()
            padded = torch.full((2, 4, 6, 8), -123.)
            padded[..., :3] = shuffled
            physical = padded.flatten()
            # Residual input is laid out exactly as the shuffled output.
            residual = F.pixel_shuffle(residual.reshape(2, 2, 3, 12).permute(0, 3, 1, 2), 2)
            residual = residual.permute(0, 2, 3, 1).contiguous()
            padded_residual = torch.zeros(2, 4, 6, 8)
            padded_residual[..., :3] = residual
            residual = padded_residual.flatten()
        elif mode == 13:
            physical = y.reshape(shape).permute(0, 3, 1, 2).contiguous().flatten()
            residual = residual.reshape(shape).permute(0, 3, 1, 2).contiguous().flatten()
        elif mode == 14:
            scattered = torch.empty_like(y)
            scattered[scatter] = y
            physical = scattered.flatten()
        header = (mode, m, n, *shape, factor, width, parts, image_rows)
        vectors = [x, bias, scale, residual, stats, gamma, beta, us, vs,
                   scatter, physical]
        if mode == 0:
            # TN=64, TM=16. Each tile's intersecting image segment owns one
            # sum entry; untouched entries retain the poison sentinel.
            row_stats = torch.stack((y.sum(-1), y.square().sum(-1)), -1)
            channel = torch.full((3, 3, n), -123.)
            for tile in range(3):
                begin, end = tile * 16, min(m, (tile + 1) * 16)
                for image in range(begin // image_rows, (end - 1) // image_rows + 1):
                    channel[image, tile] = y[max(begin, image * image_rows):
                                                  min(end, (image + 1) * image_rows)].sum(0)
            # stable=True supplies the lower-index tie break independently of
            # the kernel's insertion algorithm.
            order = torch.argsort(y, descending=True, stable=True)
            top = torch.gather(y, 1, order[:, :8])
            partial = torch.zeros(m, 1, 64)
            partial[:, 0, :n] = y
            vectors += [row_stats, channel, top, order[:, :8],
                        top[:, :1], order[:, :1], partial]
        records.append(struct.pack('<11I', *header) + b''.join(map(vector, vectors)))
        inventory.append(dict(mode=mode, m=m, n=n, output_shape=shape, factor=factor))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(struct.pack('<II', 0x444D4531, len(records)) + b''.join(records))
    args.out.with_suffix('.json').write_text(json.dumps(dict(
        evidence='verified', torch=torch.__version__, seed=20261007,
        fixture_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(), cases=inventory,
        scope='tile epilogue only; independent PyTorch maps and reduction oracles'), indent=2) + '\n')
    print(f'{len(records)} tile references generated', flush=True)


if __name__ == '__main__':
    main()
