#!/usr/bin/env python3
"""Vendor GEMM ceiling for decode linear layers on the C550 (MX3-7 item 1).

vLLM-MetaX runs an unquantized decode linear layer as torch.nn.functional.linear
(UnquantizedLinearMethod), which on this torch dispatches to the vendor BLAS
(mcBLAS through the cuBLAS shim). This times exactly that call on each
development cell's GEMM shapes, so the result is "how fast this shape runs on
this device through the stack the baseline uses", to set beside the stage
ledger's TileMega body time for the same class.

Each shape is timed with CUDA events over a back-to-back loop after warm-up;
weights are distinct per iteration so the L2 cannot hold them (the decode step
reads every weight once). Output: one CSV row per (model, batch, class).
"""
import argparse
import csv
import statistics
from pathlib import Path

import torch

MODELS = {
    # hidden, q width, kv width, intermediate, vocab
    'qwen3_0_6b': (1024, 2048, 1024, 3072, 151936),
    'qwen3_1_7b': (2048, 2048, 1024, 6144, 151936),
    'llama3_2_1b': (2048, 2048, 512, 8192, 128256),
    'llama3_1_8b': (4096, 4096, 1024, 14336, 128256),
}
PEAK = 1.61e12


def shapes(model):
    h, q, kv, inter, vocab = MODELS[model]
    return [('qkv', q + 2 * kv, h), ('o', h, q), ('gate_up', 2 * inter, h),
            ('down', h, inter), ('lm_head', vocab, h)]


def time_linear(m, n, k, copies, iters, warmup):
    dev = torch.device('cuda')
    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
    for i in range(warmup):
        torch.nn.functional.linear(x, ws[i % copies])
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for i in range(iters):
            torch.nn.functional.linear(x, ws[i % copies])
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1e3 / iters)
    return statistics.median(samples), min(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cells', default='qwen3_0_6b:1,qwen3_0_6b:16,llama3_2_1b:1,llama3_1_8b:1')
    ap.add_argument('--iters', type=int, default=50)
    a = ap.parse_args()
    rows = []
    for cell in a.cells.split(','):
        model, batch = cell.split(':')
        batch = int(batch)
        for name, n, k in shapes(model):
            weight_bytes = 2 * n * k
            # Enough copies that the working set is well past the 8 MiB L2 knee.
            copies = max(2, -(-48 * 2**20 // weight_bytes))
            copies = min(copies, max(2, 6 * 2**30 // weight_bytes))
            med, best = time_linear(batch, n, k, copies, a.iters, warmup=10)
            gbps = weight_bytes / (med * 1e-6) / 1e9
            rows.append(dict(model=model, batch=batch, cls=name, n=n, k=k,
                             weight_mib=round(weight_bytes / 2**20, 2),
                             vendor_us=round(med, 2), vendor_best_us=round(best, 2),
                             bound_us=round(weight_bytes / PEAK * 1e6, 2),
                             gbps=round(gbps, 1), pct_peak=round(100 * gbps * 1e9 / PEAK, 1)))
            print('%-12s B%-3d %-8s N=%6d K=%5d  %8.2f us  bound %7.2f us  %6.1f GB/s (%4.1f%%)' % (
                model, batch, name, n, k, med, rows[-1]['bound_us'], gbps, rows[-1]['pct_peak']))
    with a.out.open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
