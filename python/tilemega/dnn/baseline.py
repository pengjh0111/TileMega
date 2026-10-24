"""Independent torch.compile baseline for the five DNN workloads (MX-4 baselines).

One process times one model at every batch size: BF16 eval, random weights
(the model builders of tilemega.dnn.export with structure_only=True, so no
checkpoint download), the forward compiled with mode="reduce-overhead"
(fullgraph=True, falling back to fullgraph=False with the error recorded),
then `--warmup` untimed runs (compilation and graph capture) and `--repeats`
timed runs of `--iterations` forwards each, timed with device events. Eager
BF16 is timed the same way as the reference row. Inputs: images 224x224
(ResNet-18, MobileNetV1/V2), 256x256 (NAFNet), token IDs of length 128 (BERT).

Usage: python -m tilemega.dnn.baseline --model resnet18 --out DIR [--batch all]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
from pathlib import Path

import torch

from .export import MODELS, model, versions

BATCHES = (1, 2, 4, 8, 16)


def inputs(name, batch, device):
    if name == 'bert':
        ids = torch.randint(1000, 20000, (batch, 128), device=device, dtype=torch.int64)
        return (), dict(input_ids=ids, token_type_ids=torch.zeros_like(ids))
    side = 256 if name == 'nafnet' else 224
    return (torch.randn((batch, 3, side, side), device=device, dtype=torch.bfloat16),), {}


def time_runs(fn, args, kwargs, warmup, repeats, iterations):
    for _ in range(warmup):
        for _ in range(iterations):
            fn(*args, **kwargs)
    torch.cuda.synchronize()
    per_forward = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn(*args, **kwargs)
        end.record()
        end.synchronize()
        per_forward.append(start.elapsed_time(end) / iterations)
    return per_forward


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--batch', default='all')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--iterations', type=int, default=50)
    p.add_argument('--mode', default='reduce-overhead')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261024)
    device = torch.device('cuda')
    module, source = model(a.model, structure_only=True)
    module = module.to(device)
    batches = BATCHES if a.batch == 'all' else (int(a.batch),)
    report = dict(model=a.model, source=source, versions=versions(), mode=a.mode, warmup=a.warmup,
                  repeats=a.repeats, iterations=a.iterations, weights='random (structure_only)', rows=[])
    compiled, fullgraph, compile_error = None, True, None
    try:
        compiled = torch.compile(module, mode=a.mode, fullgraph=True)
    except Exception:  # noqa: BLE001 -- recorded, then the partial graph is tried
        compile_error = traceback.format_exc(limit=3)
    for batch in batches:
        args, kwargs = inputs(a.model, batch, device)
        row = dict(batch=batch)
        with torch.inference_mode():
            eager = time_runs(module, args, kwargs, a.warmup, a.repeats, a.iterations)
            row['eager_ms'] = statistics.median(eager)
            row['eager_runs_ms'] = eager
            begin = time.monotonic()
            try:
                runs = time_runs(compiled, args, kwargs, a.warmup, a.repeats, a.iterations)
            except Exception:  # noqa: BLE001
                if fullgraph:
                    compile_error = (compile_error or '') + traceback.format_exc(limit=3)
                    fullgraph = False
                    torch._dynamo.reset()
                    compiled = torch.compile(module, mode=a.mode, fullgraph=False)
                runs = time_runs(compiled, args, kwargs, a.warmup, a.repeats, a.iterations)
            row['compile_warmup_s'] = time.monotonic() - begin
            row['torchcompile_ms'] = statistics.median(runs)
            row['torchcompile_runs_ms'] = runs
            row['fullgraph'] = fullgraph
        row['items_per_s'] = batch * 1000.0 / row['torchcompile_ms']
        report['rows'].append(row)
        print(json.dumps(dict(model=a.model, batch=batch, eager_ms=round(row['eager_ms'], 4),
                              torchcompile_ms=round(row['torchcompile_ms'], 4), fullgraph=fullgraph)),
              flush=True)
    report['compile_error'] = compile_error
    (a.out / 'measurements.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
