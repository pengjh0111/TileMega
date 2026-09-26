#!/usr/bin/env python3
"""Diagnostic page-stall trace for one static-batch serving request."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import torch

from tilemega.serving.engine import ServingEngine
from tilemega.serving.measure import _exclusive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--prefill-so', type=Path, required=True)
    parser.add_argument('--decode-so', type=Path, required=True)
    parser.add_argument('--prompt-ids', type=Path, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--steps', type=int, default=1024)
    parser.add_argument('--mode', choices=('L1', 'L2'), default='L2')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ['TILEMEGA_PAGE_TRACE_OUT'] = str((args.out/'page_trace.tsv').resolve())
    torch.cuda.set_device(0)
    prompts = torch.tensor(json.loads(args.prompt_ids.read_text())[:args.batch], dtype=torch.int32)
    with ServingEngine(args.model, args.prefill_so, args.decode_so, args.batch,
                       mode=args.mode) as engine:
        if not _exclusive(args.out/'guard.jsonl', 'before', True):
            raise RuntimeError('GPU is occupied before the diagnostic request')
        result = engine.generate(prompts, args.steps)
        if not _exclusive(args.out/'guard.jsonl', 'after', False):
            raise RuntimeError('GPU was occupied during the diagnostic request')
        report = dict(model=str(args.model), batch=args.batch, steps=args.steps,
                      mode=args.mode, e2e_ms=result.e2e_ms, ttft_ms=result.ttft_ms,
                      step_ms=result.step_ms, diagnostic=True)
        (args.out/'request.json').write_text(json.dumps(report, indent=2)+'\n')
    trace = args.out/'page_trace.tsv'
    if not trace.exists() or trace.stat().st_size == 0:
        raise RuntimeError('trace-enabled decode library did not write page_trace.tsv')
    print(json.dumps({key: value for key, value in report.items() if key != 'step_ms'}))


if __name__ == '__main__':
    main()
