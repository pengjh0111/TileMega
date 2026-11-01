"""Run a source-bound TP token check; timing acceptance is deliberately absent."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from .measure_tp import capture_harness, require_group
from .tp_engine import TpServingEngine
from .workload import validate_prompt_ids


def main():
    parser = argparse.ArgumentParser()
    for name in ('model', 'decode-so', 'prompt-ids', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--batch', type=int, choices=(1, 2, 4, 8, 16), required=True)
    parser.add_argument('--devices', required=True)
    parser.add_argument('--steps', type=int, choices=(512, 1024), default=1024)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    physical = require_group()
    args.out.mkdir(parents=True, exist_ok=False)
    root, harness = capture_harness(args.out)
    ids = json.loads(args.prompt_ids.read_text())
    prompt_len = validate_prompt_ids(ids, args.batch)
    with TpServingEngine(args.model, args.decode_so, args.batch, prompt_len=prompt_len,
                         max_new_tokens=args.steps, devices=tuple(map(int, args.devices.split(',')))) as engine:
        try:
            result = engine.generate(torch.tensor(ids[:args.batch], dtype=torch.int32))
        except BaseException:
            (args.out / 'watchdog.json').write_text(json.dumps(engine.last_failure, indent=2) + '\n')
            raise
        tokens = result.tokens.tolist()
        token_file = args.out / 'tokens.json'
        token_file.write_text(json.dumps(tokens, separators=(',', ':')) + '\n')
        matches = args.reference is None or tokens == json.loads(args.reference.read_text())
        check = dict(tp=engine.tp, gpu_indices=physical, batch=args.batch, steps=args.steps,
                     pass_=matches, rank_tokens_identical=True, PID=os.getpid(),
                     token_sha256=hashlib.sha256(token_file.read_bytes()).hexdigest(),
                     execution_identity=engine.ranks[0]['plan'].execution_identity(1, True),
                     reference=str(args.reference) if args.reference else None,
                     host_source_identity=harness, timing_eligible=False,
                     rank_kernel_ms_diagnostic=result.rank_kernel_ms)
        check['pass'] = check.pop('pass_')
        for name, expected in harness['source_files'].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError('host source changed during TP check')
        (args.out / 'check.json').write_text(json.dumps(check, indent=2) + '\n')
        if not matches:
            raise AssertionError('TP C-2 / litmus token sequence differs from reference')
    print(json.dumps(check), flush=True)


if __name__ == '__main__':
    main()
