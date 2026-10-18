#!/usr/bin/env python3
"""Repeat a complete request with reset state under the shared GPU guard."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefill', type=Path, required=True)
    p.add_argument('--decode', type=Path, required=True)
    p.add_argument('--prompt-ids', type=Path, required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--steps', type=int, default=1024)
    p.add_argument('--repeats', type=int, default=8)
    p.add_argument('--mode', choices=('L1', 'L2'), default='L1')
    p.add_argument('--decode-loop', type=int, choices=(0, 1), default=1)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.batch not in (1, 2, 4, 8, 16) or min(a.steps, a.repeats) < 1:
        p.error('invalid batch, step count or repetitions')
    from tilemega.serving.baseline_common import require_guard
    require_guard()
    import torch
    from tilemega.serving.engine import ServingEngine

    a.out.mkdir(parents=True, exist_ok=True)
    ids = torch.tensor(json.loads(a.prompt_ids.read_text())[:a.batch], dtype=torch.int32)
    if tuple(ids.shape) != (a.batch, 64):
        raise ValueError('the fixed workload requires B rows of 64 prompt tokens')
    result = dict(pid=os.getpid(), model=os.environ['TILEMEGA_RANDOM_CHECKPOINT'],
                  batch=a.batch, steps=a.steps, repeats=a.repeats, mode=a.mode,
                  decode_loop=a.decode_loop, failures=[], pass_check=False,
                  binary_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for name, path in [('prefill', a.prefill), ('decode', a.decode)]},
                  timing_valid=False)
    outputs = []
    try:
        with ServingEngine(Path(result['model']), a.prefill, a.decode, a.batch,
                           prompt_len=64, max_new_tokens=a.steps, mode=a.mode,
                           decode_loop=bool(a.decode_loop), step_events=False) as engine:
            for iteration in range(a.repeats + 1):
                engine.state.kv_storage.zero_()
                engine.state.tokens.zero_()
                generated = engine.generate(ids)
                if a.steps > 1 and generated.decode_loop_used != bool(a.decode_loop):
                    raise ValueError('requested device-loop mode was not executed')
                tokens = generated.tokens.cpu()
                if tuple(tokens.shape) != (a.batch, a.steps):
                    raise ValueError('wrong generated token count')
                path = a.out / f'tokens-{iteration}.json'
                path.write_text(json.dumps(tokens.tolist()) + '\n')
                outputs.append(hashlib.sha256(path.read_bytes()).hexdigest())
                if iteration == 0:
                    reference = tokens
                elif not torch.equal(reference, tokens):
                    diff = (reference != tokens).nonzero()[0].tolist()
                    result['failures'].append(dict(iteration=iteration, first_diff=diff))
                print(json.dumps(dict(iteration=iteration, token_sha256=outputs[-1],
                                      mismatches=len(result['failures']))), flush=True)
        result.update(completed_requests=len(outputs), token_sha256=outputs,
                      pass_check=not result['failures'])
    except Exception as error:
        result.update(error=repr(error), completed_requests=len(outputs), token_sha256=outputs)
        raise
    finally:
        (a.out / 'repetition.json').write_text(json.dumps(result, indent=2) + '\n')
    return 0 if result['pass_check'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
