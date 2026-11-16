#!/usr/bin/env python3
"""Compare a native realization with its frozen-theta reference and run C-2."""
import argparse
import gc
import json
import os
from pathlib import Path

import torch

from tilemega.cache import atomic_json, file_sha
from tilemega.serving.engine import ServingEngine


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--prompts', type=Path, required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--steps', type=int, default=1024)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise ValueError('realization checks require the physical-card guard')
    if not 1 <= a.steps <= 1024:
        p.error('steps must lie in [1,1024]')
    checkpoint = Path(os.environ['TILEMEGA_RANDOM_CHECKPOINT'])
    config = json.loads((checkpoint / 'config.json').read_text())
    rows = json.loads(a.prompts.read_text())[:a.batch]
    if len(rows) != a.batch or any(len(row) != 64 for row in rows) or any(
            type(t) is not int or not 0 <= t < config['vocab_size'] for row in rows for t in row):
        raise ValueError('realization checks require B frozen in-vocabulary ISL64 prompts')
    prompt = torch.tensor(rows, dtype=torch.int64)
    a.out.mkdir(parents=True, exist_ok=True)
    reference = None
    checks = []
    implementations = [('reference', a.reference), ('candidate', a.candidate)]
    if a.reference.resolve() == a.candidate.resolve():
        implementations = [('candidate', a.candidate)]
    for name, binary in implementations:
        for loop in [False, True]:
            with ServingEngine(checkpoint, None, binary, a.batch, prompt_len=64,
                    max_new_tokens=a.steps, mode='L1', decode_loop=loop,
                    step_events=False) as engine:
                engine.state.kv_storage.zero_()
                engine.state.tokens.fill_(-1)
                generation = engine.generate(prompt, a.steps)
                tokens = generation.tokens.tolist()
                if generation.decode_loop_used != loop:
                    raise AssertionError('requested C-2 execution mode was not used')
            del engine
            gc.collect()
            torch.cuda.empty_cache()
            path = a.out / f'{name}-loop{int(loop)}.tokens.json'
            atomic_json(path, tokens)
            if reference is None:
                reference = tokens
            passed = tokens == reference and all(
                0 <= token < config['vocab_size'] for row in tokens for token in row)
            checks.append(dict(implementation=name, decode_loop_used=loop, passed=passed,
                               binary_sha256=file_sha(binary), tokens_sha256=file_sha(path)))
            atomic_json(a.out / 'checks.json', checks)
            print(json.dumps(checks[-1]), flush=True)
            if not passed:
                raise AssertionError('realization or C-2 token mismatch; full token arrays retained')
    atomic_json(a.out / 'result.json', dict(passed=True, checks=checks, batch=a.batch,
        steps=a.steps, prompt_sha256=file_sha(a.prompts),
        checkpoint=json.loads((checkpoint / 'random_checkpoint.json').read_text())['checkpoint_sha256'],
        gpu=os.environ['TILEMEGA_DEVICE_INDEX'], evidence='verified', timing=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
