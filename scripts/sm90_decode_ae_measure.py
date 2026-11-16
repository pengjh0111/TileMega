#!/usr/bin/env python3
"""Time one fresh A/B/C/E process with the SM90R3 request-wall TPOT protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from sm90d1_common import ROOT


def main():
    from tilemega.serving.baseline_common import guarded_main
    guarded_main(run)


def run():
    from tilemega.serving.baseline_common import measure, require_guard
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--arm', choices=('A', 'B', 'C', 'E'), required=True)
    parser.add_argument('--fresh', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require_guard()
    spec = json.loads(args.spec.read_text())
    arm = spec['arms'][args.arm]
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for name in ('decode', 'prefill'):
        if digest(arm[name]) != arm[name + '_sha256']:
            raise ValueError('artifact changed after freezing: ' + name)
    checkpoint = Path(os.environ['TILEMEGA_RANDOM_CHECKPOINT'])
    prompt_path = ROOT / 'configs/prompts/qwen_isl64.json'
    prompts = json.loads(prompt_path.read_text())
    if len(prompts[0]) != 64:
        raise ValueError('A-E requires ISL=64')
    import torch
    from tilemega.serving.engine import ServingEngine
    frozen = torch.tensor(prompts[:1], dtype=torch.int32)
    options = argparse.Namespace(model=checkpoint, batch='1', max_tokens=1024, warmup=1,
        repeats=3, round_index=args.fresh, load_format='auto', policy=None, tp=1, out=args.out)
    metadata = dict(arm=args.arm, protocol='SM90R3 P-FINAL statistics; performance only',
        spec=str(args.spec.resolve()), spec_sha256=digest(args.spec),
        decode=arm['decode'], decode_sha256=arm['decode_sha256'],
        prefill=arm['prefill'], prefill_sha256=arm['prefill_sha256'],
        mode='L1', decode_loop=1, prefill_mode='L1', dtype='BF16',
        driver_sha256=digest(__file__), checks=dict(c1=False, c2=False, hf=False,
            token_logit_comparison=False, profiling=False), executions=[])
    with ServingEngine(checkpoint, arm['prefill'], arm['decode'], 1, prompt_len=64,
            max_new_tokens=1024, mode='L1', decode_loop=True, step_events=False,
            prefill_mode='L1') as engine:
        def generate(_, count):
            result = engine.generate(frozen, count)
            metadata['executions'].append(dict(tokens=count, decode_loop_used=result.decode_loop_used))
            return result.tokens.tolist()
        # The common harness validates only output shape, without comparing token values.
        measure(options, prompts, 'TileMega', generate, metadata, synchronize=torch.cuda.synchronize)


if __name__ == '__main__':
    main()
