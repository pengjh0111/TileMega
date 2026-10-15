"""Guarded end-to-end runs and executor checks for an immutable plan pair."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .baseline_common import guarded_main, measure, require_guard, validate_prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--prefill', type=Path, required=True)
    parser.add_argument('--decode', type=Path, required=True)
    parser.add_argument('--prompt-ids', type=Path, required=True)
    parser.add_argument('--batch', choices=('1', '2', '4', '8', '16'), required=True)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--round-index', type=int, default=0)
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('L1', 'L2'), default='L2')
    parser.add_argument('--decode-loop', type=int, choices=(0, 1), default=0)
    parser.add_argument('--check-executors', action='store_true')
    parser.add_argument('--hf-check', action='store_true')
    args = parser.parse_args()
    require_guard()
    if args.model is None:
        checkpoint = os.environ.get('TILEMEGA_RANDOM_CHECKPOINT')
        if not checkpoint:
            parser.error('--model or TILEMEGA_RANDOM_CHECKPOINT is required')
        args.model = Path(checkpoint)
    if min(args.max_tokens, args.warmup, args.repeats) < 1:
        parser.error('token and repetition counts must be positive')
    args.load_format = 'auto'
    args.out.mkdir(parents=True, exist_ok=True)
    ids = json.loads(args.prompt_ids.read_text())
    prompt_len = validate_prompts(ids, int(args.batch))
    import torch
    from .engine import ServingEngine
    from .execution import execution_combinations
    manifests = [json.loads(Path(str(path)+'.plan.json').read_text())
                 for path in (args.prefill, args.decode)]
    metadata = {'prefill_sha256': hashlib.sha256(args.prefill.read_bytes()).hexdigest(),
                'decode_sha256': hashlib.sha256(args.decode.read_bytes()).hexdigest(),
                'plan_manifests': manifests, 'mode': args.mode,
                'decode_loop': args.decode_loop, 'weight_source': 'local shared checkpoint'}
    def engine(mode, loop):
        return ServingEngine(args.model, args.prefill, args.decode, int(args.batch),
            prompt_len=prompt_len, max_new_tokens=args.max_tokens, mode=mode,
            decode_loop=bool(loop), step_events=False)
    frozen = torch.tensor(ids[:int(args.batch)], dtype=torch.int32)
    with engine(args.mode, args.decode_loop) as current:
        measure(args, ids, 'TileMega',
            lambda rows, count: current.generate(frozen, count).tokens.tolist(),
            metadata, synchronize=torch.cuda.synchronize)
        reference = current.generate(frozen, args.max_tokens).tokens.clone()
    token_path = args.out/'reference_tokens.json'
    token_path.write_text(json.dumps(reference.tolist())+'\n')
    if args.check_executors:
        checks = []
        for mode, loop in execution_combinations(manifests[1].get('pg', 'off')):
            with engine(mode, loop) as other:
                tokens = other.generate(frozen, args.max_tokens).tokens
            passed = torch.equal(reference, tokens)
            checks.append({'mode': mode, 'decode_loop': loop, 'pass': passed})
            (args.out/f'tokens_{mode}_loop{loop}.json').write_text(json.dumps(tokens.tolist())+'\n')
        (args.out/'c2.json').write_text(json.dumps({'evidence': 'verified',
            'same_binary': True, 'checks': checks, 'pass': all(c['pass'] for c in checks)}, indent=2)+'\n')
        if not all(c['pass'] for c in checks):
            raise AssertionError('executor token comparison failed')
    if args.hf_check:
        import subprocess
        import sys
        subprocess.run([sys.executable, '-m', 'tilemega.serving.hf_check',
            '--model', str(args.model), '--prompt-ids', str(args.prompt_ids),
            '--generated', str(token_path), '--out', str(args.out/'c1.json'),
            '--skip-free-greedy'], check=True)


if __name__ == '__main__':
    guarded_main(main)
