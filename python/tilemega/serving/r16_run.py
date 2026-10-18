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
    parser.add_argument('--system-label', choices=('TileMega', 'TM-R15', 'TM-RF1-on', 'TM-RF1-off', 'TM-seed', 'TM-H1', 'TM-H1H4', 'TM-sm80', 'TM-noWGMMA'), default='TileMega')
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
    parser.add_argument('--prefill-mode', choices=('L1', 'L2'))
    parser.add_argument('--decode-loop', type=int, choices=(0, 1), default=0)
    parser.add_argument('--check-executors', action='store_true')
    parser.add_argument('--hf-check', action='store_true')
    parser.add_argument('--hf-positions', type=int)
    args = parser.parse_args()
    require_guard()
    if args.model is None:
        checkpoint = os.environ.get('TILEMEGA_RANDOM_CHECKPOINT')
        if not checkpoint:
            parser.error('--model or TILEMEGA_RANDOM_CHECKPOINT is required')
        args.model = Path(checkpoint)
    if min(args.max_tokens, args.warmup, args.repeats) < 1:
        parser.error('token and repetition counts must be positive')
    if args.hf_positions is not None and (
            not args.hf_check or not 1 <= args.hf_positions <= args.max_tokens):
        parser.error('--hf-positions requires --hf-check and must fit the generated sequence')
    args.load_format = 'auto'
    args.out.mkdir(parents=True, exist_ok=True)
    ids = json.loads(args.prompt_ids.read_text())
    prompt_len = validate_prompts(ids, int(args.batch))
    import torch
    from .engine import ServingEngine
    from .execution import execution_combinations
    identities = {phase: json.loads(Path(str(path)+'.identity.json').read_text())
                  for phase, path in (('prefill', args.prefill), ('decode', args.decode))}
    manifests = [json.loads(Path(str(path)+'.plan.json').read_text())
                 for path in (args.prefill, args.decode)]
    metadata = {'prefill_sha256': hashlib.sha256(args.prefill.read_bytes()).hexdigest(),
                'decode_sha256': hashlib.sha256(args.decode.read_bytes()).hexdigest(),
                'plan_manifests': manifests, 'mode': args.mode,
                'decode_loop': args.decode_loop, 'prefill_mode': args.prefill_mode or args.mode,
                'hf_positions': args.hf_positions, 'weight_source': 'local checkpoint',
                'system_label': args.system_label,
                'selection_status': 'diagnostic candidate; no RF2 final selection claimed',
                'prefill_binary': str(args.prefill.resolve()), 'decode_binary': str(args.decode.resolve()),
                'identities': identities,
                'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'hf_checker_sha256': hashlib.sha256(Path(__file__).with_name('hf_check.py').read_bytes()).hexdigest()}
    runtime_manifest=os.environ.get('TILEMEGA_RUNTIME_MANIFEST')
    if runtime_manifest:
        path=Path(runtime_manifest)
        recorded=json.loads(path.read_text())
        for name,expected in recorded['files'].items():
            if hashlib.sha256((path.parent/'python'/name).read_bytes()).hexdigest()!=expected:
                raise ValueError('frozen runtime Python source changed: '+name)
        metadata['runtime_manifest']=str(path.resolve())
        metadata['runtime_manifest_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint_manifest = args.model/'random_checkpoint.json'
    if checkpoint_manifest.is_file():
        checkpoint = json.loads(checkpoint_manifest.read_text())
        metadata.update(checkpoint_manifest=str(checkpoint_manifest.resolve()),
                        checkpoint_sha256=checkpoint['checkpoint_sha256'],
                        weight_source='shared synthetic TileMega/HF checkpoint; pretrained C-1 unavailable')
    def engine(mode, loop):
        return ServingEngine(args.model, args.prefill, args.decode, int(args.batch),
            prompt_len=prompt_len, max_new_tokens=args.max_tokens, mode=mode,
            decode_loop=bool(loop), step_events=False, prefill_mode=args.prefill_mode)
    frozen = torch.tensor(ids[:int(args.batch)], dtype=torch.int32)
    with engine(args.mode, args.decode_loop) as current:
        executions=[]
        metadata['actual_execution']=executions
        def generate(rows,count):
            result=current.generate(frozen,count)
            if count>1 and bool(args.decode_loop)!=result.decode_loop_used:
                raise ValueError('requested decode-loop mode was not executed')
            executions.append(dict(tokens=count,decode_mode=args.mode,
                                   prefill_mode=args.prefill_mode or args.mode,
                                   decode_loop_used=result.decode_loop_used))
            return result.tokens.tolist()
        measure(args, ids, args.system_label,
            generate,
            metadata, synchronize=torch.cuda.synchronize)
        last = args.out/f'B{args.batch}'/f'tokens_N{args.max_tokens}_run{args.warmup+args.repeats-1}.json'
        reference = torch.tensor(json.loads(last.read_text()), dtype=torch.int32)
    # close() releases native contexts; Python still owns the full checkpoint.
    # Drop it before loading another executor or the separate HF reference.
    del current
    torch.cuda.empty_cache()
    token_path = args.out/'reference_tokens.json'
    token_path.write_text(json.dumps(reference.tolist())+'\n')
    if args.check_executors:
        checks = []
        for mode, loop in execution_combinations(manifests[1].get('pg', 'off')):
            with engine(mode, loop) as other:
                generated = other.generate(frozen, args.max_tokens)
                if args.max_tokens>1 and generated.decode_loop_used!=bool(loop):
                    raise ValueError('C-2 requested an unavailable decode-loop mode')
                tokens = generated.tokens.cpu()
            del other
            torch.cuda.empty_cache()
            passed = torch.equal(reference, tokens)
            checks.append({'mode': mode, 'decode_loop': loop,
                           'decode_loop_used': generated.decode_loop_used, 'pass': passed})
            (args.out/f'tokens_{mode}_loop{loop}.json').write_text(json.dumps(tokens.tolist())+'\n')
        (args.out/'c2.json').write_text(json.dumps({'evidence': 'verified',
            'same_binary': True, 'checks': checks, 'pass': all(c['pass'] for c in checks)}, indent=2)+'\n')
        if not all(c['pass'] for c in checks):
            raise AssertionError('executor token comparison failed')
    if args.hf_check:
        import subprocess
        import sys
        hf_tokens=token_path
        model_config=json.loads((args.model/'config.json').read_text())
        positions=args.hf_positions or (min(256,args.max_tokens)
            if model_config.get('model_type')=='qwen3_moe' else args.max_tokens)
        if positions<args.max_tokens:
            hf_tokens=args.out/f'teacher_forced_prefix_{positions}.json'
            hf_tokens.write_text(json.dumps(reference[:,:positions].tolist())+'\n')
        subprocess.run([sys.executable, '-m', 'tilemega.serving.hf_check',
            '--model', str(args.model), '--prompt-ids', str(args.prompt_ids),
            '--generated', str(hf_tokens), '--out', str(args.out/'c1.json'),
            '--skip-free-greedy'], check=True)


if __name__ == '__main__':
    guarded_main(main)
