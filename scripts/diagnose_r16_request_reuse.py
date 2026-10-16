#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Non-timing, frozen-binary diagnostics for repeated request state reuse."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_spec(path):
    spec = json.loads(Path(path).read_text())
    if sha(__file__) != spec['diagnostic_sha256']:
        raise ValueError('diagnostic changed after freezing')
    for name, expected in spec['files'].items():
        if sha(name) != expected:
            raise ValueError('frozen dependency changed: ' + name)
    candidate = json.loads(Path(spec['candidate']).read_text())
    index = candidate['gpu']
    allowed = spec.get('allowed_gpu_indices', [7])
    if (index not in allowed or index not in (5, 6, 7) or
            os.environ.get('CUDA_VISIBLE_DEVICES') != str(index) or
            os.environ.get('TILEMEGA_DEVICE_INDEX') != str(index) or
            os.environ.get('TILEMEGA_GPU_LOCK') != f'/tmp/tilemega_gpu{index}.lock' or
            os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1'):
        raise ValueError('diagnostic requires its authorized guarded physical GPU')
    return spec, candidate


def differences(reference, actual):
    if len(reference) != len(actual) or any(len(a) != len(b) for a, b in zip(reference, actual)):
        raise ValueError('token output shape changed')
    return [next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None)
            for left, right in zip(reference, actual)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--strategy', choices=('reuse', 'zero-kv', 'nan-kv', 'fresh-engine'), required=True)
    parser.add_argument('--requests', type=int, default=4)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.requests < 2 or args.max_tokens < 1:
        parser.error('need at least two requests and a positive token count')
    spec, candidate = validate_spec(args.spec)
    sys.path.insert(0, spec['validation_dir'])
    from collect_r16_candidates import load_driver
    pair = load_driver(candidate).check_pair(candidate)
    from tilemega.serving.baseline_common import require_guard
    require_guard()
    from tilemega.serving.engine import ServingEngine
    import tilemega.serving.engine as engine_module
    if Path(engine_module.__file__).resolve() != Path(spec['engine_source']).resolve():
        raise ValueError('runtime imported another source snapshot')
    import torch
    model = Path(os.environ['TILEMEGA_RANDOM_CHECKPOINT'])
    checkpoint = json.loads((model/'random_checkpoint.json').read_text())
    if (checkpoint['seed'] != candidate['seed'] or
            sha(model/'config.json') != candidate['config_sha256']):
        raise ValueError('checkpoint seed or model config differs')
    ids = json.loads(Path(candidate['prompt_ids']).read_text())[:candidate['batch']]
    prompt = torch.tensor(ids, dtype=torch.int32)
    args.out.mkdir(parents=True, exist_ok=True)
    evidence = dict(evidence='verified execution only; no performance acceptance',
        strategy=args.strategy, spec_sha256=sha(args.spec), pid=os.getpid(),
        candidate=candidate['id'], checkpoint=checkpoint,
        artifact_ids={p: value['artifact_id'] for p, value in pair.items()},
        max_tokens=args.max_tokens, requests=args.requests, runs=[], default_promoted=False)
    output = args.out/'request-reuse.json'

    def save():
        output.write_text(json.dumps(evidence, indent=2)+'\n')

    def create():
        return ServingEngine(model, candidate['prefill'], candidate['decode'],
            candidate['batch'], prompt_len=len(ids[0]), max_new_tokens=args.max_tokens,
            mode=candidate['mode'], decode_loop=bool(candidate['loop']), step_events=False)

    current = None
    reference = None
    save()
    try:
        for index in range(args.requests):
            if current is None:
                current = create()
                # Match the production N=1 warmup/repeats before its long requests.
                for _ in range(4):
                    current.generate(prompt, 1)
            if args.strategy in ('zero-kv', 'nan-kv'):
                current.state.kv_storage.fill_(0 if args.strategy == 'zero-kv' else float('nan'))
                current.state.tokens.fill_(-1)
                torch.cuda.synchronize()
            before = dict(prefill=current.prefill.iteration.copy(), decode=current.decode.iteration.copy())
            result = current.generate(prompt, args.max_tokens)
            tokens = result.tokens.tolist()
            if reference is None:
                reference = tokens
            token_path = args.out/f'tokens-request{index}.json'
            token_path.write_text(json.dumps(tokens)+'\n')
            expected_loop = bool(candidate['loop'] and args.max_tokens > 1)
            if result.decode_loop_used != expected_loop:
                raise ValueError('requested execution silently fell back')
            positions = [p for p in (0, 63, 64, 125, 126, 127, 128, 191, 255, 575)
                         if p < len(ids[0])+args.max_tokens-1]
            # Sample active boundary rows, retaining layer attribution without copying the full cache.
            kv = current.state.kv_storage[:, :, :, :, positions, :].contiguous().cpu()
            raw = kv.view(torch.uint16).numpy()
            row = dict(index=index, before_epochs=before,
                after_epochs=dict(prefill=current.prefill.iteration.copy(), decode=current.decode.iteration.copy()),
                actual_device_loop=result.decode_loop_used,
                tokens=str(token_path), token_sha256=sha(token_path),
                first_differences=differences(reference, tokens), kv_positions=positions,
                kv_boundary_sha256=[hashlib.sha256(layer.tobytes()).hexdigest() for layer in raw],
                active_kv_nonfinite=int((~torch.isfinite(kv)).sum().item()))
            evidence['runs'].append(row)
            save()
            print(json.dumps(dict(strategy=args.strategy, request=index,
                first_differences=row['first_differences'], active_kv_nonfinite=row['active_kv_nonfinite'])), flush=True)
            if args.strategy == 'fresh-engine':
                current.close()
                del current
                current = None
                torch.cuda.empty_cache()
        evidence['repeat_tokens_equal'] = all(all(p is None for p in r['first_differences']) for r in evidence['runs'])
        evidence['completed'] = True
    except BaseException as error:
        evidence.update(completed=False, error=repr(error))
        raise
    finally:
        save()
        if current is not None:
            current.close()


if __name__ == '__main__':
    main()
