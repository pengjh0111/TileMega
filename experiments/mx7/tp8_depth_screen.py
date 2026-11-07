"""Diagnostic full-model depth screening; final timing uses independent processes."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from tilemega.serving.measure_tp import require_group
from tilemega.serving import tp_engine
from tilemega.serving.weights import weight_recipes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if require_group() != list(range(8)):
        raise ValueError('eight-card physical group required')
    data = json.loads(args.inputs.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'producer.py').write_bytes(Path(__file__).read_bytes())
    original_loader = tp_engine.load_weights
    cache, signatures = {}, {}

    def cached(model, *plans, device, tp):
        signature = sorted((name, b.dtype, b.elements_constant, b.pack_json)
                           for name, b in weight_recipes(*plans).items())
        key = (str(device), tp)
        if key in cache:
            if signatures[key] != signature:
                raise ValueError('depth screening changed the immutable weight layout')
            return cache[key]
        value = original_loader(model, *plans, device=device, tp=tp)
        cache[key], signatures[key] = value, signature
        return value

    # This cache only shortens diagnostic startup. Ordinary measure_tp is used
    # in fresh processes for every final paired comparison and reported TPOT.
    tp_engine.load_weights = cached
    prompt = json.loads(Path(data['prompt_ids']).read_text())[:data['batch']]
    expected = json.loads(Path(data['reference']).read_text())
    prompts = torch.tensor(prompt, dtype=torch.int32)
    results = []
    try:
        for ordinal, item in enumerate(data['candidates']):
            with Path(item['binary']).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != item['binary_sha256']:
                    raise ValueError('screen binary changed')
            print('screen', data['batch'], item['name'], flush=True)
            rows = []
            with tp_engine.TpServingEngine(data['model'], item['binary'], data['batch'],
                    prompt_len=64, max_new_tokens=1024, devices=tuple(range(8))) as engine:
                for repeat in range(3):
                    generated = engine.generate(prompts)
                    tokens = generated.tokens.tolist()
                    if tokens != expected:
                        (args.out / 'TOKEN_FAILURE.json').write_text(json.dumps(dict(
                            candidate=item, repeat=repeat, timing_eligible=False)) + '\n')
                        raise RuntimeError('depth-only candidate changed the reference token sequence')
                    rows.append(dict(repeat=repeat, warmup=repeat == 0,
                        e2e_ms=generated.e2e_ms, ttft_ms=generated.ttft_ms,
                        tpot_ms=(generated.e2e_ms - generated.ttft_ms) / 1023,
                        rank_tokens_identical=True, reference_tokens_identical=True))
                if ordinal + 1 == len(data['candidates']):
                    # Release the diagnostic cache while the final engine still
                    # owns the tensors and its SDK streams remain alive.
                    cache.clear()
            result = dict(candidate=item, runs=rows, median_tpot_ms=statistics.median(
                row['tpot_ms'] for row in rows if not row['warmup']),
                timing_eligible=False, classification='diagnostic_shared_immutable_weights',
                note='requires ordinary fresh-process paired timing before selection')
            results.append(result)
            (args.out / f'{item["name"]}.json').write_text(json.dumps(result, indent=2) + '\n')
            print('screened', data['batch'], item['name'], result['median_tpot_ms'], flush=True)
        (args.out / 'results.json').write_text(json.dumps(dict(batch=data['batch'], results=results,
            timing_eligible=False, finished=time.time()), indent=2) + '\n')
    finally:
        tp_engine.load_weights = original_loader
        cache.clear()


if __name__ == '__main__':
    main()
