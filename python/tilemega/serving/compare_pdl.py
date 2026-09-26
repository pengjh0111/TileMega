"""Compare two-step generation with and without programmatic launch overlap."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .engine import ServingEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--prefill-so', type=Path, required=True)
    parser.add_argument('--pdl-so', type=Path, required=True)
    parser.add_argument('--plain-so', type=Path, required=True)
    parser.add_argument('--prompt-ids', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    ids = json.loads(args.prompt_ids.read_text())
    prompts = torch.tensor(ids[:1], dtype=torch.int32)
    results = {}
    for label, library in (('pdl', args.pdl_so), ('plain', args.plain_so)):
        with ServingEngine(args.model, args.prefill_so, library, 1, mode='L2') as engine:
            results[label] = engine.generate(prompts, 2).tokens.tolist()
    passed = results['pdl'] == results['plain']
    report = {'pass': passed, 'steps': 2, 'batch': 1, 'tokens': results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    if not passed:
        raise AssertionError('PDL and plain launch changed two-step tokens')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
