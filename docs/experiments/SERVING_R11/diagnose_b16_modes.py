#!/usr/bin/env python3
"""Isolate prefill and decode contributions to the B=16 C-2 failure."""
import json
from pathlib import Path

import torch

from tilemega.serving.engine import ServingEngine


def main():
    run = Path('runs/llama-r11-acceptance')
    plans = json.loads((run / 'plans.json').read_text())['16']
    prompt = torch.tensor(json.loads((run / 'prompt_ids.json').read_text())[:16],
                          dtype=torch.int32)
    with ServingEngine('/root/models/llama3_2_1b', plans['prefill'],
                       plans['decode'], 16) as engine:
        results = {}
        for prefill, decode in ((1, 1), (1, 2), (2, 1), (2, 2)):
            engine.prefill_mode, engine.decode_mode = prefill, decode
            results[f'{prefill}{decode}'] = engine.generate(prompt, 8).tokens.tolist()
    baseline = results['11']
    report = {}
    for mode, tokens in results.items():
        differences = [(batch, step, baseline[batch][step], tokens[batch][step])
                       for batch in range(16) for step in range(8)
                       if baseline[batch][step] != tokens[batch][step]]
        report[mode] = {'mismatches': len(differences),
                        'first': differences[:16]}
    output = Path('/root/r11_work/b16_mode_isolation.json')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
