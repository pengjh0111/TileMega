#!/usr/bin/env python3
"""H2 (prompt section 9.2): the routing trace the MoE lower bound needs.

1024 decode steps, one launch at a time (C-2's first execution mode), reading
every layer's top-k index buffer after each launch the way
`scripts/sm90r3_moe_routing.py` does, and reducing it to the distinct expert
count per layer per step. Section 9.3's `bytes_step` for a MoE model is the
attention and router weights plus, per layer, that distinct count times one
expert's bytes - so this is what makes the MoE bound a measurement rather than
a guess between the 1-expert and top_k-expert extremes.

Diagnostic, not a timing path: the per-step launches and the buffer reads make
the wall clock meaningless, which is why the prompt puts it outside P-FINAL.
"""
import argparse, json, os, statistics
from pathlib import Path

import torch

from tilemega.cache import atomic_json
from tilemega.moe.full import generation_tokens
from tilemega.moe.routing_check import capture, index_names
from tilemega.serving.engine import ServingEngine

ROOT = Path(__file__).resolve().parents[1]


def expert_bytes(config):
    """One expert's weight bytes: gate, up and down at bf16."""
    hidden = int(config['hidden_size'])
    inter = int(config['moe_intermediate_size'])
    return 2 * (2 * inter * hidden + inter * hidden)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # Prompt mode (engine.py:38): the decode plan runs the prompt positions
    # teacher-forced, so no prefill binary is needed and none can disagree
    # with the decode plan's workload.
    p.add_argument('--prefill', type=Path)
    p.add_argument('--decode', type=Path, required=True)
    p.add_argument('--bridge', type=Path, required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--steps', type=int, default=1024)
    p.add_argument('--prompts', type=Path, default=ROOT / 'configs/prompts/qwen_isl64.json')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise ValueError('GPU work requires the scheduler guard')
    checkpoint = Path(os.environ['TILEMEGA_RANDOM_CHECKPOINT'])
    config = json.loads((checkpoint / 'config.json').read_text())
    layers, top_k = config['num_hidden_layers'], config['num_experts_per_tok']
    names = index_names(json.loads(a.bridge.read_text()), layers)
    rows = torch.tensor(json.loads(a.prompts.read_text())[:a.batch], dtype=torch.int64)

    # Only the decode steps are wanted; the prefill launch is skipped so step 0
    # is the first generated token, matching how the bound counts steps.
    trace, phases = [], []

    def capture_step(plan, phase):
        phases.append(phase)
        if phase != 'prefill' and len(trace) < a.steps:
            step = capture(plan, names, a.batch, 1, top_k)
            # Per layer, the experts this step actually touches across the batch.
            trace.append([len({e for row in step[layer] for e in row}) for layer in range(layers)])

    with ServingEngine(checkpoint, None, a.decode, a.batch, prompt_len=rows.shape[1],
                       max_new_tokens=a.steps, mode='L1', prefill_mode='L1', decode_loop=False,
                       step_events=False) as engine:
        tokens = generation_tokens(engine, rows, a.steps, 1, config['vocab_size'], capture_step)
    del engine
    torch.cuda.empty_cache()

    a.out.mkdir(parents=True, exist_ok=True)
    atomic_json(a.out / 'routing_trace.json', trace)
    per_expert = expert_bytes(config)
    rows_out = []
    for layer in range(layers):
        counts = [t[layer] for t in trace]
        counts.sort()
        n = len(counts)
        rows_out.append(dict(
            model=config.get('_name_or_path') or config['architectures'][0], batch=a.batch,
            layer=layer, steps=n, experts=config['num_experts'], top_k=top_k,
            distinct_mean=f'{statistics.fmean(counts):.4f}',
            distinct_p10=counts[max(0, int(0.10 * n) - 1)],
            distinct_p90=counts[min(n - 1, int(0.90 * n))],
            distinct_min=counts[0], distinct_max=counts[-1],
            expert_bytes=per_expert,
            layer_bytes_mean=int(statistics.fmean(counts) * per_expert)))
    atomic_json(a.out / 'summary.json', rows_out)
    total = sum(float(r['distinct_mean']) for r in rows_out)
    print(json.dumps(dict(batch=a.batch, steps=len(trace), layers=layers,
                          distinct_per_step_total=round(total, 3),
                          step_expert_bytes=int(total * per_expert),
                          tokens=len(tokens))), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
