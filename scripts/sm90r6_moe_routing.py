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
import argparse, json, math, os, statistics
from pathlib import Path

import torch

from tilemega.cache import atomic_json, file_sha
from tilemega.moe.routing_check import capture, index_names
from tilemega.serving.engine import ServingEngine

ROOT = Path(__file__).resolve().parents[1]


def prompt_rows(path, model, batch, vocab_size):
    if path is None:
        family = {'olmoe_1b_7b': 'olmo', 'qwen3_30b_a3b': 'qwen'}[model]
        path = ROOT / f'configs/prompts/{family}_isl64.json'
    values = json.loads(path.read_text())
    if len(values) < batch or any(len(row) != 64 for row in values[:batch]) or any(
            type(token) is not int or token < 0 or token >= vocab_size
            for row in values[:batch] for token in row):
        raise ValueError('H2 prompts require B rows of 64 in-vocabulary token IDs')
    return path, torch.tensor(values[:batch], dtype=torch.int64)


def expert_bytes(config):
    """One expert's weight bytes: gate, up and down at bf16."""
    hidden = int(config['hidden_size'])
    inter = int(config['moe_intermediate_size'] if 'moe_intermediate_size' in config
                else config['intermediate_size'])
    return 2 * (2 * inter * hidden + inter * hidden)


def generation_routes(engine, rows, steps, vocab_size, capture_step):
    """Capture each generated token's routes, including the prompt's last position."""
    if engine.prefill is not None or engine.decode_loop:
        raise ValueError('H2 requires prompt-mode decode, one launch per position')
    stream = torch.cuda.current_stream()
    engine.state.kv_storage.zero_()
    engine.state.tokens.fill_(-1)
    engine.state.tokens[:, :engine.prompt_len].copy_(rows.to(
        device=engine.state.tokens.device, dtype=torch.int32))
    for position in range(engine.prompt_len - 1):
        engine.decode.launch(position, engine.decode_mode, stream.cuda_stream)
    for step in range(steps):
        position = engine.prompt_len - 1 + step
        engine.decode.launch(position, engine.decode_mode, stream.cuda_stream)
        stream.synchronize()
        capture_step(engine.decode, step, position)
    tokens = engine.state.tokens[:, engine.prompt_len:engine.prompt_len + steps].cpu()
    if not torch.all((tokens >= 0) & (tokens < vocab_size)):
        raise AssertionError('H2 generated unwritten or out-of-vocabulary tokens')
    return tokens.tolist()


def distinct_experts(routes, batch, experts, top_k):
    counts = []
    for layer in routes:
        if len(layer) != batch:
            raise ValueError('routing batch extent differs')
        for row in layer:
            if len(row) != top_k or len(set(row)) != top_k or any(
                    not isinstance(e, int) or e < 0 or e >= experts for e in row):
                raise ValueError('routing buffer contains unwritten, repeated or invalid experts')
        counts.append(len({e for row in layer for e in row}))
    return counts


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
    p.add_argument('--model', required=True, choices=['qwen3_30b_a3b', 'olmoe_1b_7b'])
    p.add_argument('--prompts', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.steps < 1 or a.steps > 1024:
        p.error('--steps must lie in [1, 1024]')
    if a.prefill is not None:
        p.error('H2 uses the native decode plan for teacher-forced prompt positions')
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise ValueError('GPU work requires the scheduler guard')
    checkpoint = Path(os.environ['TILEMEGA_RANDOM_CHECKPOINT'])
    config = json.loads((checkpoint / 'config.json').read_text())
    layers, top_k = config['num_hidden_layers'], config['num_experts_per_tok']
    names = index_names(json.loads(a.bridge.read_text()), layers)
    a.prompts, rows = prompt_rows(a.prompts, a.model, a.batch, config['vocab_size'])

    a.out.mkdir(parents=True, exist_ok=True)
    trace, indices, positions = [], [], []

    def capture_step(plan, step, position):
        routes = capture(plan, names, a.batch, 1, top_k)
        if len(routes) != layers:
            raise ValueError('routing layer extent differs')
        try:
            counts = distinct_experts(routes, a.batch, config['num_experts'], top_k)
        except ValueError:
            atomic_json(a.out / 'invalid_routing.json', dict(step=step, position=position,
                names=names, routes=routes, decode_sha256=file_sha(a.decode)))
            raise
        trace.append(counts)
        indices.append(routes)
        positions.append(position)
        trajectory.write(json.dumps(dict(step=step, position=position, indices=routes,
                                          distinct=counts)) + '\n')
        if (step + 1) % 64 == 0:
            trajectory.flush()
            print(json.dumps({'event': 'routing_progress', 'steps': step + 1}), flush=True)

    with (a.out / 'routing_steps.jsonl').open('x') as trajectory, ServingEngine(
                       checkpoint, None, a.decode, a.batch, prompt_len=rows.shape[1],
                       max_new_tokens=a.steps, mode='L1', prefill_mode='L1', decode_loop=False,
                       step_events=False) as engine:
        tokens = generation_routes(engine, rows, a.steps, config['vocab_size'], capture_step)
    del engine
    torch.cuda.empty_cache()

    if len(trace) != a.steps:
        raise ValueError('incomplete routing trajectory')
    atomic_json(a.out / 'routing_trace.json', trace)
    atomic_json(a.out / 'routing_indices.json', indices)
    atomic_json(a.out / 'tokens.json', tokens)
    atomic_json(a.out / 'metadata.json', dict(model=a.model, batch=a.batch, steps=a.steps,
        positions=positions, first_generated_position=int(rows.shape[1]) - 1,
        decode_sha256=file_sha(a.decode), bridge_sha256=file_sha(a.bridge),
        prompt_sha256=file_sha(a.prompts), gpu=os.environ['TILEMEGA_DEVICE_INDEX'],
        checkpoint=json.loads((checkpoint / 'random_checkpoint.json').read_text())['checkpoint_sha256'],
        evidence='verified', timing=False))
    per_expert = expert_bytes(config)
    rows_out = []
    for layer in range(layers):
        counts = [t[layer] for t in trace]
        counts.sort()
        n = len(counts)
        rows_out.append(dict(
            model=a.model, batch=a.batch,
            layer=layer, steps=n, experts=config['num_experts'], top_k=top_k,
            distinct_mean=statistics.fmean(counts),
            distinct_p10=counts[math.ceil(0.10 * n) - 1],
            distinct_p90=counts[math.ceil(0.90 * n) - 1],
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
