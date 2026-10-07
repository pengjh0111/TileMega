#!/usr/bin/env python3
"""G-REG real-weight smoke without collecting timing data."""
import argparse
import json
from pathlib import Path

import torch
from tilemega.serving.engine import ServingEngine


def main():
    parser = argparse.ArgumentParser()
    for key in ('model', 'prefill', 'decode', 'prompts', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--batch', type=int, required=True)
    args = parser.parse_args()
    prompts = torch.tensor(json.loads(args.prompts.read_text())[:args.batch],
                           dtype=torch.int32, device='cuda')
    outputs = {}
    with ServingEngine(args.model, args.prefill, args.decode, args.batch,
                       max_new_tokens=64, mode='L1', decode_loop=False,
                       step_events=False, prefill_mode='L1') as engine:
        stream = torch.cuda.current_stream()
        for name, mode in [('L1', 1), ('L2', 2)]:
            engine.state.tokens[:, :64].copy_(prompts)
            engine.prefill.launch(0, mode, stream.cuda_stream)
            for step in range(63):
                engine.decode.launch(step, mode, stream.cuda_stream)
            stream.synchronize()
            outputs[name] = engine.state.tokens[:, 64:128].cpu().tolist()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(batch=args.batch, steps=64, tokens=outputs,
                                        executor_equal=outputs['L1'] == outputs['L2'])) + '\n')
    if outputs['L1'] != outputs['L2']:
        raise SystemExit('G-REG L1/L2 token mismatch')
    print('64-step tokens saved ' + str(args.out), flush=True)


if __name__ == '__main__':
    main()
