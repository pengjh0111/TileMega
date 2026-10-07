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
    # G-REG compares the reference and candidate at fixed execution settings.
    # Cross-executor C-2 is a separate MoE gate in section 8.C.
    decode_mode, prefill_mode = 'L2', 'L1'
    with ServingEngine(args.model, args.prefill, args.decode, args.batch,
                       max_new_tokens=64, mode=decode_mode, decode_loop=False,
                       step_events=False, prefill_mode=prefill_mode) as engine:
        stream = torch.cuda.current_stream()
        engine.state.tokens[:, :64].copy_(prompts)
        engine.prefill.launch(0, 1, stream.cuda_stream)
        for step in range(63):
            engine.decode.launch(step, 2, stream.cuda_stream)
        stream.synchronize()
        tokens = engine.state.tokens[:, 64:128].cpu().tolist()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(batch=args.batch, steps=64, tokens=tokens,
                                        execution=dict(prefill=prefill_mode,
                                                       decode=decode_mode, loop=False))) + '\n')
    print('64-step tokens saved ' + str(args.out), flush=True)


if __name__ == '__main__':
    main()
