#!/usr/bin/env python3
"""Record native stage boundaries after uninstrumented request history."""
import argparse
import ctypes as C
import os
from pathlib import Path

from sm90d1_common import read, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefill', type=Path, required=True)
    p.add_argument('--decode', type=Path, required=True)
    p.add_argument('--decode-plain', type=Path, required=True)
    p.add_argument('--prompt-ids', type=Path, required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--mode', choices=('L1',), required=True)
    p.add_argument('--loop', type=int, choices=(0, 1), required=True)
    p.add_argument('--past', type=int, nargs='+', default=[575])
    p.add_argument('--warmup-windows', type=int, default=4)
    p.add_argument('--control', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.warmup_windows < 1 or any(v < 64 or v + 7 > 1086 for v in a.past):
        p.error('positive warmup and an eight-step window within 64:1086 required')
    import torch
    from tilemega.build.identity import verify
    from tilemega.serving.baseline_common import require_guard
    from tilemega.serving.engine import ServingEngine
    from tilemega.serving.plan import PlanLibrary
    require_guard()
    plain, trace = verify(a.decode_plain), verify(a.decode)
    if plain['trace'] or trace['trace'] == a.control:
        raise ValueError('explicit uninstrumented history and instrumented window required')
    if Path(str(a.decode_plain) + '.cu').read_bytes() != Path(str(a.decode) + '.cu').read_bytes():
        raise ValueError('history/window generated programs differ')
    ids = read(a.prompt_ids)
    if isinstance(ids, dict):
        ids = ids['input_ids']
    prompt = torch.tensor(ids[:a.batch], dtype=torch.int32)
    if tuple(prompt.shape) != (a.batch, 64):
        raise ValueError('frozen prompt must cover this 64-token batch')
    slots = a.warmup_windows * 8 + 16
    os.environ['TILEMEGA_TRACE_LAUNCHES'] = str(slots)
    trace_lib = PlanLibrary(a.decode)
    for past in a.past:
        folder = a.out / f'past{past}'
        folder.mkdir(parents=True, exist_ok=False)
        with ServingEngine(os.environ['TILEMEGA_RANDOM_CHECKPOINT'], a.prefill,
                           a.decode_plain, a.batch, mode='L1', decode_loop=True,
                           prefill_mode='L1') as engine:
            # count=1 gives token64 from prefill; the prefix ends at token[past].
            engine.generate(prompt, new_tokens=past - 64 + 1)
            pointers = {name: value.data_ptr() for name, value in engine.weights.items()}
            pointers.update(engine.state.pointers)
            traced = trace_lib.create(a.batch, pointers, 0)
            try:
                traced.set_steps(list(range(64, 1087)))
                stream = torch.cuda.current_stream()
                def launch():
                    if a.loop:
                        traced.launch_steps(past - 64, 8, 1, stream.cuda_stream)
                    else:
                        for step in range(past - 64, past - 64 + 8):
                            traced.launch(step, 1, stream.cuda_stream)
                for _ in range(a.warmup_windows):
                    launch()
                stream.synchronize()
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record(stream)
                launch()
                end.record(stream)
                end.synchronize()
                stamps = traced.read_step_ns(past - 64, 9) if a.loop else None
                if not a.control:
                    dump = trace_lib.lib.tm_plan_dump_serving_trace
                    dump.argtypes = [C.c_void_p, C.c_char_p]
                    dump.restype = C.c_int
                    if dump(traced.handle, os.fsencode(folder.resolve())):
                        raise RuntimeError('native trace dump failed')
                write(folder / 'trace.json', dict(past=past, mode='L1', loop=bool(a.loop),
                    mean_step_ms=start.elapsed_time(end) / 8, measured_steps=8,
                    warmup_steps=a.warmup_windows * 8, history_decode_steps=past - 64,
                    trace_ring_slots=slots, protocol_version=3, retain_latest_iterations=8,
                    loop_step_ns=[stamps[i + 1] - stamps[i] for i in range(8)] if stamps else None,
                    grid=trace_lib.info.grid, residency=trace_lib.info.residency,
                    execution_identity=traced.execution_identity(1, bool(a.loop)),
                    driver_sha256=sha(__file__), plain_binary_sha256=sha(a.decode_plain),
                    trace_binary_sha256=sha(a.decode), prompt_ids_sha256=sha(a.prompt_ids),
                    input='fixed random weights; actual prefill and greedy decode history; shared external KV/token buffers',
                    internal_buffers='fresh traced-plan temporaries; embedding writes first hidden state',
                    body_equivalence='requires separate compiled-function audit',
                    matched_window_control=a.control, diagnostic_only=True, performance_eligible=False))
            finally:
                traced.close()


if __name__ == '__main__':
    main()
