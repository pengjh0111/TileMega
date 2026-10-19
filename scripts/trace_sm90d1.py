#!/usr/bin/env python3
"""Warm and record eight consecutive diagnostic decode steps at each past."""
import argparse
import ctypes as C
import os
from pathlib import Path

from sm90d1_common import read, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefill', type=Path, required=True)
    p.add_argument('--decode', type=Path, required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--mode', choices=('L1', 'L2'), required=True)
    p.add_argument('--loop', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    import torch
    from tilemega.serving.baseline_common import require_guard
    from tilemega.serving.engine import ServingEngine
    require_guard()
    os.environ['TILEMEGA_TRACE_LAUNCHES'] = '64'
    for past in (64, 575, 1000):
        folder = a.out / f'past{past}'
        folder.mkdir(parents=True, exist_ok=False)
        os.environ['TILEMEGA_PAGE_TRACE_OUT'] = str(folder / 'page_trace.tsv')
        os.environ['TILEMEGA_TRACE_V2'] = '1'
        os.environ['TILEMEGA_TRACE_V2_OUT'] = str(folder / 'v2')
        with ServingEngine(os.environ['TILEMEGA_RANDOM_CHECKPOINT'], a.prefill, a.decode,
                           a.batch, mode=a.mode, decode_loop=bool(a.loop),
                           prefill_mode='L1') as engine:
            engine.state.tokens.zero_()
            engine.state.kv_storage.zero_()
            stream = torch.cuda.current_stream()
            mode = 1 if a.mode == 'L1' else 2
            def launch():
                if a.loop:
                    engine.decode.launch_steps(past-64, 8, mode, stream.cuda_stream)
                else:
                    for step in range(past-64, past-64+8):
                        engine.decode.launch(step, mode, stream.cuda_stream)
            launch()
            stream.synchronize()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record(stream)
            launch()
            end.record(stream)
            end.synchronize()
            elapsed = start.elapsed_time(end)/8
            dump = getattr(engine.decode_lib.lib, 'tm_plan_dump_serving_trace', None)
            if dump:
                dump.argtypes = [C.c_void_p, C.c_char_p]
                dump.restype = C.c_int
                if dump(engine.decode.handle, os.fsencode(folder.resolve())):
                    raise RuntimeError('serving trace dump failed')
            v2 = getattr(engine.decode_lib.lib, 'tm_plan_dump_trace_v2', None)
            if a.mode == 'L2' and v2:
                v2.argtypes = [C.c_void_p, C.c_float]
                v2.restype = C.c_int
                if v2(engine.decode.handle, elapsed):
                    raise RuntimeError('event trace dump failed')
            info = engine.decode_lib.info
            write(folder / 'trace.json', dict(past=past, mode=a.mode, loop=bool(a.loop),
                mean_step_ms=elapsed, measured_steps=8, warmup_steps=8,
                retain_latest_iterations=8, grid=info.grid, residency=info.residency,
                execution_identity=engine.decode.execution_identity(mode, bool(a.loop)),
                driver_sha256=sha(__file__), diagnostic_only=True, performance_eligible=False,
                input='fixed random weights; zero initial KV/token buffers; same R16 diagnostic flow'))


if __name__ == '__main__':
    main()
