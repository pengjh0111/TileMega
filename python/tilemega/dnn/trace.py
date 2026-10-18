"""Dump an instrumented forward plan's L2 task DAG; never report performance."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('library', 'export', 'bridge', 'inputs', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    from tilemega.serving.baseline_common import require_guard
    require_guard()
    identity = json.loads(Path(str(a.library) + '.identity.json').read_text())
    if not identity.get('trace'):
        raise ValueError('forward task traces require a separate instrumented binary')
    a.out.mkdir(parents=True, exist_ok=True)
    os.environ['TILEMEGA_TRACE_V2'] = '1'
    os.environ['TILEMEGA_TRACE_V2_OUT'] = str((a.out / 'v2').resolve())
    os.environ['TILEMEGA_TRACE_LAUNCHES'] = '1'
    import torch
    from safetensors.torch import load_file
    from .check import NativeForward
    inputs = load_file(str(a.inputs), device='cuda')
    with torch.inference_mode(), NativeForward(a.library, a.export, a.bridge, 1) as native:
        reference = native([inputs[name] for name in native.inputs])
        begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        begin.record()
        native.plan.launch(0,2,torch.cuda.current_stream().cuda_stream)
        end.record()
        end.synchronize()
        if any(not torch.equal(value, native.tensors[name])
               for value, name in zip(reference, native.outputs)):
            raise AssertionError('instrumented L2 output differs from the checked forward')
        dump = native.library.lib.tm_plan_dump_trace_v2
        dump.argtypes = [C.c_void_p, C.c_float]
        dump.restype = C.c_int
        status = dump(native.plan.handle, begin.elapsed_time(end))
        if status:
            raise RuntimeError(f'forward L2 trace dump failed: {status}')
        profile = native.library.lib.tm_plan_dump_serving_trace
        profile.argtypes = [C.c_void_p, C.c_char_p]
        profile.restype = C.c_int
        status = profile(native.plan.handle, os.fsencode(a.out.resolve()))
        if status:
            raise RuntimeError(f'forward task profile dump failed: {status}')
    (a.out / 'trace.json').write_text(json.dumps(dict(
        evidence='verified', diagnostic_only=True, timing_valid=False, mode='L2',
        library=str(a.library.resolve()), artifact_id=identity['artifact_id'],
        input_sha256=hashlib.sha256(a.inputs.read_bytes()).hexdigest(),
        profile_scope='thread-0 stamps, one in eight workers; V2 records every task',
        batch=1, execution_identity=identity), indent=2) + '\n')


if __name__ == '__main__':
    main()
