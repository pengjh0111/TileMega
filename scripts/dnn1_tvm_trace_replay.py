#!/usr/bin/env python3
"""Replay immutable failed tuning traces in a fresh CPU-only output directory."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]


def signature(error):
    for marker in ('only support even lane for bfloat16', 'too many arguments in function call', 'eval_vec_'):
        if marker in error:
            return marker
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--tvm-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expected', choices=('success', 'original-failure'), required=True)
    args = parser.parse_args()
    snapshot, out = args.snapshot.resolve(), args.out.resolve()
    if out == snapshot or out.is_relative_to(snapshot):
        raise ValueError('replay output must be outside the immutable snapshot')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or 'reject_cuda.so' not in os.environ.get('LD_PRELOAD', ''):
        raise ValueError('CPU replay requires empty CUDA_VISIBLE_DEVICES and reject_cuda preload')
    out.mkdir(parents=True, exist_ok=False)
    files = [snapshot / n for n in ('selected.json', 'database_workload.json')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    result = dict(scope='CPU compilation only; no GPU execution, correctness or performance claim',
        snapshot_sha256=hashes, tvm_root=str(args.tvm_root.resolve()), expected=args.expected,
        cuda_visible_devices='', reject_cuda_preload=True, results=[])
    def save():
        (out/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    save()
    os.environ['TILEMEGA_DNN1_TVM_ROOT'] = str(args.tvm_root.resolve())
    sys.path.insert(0, str(ROOT/'python'))
    import torch
    torch.set_num_threads(1)
    torch.cuda.is_available = lambda: False
    from tilemega.dnn import tvm_baseline as adapter
    tvm, _, _ = adapter._load_tvm()
    from tvm.s_tir import meta_schedule as ms
    from tvm.s_tir.meta_schedule.builder.local_builder import default_build
    from tvm.support import nvcc
    import tvm_ffi
    original_compile = tvm_ffi.get_global_func('tvm_callback_cuda_compile', allow_missing=True)
    original_compile = original_compile or nvcc.tvm_callback_cuda_compile
    current = [None]
    def capture(code):
        (out/(current[0]+'.cuda.cu')).write_text(code)
        return original_compile(code)
    tvm_ffi.register_global_func('tvm_callback_cuda_compile', capture, override=True)
    workloads = [json.loads(s) for s in files[1].read_text().splitlines()]
    for case in json.loads(files[0].read_text()):
        current[0] = case['kind']
        row = dict(kind=case['kind'], database_record_line=case['database_record_line'])
        try:
            workload = ms.database.Workload.from_json(workloads[case['workload_id']])
            record = ms.database.TuningRecord.from_json(case['record'], workload)
            schedule = tvm.s_tir.Schedule(workload.mod)
            record.trace.apply_to_schedule(schedule, remove_postproc=False)
            (out/(case['kind']+'.tir.py')).write_text(schedule.mod.script())
            target = tvm.target.Target(record.target)
            with target:
                default_build(schedule.mod, target, None)
            row.update(status='compiled', passed=args.expected == 'success')
        except Exception as error:
            message = str(error)
            row.update(status='compile_failed', error=message, traceback=traceback.format_exc(),
                passed=args.expected == 'original-failure' and signature(message) is not None
                    and signature(message) == signature(case['error']))
        result['results'].append(row)
        result['source_identity'] = adapter._source_identity()
        save()
        print(json.dumps(dict(kind=row['kind'], status=row['status'], passed=row['passed'])), flush=True)
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest for p,digest in hashes.items()):
        raise ValueError('immutable replay inputs changed')
    result['passed'] = bool(result['results']) and all(r['passed'] for r in result['results'])
    save()
    return int(not result['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
