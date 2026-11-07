#!/usr/bin/env python3
"""Collect attribution-only forward records and write a per-stage ledger."""
import argparse
import csv
import ctypes as C
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from tilemega.build.identity import sha, verify


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def collect(args, identity):
    from tilemega.serving.baseline_common import require_guard
    require_guard()
    import torch
    from safetensors.torch import load_file
    from tilemega.dnn.check import NativeForward
    inputs = load_file(str(args.inputs), device='cuda')
    reference = None
    if args.reference:
        with torch.inference_mode(), NativeForward(args.reference, args.export,
                                                   args.bridge, args.batch) as accepted:
            reference = [value.clone() for value in accepted([inputs[name] for name in accepted.inputs])]
    with torch.inference_mode(), NativeForward(args.library, args.export, args.bridge, args.batch) as native:
        if reference is None:
            reference = [value.clone() for value in native([inputs[name] for name in native.inputs])]
        lib = native.library.lib
        dimensions = {}
        for name in ('schema', 'stages', 'grid', 'segments', 'record_bytes', 'bytes'):
            symbol = getattr(lib, 'tm_forward_trace_' + name)
            symbol.restype = C.c_ulonglong if name == 'bytes' else C.c_uint
            symbol.argtypes = []
            dimensions[name] = symbol()
        if dimensions['schema'] != 1 or dimensions['record_bytes'] != 8*(4+4*dimensions['segments']):
            raise ValueError('unknown forward trace record layout')
        if dimensions['bytes'] != dimensions['stages']*dimensions['grid']*dimensions['record_bytes']:
            raise ValueError('trace record count differs from dimensions')
        name_of = lib.tm_forward_trace_segment_name
        name_of.restype = C.c_char_p
        name_of.argtypes = [C.c_uint]
        names = [name_of(i).decode() for i in range(dimensions['segments'])]
        reset = lib.tm_forward_trace_reset
        reset.argtypes = []
        reset.restype = C.c_int
        if reset():
            raise RuntimeError('forward trace reset failed')
        native.plan.launch(0, 1, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        if any(not torch.equal(value, native.tensors[name])
               for value, name in zip(reference, native.outputs)):
            raise AssertionError('traced execution differs from checked output')
        data = C.create_string_buffer(dimensions['bytes'])
        copy = lib.tm_forward_trace_copy
        copy.argtypes = [C.c_void_p, C.c_ulonglong]
        copy.restype = C.c_int
        if copy(data, dimensions['bytes']):
            raise RuntimeError('forward trace copy failed')
    (args.out / 'records.bin').write_bytes(data.raw)
    manifest = identity['plan']
    stages = manifest['native_forward']['stages']
    rows = []
    words = dimensions['record_bytes']//8
    for stage in range(dimensions['stages']):
        records = [struct.unpack_from('<'+'Q'*words, data.raw,
                   (stage*dimensions['grid']+cta)*dimensions['record_bytes'])
                   for cta in range(dimensions['grid'])]
        if any(not r[0] or not (r[0] <= r[1] <= r[2]) for r in records):
            raise ValueError('incomplete or unordered stage trace')
        enter = min(r[0] for r in records)
        arrive = max(r[1] for r in records)
        leave = max(r[2] for r in records)
        row = dict(stage=stage, body=stages[stage]['body'], mode=stages[stage]['mode'],
                   stage_ns=leave-enter, work_window_ns=arrive-enter,
                   gate_tail_ns=leave-arrive,
                   first_ready_ns=min((r[3]-enter for r in records if r[3]), default=0))
        for i, name in enumerate(names):
            spans = [r[4+4*i:8+4*i] for r in records]
            row[name+'_max_cta_ns'] = max(s[0] for s in spans)
            row[name+'_mean_cta_ns'] = sum(s[0] for s in spans)/dimensions['grid']
            row[name+'_count'] = sum(s[1] for s in spans)
        rows.append(row)
    with (args.out / 'ledger.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    save(args.out / 'result.json', dict(evidence='verified', diagnostic_only=True,
        timing_valid=False, artifact_id=identity['artifact_id'], dimensions=dimensions,
        input_sha256=sha(args.inputs), ledger=str(args.out / 'ledger.csv'),
        reference=str(args.reference) if args.reference else 'independent clone of first trace launch',
        reference_sha256=sha(args.reference) if args.reference else None,
        limitation='Thread 0 attribution; nested segments are not additive; trace overhead changes scheduling.',
        largest_stages=sorted(rows, key=lambda row: row['stage_ns'], reverse=True)[:3]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('library', 'export', 'bridge', 'inputs', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--cpu-set', default='128-135')
    parser.add_argument('--reference', type=Path, help='accepted non-trace artifact at identical coordinates')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    for name in ('library', 'export', 'bridge', 'inputs', 'out'):
        setattr(args, name, getattr(args, name).resolve())
    identity = verify(args.library)
    if not identity['trace']:
        raise ValueError('attribution requires a separate trace artifact')
    if args.reference:
        args.reference = args.reference.resolve()
        accepted = verify(args.reference)
        if accepted['trace'] or accepted['plan']['coordinates_sha256'] != identity['plan']['coordinates_sha256']:
            raise ValueError('trace reference must be non-trace and use identical coordinates')
    entries = [r for name, r in identity['resources'].items() if 'sm90_forward_f3_kernel' in name]
    if len(entries) != 1 or entries[0]['registers'] > 168:
        raise ValueError('trace entry cannot reside with288 Hopper threads')
    log = Path(str(args.library)+'.ptxas.log').read_text()
    spills = re.findall(r'(\d+) bytes spill (?:stores|loads)', log)
    registers = re.findall(r'Used (\d+) registers', log)
    if not spills or any(map(int, spills)) or not registers or max(map(int, registers)) > 224:
        raise ValueError('DT-SPILL rejects trace execution')
    if args.execute:
        args.out.mkdir(parents=True, exist_ok=True)
        return collect(args, identity)
    args.out.mkdir(parents=True, exist_ok=False)
    environment = dict(CUDA_VISIBLE_DEVICES=str(args.device), TILEMEGA_DEVICE_INDEX=str(args.device),
        TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{args.device}.lock', TILEMEGA_NUMA_NODE='0',
        TILEMEGA_GPU_PID_MODE='namespace_limited', TILEMEGA_NUMA_CONCURRENCY_TRIAL='1',
        PYTHONPATH=str(ROOT / 'python'))
    queue = args.out / 'queue'
    queue.mkdir()
    command = ['taskset', '-c', args.cpu_set, str(ROOT/'scripts/dnn3_idle_floor_python.py'),
               str(Path(__file__).resolve()), '--execute', '--batch', str(args.batch)]
    for name in ('library', 'export', 'bridge', 'inputs'):
        command += ['--'+name, str(getattr(args, name))]
    command += ['--out', '{attempt_dir}/trace']
    if args.reference:
        command += ['--reference', str(args.reference)]
    save(queue / 'queue_marks.json', [dict(name='dnn3-marks', gpu=True, timing=False,
        timeout_s=1200, needs_free_mib=8192, command=command, cwd=str(ROOT), env=environment)])
    paths = [Path(__file__), args.library, Path(str(args.library)+'.identity.json'),
             args.inputs, args.bridge, args.export/'exported_program.pt2',
             args.export/'checkpoint/model.safetensors',
             *sorted((ROOT/'python/tilemega/dnn').glob('*.py'))]
    if args.reference:
        paths.extend((args.reference, Path(str(args.reference)+'.identity.json')))
    frozen = {str(path): sha(path) for path in paths}
    save(args.out/'inputs.json', frozen)
    scheduler = [str(ROOT/'scripts/dnn3_idle_floor_python.py'), '-m', 'tilemega.bench.scheduler',
        '--queue-dir', str(queue), '--out', str(args.out/'scheduler'),
        '--policy', str(ROOT/'configs/bench/sm90r2_guard_policy.json'), '--deadline-hours', '6']
    with (args.out/'dispatch.log').open('w') as stream:
        subprocess.run(scheduler, cwd=ROOT, env=dict(os.environ, **environment),
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    if any(sha(path) != value for path, value in frozen.items()):
        raise ValueError('trace inputs changed during the cohort')
    state = json.loads((args.out/'scheduler/state.json').read_text())['dnn3-marks']
    print(json.dumps(state))
    return 0 if state['status'] == 'done' else 1


if __name__ == '__main__':
    raise SystemExit(main())
