#!/usr/bin/env python3
"""Measure one fresh-process batch of MX-5 head keys under the GPU guard."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics


class GemmOperands(C.Structure):
    _fields_ = [(name, C.c_void_p) for name in ['a', 'w', 'residual', 'out', 'partial', 'tickets',
                                              'a_tokens', 'residual_tokens']] + [
        ('a_stride', C.c_int), ('residual_stride', C.c_int), ('epsilon', C.c_float)]


class HeadOperands(C.Structure):
    _fields_ = [('gemm', GemmOperands)] + [(name, C.c_void_p) for name in ['partial', 'ticket', 'tokens']] + [
        ('capacity', C.c_int), ('past', C.c_int)]


class Args(C.Structure):
    _fields_ = [('head', HeadOperands), ('counter', C.c_void_p), ('stamps', C.c_void_p),
               ('resources', C.POINTER(C.c_int))] + [
        (name, C.c_int) for name in ['n', 'k', 'b', 'iterations', 'copies']]


def run_key(key, round_number, requested_iterations, ordinal):
    import torch as t
    from tilemega.serving.weights import _packed_gpu
    path = Path(key['library']).resolve()
    meta = json.loads(path.with_suffix('.json').read_text())
    binary = hashlib.sha256(path.read_bytes()).hexdigest()
    if meta.get('body') != 'head' or not meta.get('bench') or meta.get('binary_sha256') != binary:
        raise RuntimeError('head bench binary identity mismatch: ' + str(path))
    lib = C.CDLL(str(path))
    lib.mx5_head_bench.argtypes = [C.POINTER(Args), C.c_int, C.c_void_p]
    lib.mx5_head_bench.restype = C.c_int
    lib.mx5_head_bench_coordinates.argtypes = [C.POINTER(C.c_int)]
    lib.mx5_head_bench_coordinates.restype = None
    if lib.mx5_head_bench_args_size() != C.sizeof(Args):
        raise RuntimeError('head bench argument ABI mismatch')
    coordinates = (C.c_int * 4)()
    lib.mx5_head_bench_coordinates(coordinates)
    n, k, b, tn, wn, depth = [int(key[v]) for v in ['n', 'k', 'b', 'tn', 'wn', 'depth']]
    width = depth if meta.get('pw') is None else int(meta['pw'])
    if list(coordinates) != [tn, depth, width, int(meta['threads'])] or int(key.get('pw', width)) != width:
        raise RuntimeError('head key does not match compiled coordinates')
    if b not in (1, 2, 4, 8, 16) or wn not in (4, 8) or n <= 0 or k <= 0:
        raise ValueError('invalid head key')
    copies = max(1, (256 * 1024**2 + n * k * 2 - 1) // (n * k * 2))
    iterations = max(requested_iterations, copies + 1)
    generator = t.Generator().manual_seed(3217 + ordinal)
    x = (t.randn(b, k, generator=generator) * .1).bfloat16().cuda()
    w = (t.randn(n, k, generator=generator) * .1).bfloat16().cuda()
    # Isolate argmax/tie correctness from FP32 reduction-order noise in the oracle.
    for row in range(b):
        winner = 13 + row * 17
        w[winner] = x[row] * 8
        w[winner + 1] = w[winner]
    packed = _packed_gpu(dict(kind='tile_pages', layout='xcore_mma', tile_n=tn, tile_k=32,
                             source=dict(kind='alias', source='w')), lambda _: w).contiguous().repeat(copies)
    capacity, past = 1088, 64
    partial = t.empty(b, 104, 2, dtype=t.int32, device='cuda')
    ticket = t.zeros(1, dtype=t.int32, device='cuda')
    tokens = t.full((b, capacity), -1, dtype=t.int32, device='cuda')
    counter = t.zeros(8, dtype=t.int64, device='cuda')
    stamps = t.zeros(iterations, dtype=t.int64, device='cuda')
    resources = (C.c_int * 4)()
    gemm = GemmOperands(x.data_ptr(), packed.data_ptr(), None, None, None, None, None, None, k, 0, 1e-6)
    head = HeadOperands(gemm, partial.data_ptr(), ticket.data_ptr(), tokens.data_ptr(), capacity, past)
    args = Args(head, counter.data_ptr(), stamps.data_ptr(), resources, n, k, b, iterations, copies)
    stream = C.c_void_p(t.cuda.current_stream().cuda_stream)
    observed = []
    for _ in range(2):
        counter.zero_(); ticket.zero_(); tokens.fill_(-1)
        result = lib.mx5_head_bench(C.byref(args), wn, stream)
        if result:
            raise RuntimeError(f'head bench launch failed rc={result}, key={key}')
        t.cuda.synchronize()
        if counter[0].item() != 104 * iterations or counter[1:].count_nonzero().item():
            raise RuntimeError('head barrier count mismatch')
        if ticket.item():
            raise RuntimeError('unreset head LA ticket')
        if not t.all(tokens[:, :past + 1] == -1).item() or not t.all(tokens[:, past + 2:] == -1).item():
            raise RuntimeError('head wrote outside its output token')
        observed.append(tokens[:, past + 1].clone())
    inv = t.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    expected = ((x.float() @ w.float().T) * inv).bfloat16().float().argmax(-1)
    if not t.equal(observed[0], observed[1]) or not t.equal(observed[1].long(), expected):
        raise RuntimeError('repeated head output/argmax mismatch')
    ticks = stamps.cpu().tolist()
    durations = sorted(y - x for x, y in zip(ticks, ticks[1:]))
    if not durations or durations[0] <= 0:
        raise RuntimeError('head stage timestamps did not advance')
    return dict(key=key, round=round_number, executor='mx_static_stage_bench', implementation='mx5_head_v2',
                median_ns=statistics.median(durations), p90_ns=durations[len(durations) * 9 // 10],
                iterations=iterations, copies=copies, working_set_bytes=packed.numel() * 2,
                active_working_set_bytes=packed.numel() * 2, allocated_weight_bytes=packed.numel() * 2,
                mtreg=resources[0], local_bytes=resources[1], static_smem_bytes=resources[2],
                resident_ctas=resources[3], threads=coordinates[3], source_sha256=meta['source_sha256'],
                binary_sha256=binary, base_head=meta['base_head'], prefetch=width,
                validation='argmax_and_ties_exact; repeated_tokens_identical; barrier_and_LA_counts')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--keys', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--round', type=int, choices=[1, 2, 3], required=True)
    p.add_argument('--iterations', type=int, default=200)
    args = p.parse_args()
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        p.error('run through scripts/mx_gpu_job.sh')
    if args.iterations < 3: p.error('need at least three iterations')
    if args.out.exists(): p.error('output already exists; preserve historical rows')
    import torch as t
    t.set_num_threads(4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as output:
        for ordinal, key in enumerate(json.loads(args.keys.read_text())):
            row = run_key(key, args.round, args.iterations, ordinal)
            output.write(json.dumps(row, sort_keys=True) + '\n'); output.flush()
            print(f'round={args.round} key={ordinal} median_ns={row["median_ns"]}', flush=True)


if __name__ == '__main__': main()
