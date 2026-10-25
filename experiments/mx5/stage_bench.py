#!/usr/bin/env python3
"""One fresh-process batch of MX-5 stage keys; raw output stays in runs/."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics


class Operands(C.Structure):
    _fields_ = [(name, C.c_void_p) for name in ['a', 'w', 'residual', 'out', 'partial', 'tickets',
                                              'a_tokens', 'residual_tokens']] + [
        ('a_stride', C.c_int), ('residual_stride', C.c_int), ('epsilon', C.c_float)]


class Args(C.Structure):
    _fields_ = [('gemm', Operands), ('counter', C.c_void_p), ('stamps', C.c_void_p),
               ('resources', C.POINTER(C.c_int))] + [
        (name, C.c_int) for name in ['n', 'k', 'b', 'iterations', 'copies']]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--keys', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--round', type=int, choices=[1, 2, 3], required=True)
    p.add_argument('--iterations', type=int, default=200)
    a = p.parse_args()
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1': p.error('run through scripts/mx_gpu_job.sh')
    if a.iterations < 3: p.error('need at least three iterations')
    if a.out.exists(): p.error('output already exists; preserve historical rows')
    import torch as t
    from tilemega.serving.weights import _packed_gpu
    t.set_num_threads(4)
    stream = C.c_void_p(t.cuda.current_stream().cuda_stream)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as output:
        for ordinal, key in enumerate(json.loads(a.keys.read_text())):
            path = Path(key['library']).resolve(); meta = json.loads(path.with_suffix('.json').read_text())
            binary = hashlib.sha256(path.read_bytes()).hexdigest()
            if not meta.get('bench') or meta.get('binary_sha256') != binary:
                raise RuntimeError('bench binary identity mismatch: ' + str(path))
            lib = C.CDLL(str(path))
            if lib.mx5_bench_args_size() != C.sizeof(Args): raise RuntimeError('bench argument ABI mismatch')
            n, k, b, tn = [int(key[v]) for v in ['n', 'k', 'b', 'tn']]
            wn, op, norm, part = [int(key[v]) for v in ['wn', 'op', 'norm', 'part']]
            copies = max(1, (256 * 1024**2 + n * k * 2 - 1) // (n * k * 2))
            generator = t.Generator().manual_seed(2017 + ordinal)
            x = (t.randn(b, k, generator=generator) * .1).bfloat16().cuda()
            w = (t.randn(n, k, generator=generator) * .1).bfloat16().cuda()
            packed = _packed_gpu(dict(kind='tile_pages', layout='xcore_mma', tile_n=tn, tile_k=32,
                                     source=dict(kind='alias', source='w')), lambda _: w).contiguous().repeat(copies)
            residual = (t.randn(b, n, generator=generator) * .1).bfloat16().cuda()
            out = t.empty(b, n // 2 if op == 2 else n, dtype=t.bfloat16, device='cuda')
            tiles = n // tn
            slots = (104 + tiles - 1) // tiles + 1
            partial = t.empty(tiles * slots * b * (tn + 1), device='cuda')
            tickets = t.zeros(tiles, dtype=t.int32, device='cuda')
            counter = t.zeros(8, dtype=t.int64, device='cuda')
            stamps = t.zeros(a.iterations, dtype=t.int64, device='cuda')
            resources = (C.c_int * 4)()
            ops = Operands(*[v.data_ptr() for v in [x, packed, residual, out, partial, tickets]],
                           None, None, k, n, 1e-6)
            args = Args(ops, counter.data_ptr(), stamps.data_ptr(), resources, n, k, b, a.iterations, copies)
            for repeat in range(2):
                counter.zero_(); tickets.zero_()
                result = lib.mx5_bench(C.byref(args), wn, op, norm, part, stream)
                if result: raise RuntimeError(f'bench launch failed rc={result}, key={key}')
                t.cuda.synchronize()
            if counter[0].item() != 104 * a.iterations: raise RuntimeError('barrier count mismatch')
            if tickets.abs().sum().item(): raise RuntimeError('unreset LA tickets')
            reference = x.float() @ w.float().T
            if norm: reference *= t.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
            reference = reference.bfloat16()
            if op == 1: reference = (reference.float() + residual.float()).bfloat16()
            elif op == 2:
                pair = reference.reshape(b, -1, 2, 16)
                gate, up = pair[:, :, 0].float(), pair[:, :, 1].float()
                reference = ((gate / (1 + (-gate).exp())).bfloat16().float() * up).bfloat16().reshape(b, -1)
            if not t.all((out.float() - reference.float()).abs() <= .02 + .016 * reference.float().abs()).item():
                raise RuntimeError('repeated-stage output mismatch')
            ticks = stamps.cpu().tolist(); durations = sorted(y - x for x, y in zip(ticks, ticks[1:]))
            row = dict(key=key, round=a.round, executor='mx_static_stage_bench', implementation='mx5_gemm_v2',
                       median_ns=statistics.median(durations), p90_ns=durations[len(durations) * 9 // 10],
                       iterations=a.iterations, copies=copies, working_set_bytes=packed.numel() * 2,
                       mtreg=resources[0], local_bytes=resources[1], static_smem_bytes=resources[2],
                       resident_ctas=resources[3], threads=meta['threads'], source_sha256=meta['source_sha256'],
                       binary_sha256=binary, base_head=meta['base_head'],
                       prefetch=meta['depth'] if meta['pw'] is None else meta['pw'])
            output.write(json.dumps(row, sort_keys=True) + '\n'); output.flush()
            print(f'round={a.round} key={ordinal} median_ns={row["median_ns"]}', flush=True)


if __name__ == '__main__': main()
