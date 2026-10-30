#!/usr/bin/env python3
"""Measure one fresh-process batch of MX-5 attention keys under the GPU guard."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics


class Operands(C.Structure):
    _fields_ = [(name, C.c_void_p) for name in ['qkv', 'q_norm', 'k_norm', 'cosine', 'sine',
                                              'k_cache', 'v_cache', 'ctx', 'partial', 'tickets']] + [
        ('capacity', C.c_int), ('past', C.c_int), ('epsilon', C.c_float)]


class Args(C.Structure):
    _fields_ = [('attention', Operands), ('counter', C.c_void_p), ('stamps', C.c_void_p),
               ('resources', C.POINTER(C.c_int))] + [
        (name, C.c_int) for name in ['b', 'hkv', 'iterations', 'copies']]


def layout_indices(dim, capacity, torch):
    token = torch.arange(capacity, device='cuda')[:, None]
    column = torch.arange(dim, device='cuda')[None, :]
    local = token % 32
    lane = 4 * (local // 8) + local % 4 + 16 * ((column % 32) // 8)
    key = ((token // 32 * (dim // 32) * 2 + column // 32 * 2 + local % 8 // 4) * 512
           + lane * 8 + column % 8)
    value = ((token // 32 * (dim // 16) + column // 16) * 512
             + (column % 16 + 16 * (local // 8)) * 8 + local % 8)
    return key.reshape(-1), value.reshape(-1)


def transform_reference(x, weight, cosine, sine, norm, torch):
    dim = x.shape[-1]
    if norm == 2:
        rows=x.reshape(-1,16*dim).float()
        inverse=torch.rsqrt(rows.square().mean(-1,keepdim=True)+1e-6)
        x=(rows*inverse).bfloat16().reshape(x.shape)
        x=(x.reshape(-1,16*dim).float()*weight.reshape(1,16*dim).float()).bfloat16().reshape(x.shape)
    elif norm:
        groups = x.float().reshape(*x.shape[:-1], 4, 4, 8)
        squares = torch.zeros_like(groups[..., 0])
        for item in range(8): squares = squares + groups[..., item].square()
        lane = (squares[..., 0, :] + squares[..., 2, :]) + (squares[..., 1, :] + squares[..., 3, :])
        total = (lane[..., 0] + lane[..., 2]) + (lane[..., 1] + lane[..., 3])
        inverse = torch.rsqrt(total / dim + 1e-6)
        x = (x.float() * inverse[..., None]).bfloat16()
        x = (x.float() * weight.float()).bfloat16()
    partner = torch.cat((-x[..., dim // 2:].float(), x[..., :dim // 2].float()), -1)
    return ((x.float() * cosine.float()).bfloat16().float()
            + (partner * sine.float()).bfloat16().float()).bfloat16()


def run_key(key, round_number, requested_iterations, ordinal):
    import torch as t
    path = Path(key['library']).resolve()
    meta = json.loads(path.with_suffix('.json').read_text())
    binary = hashlib.sha256(path.read_bytes()).hexdigest()
    if meta.get('body') != 'attention' or not meta.get('bench') or meta.get('binary_sha256') != binary:
        raise RuntimeError('attention bench binary identity mismatch: ' + str(path))
    lib = C.CDLL(str(path))
    lib.mx5_attention_bench.argtypes = [C.POINTER(Args), C.c_int, C.c_int, C.c_void_p]
    lib.mx5_attention_bench.restype = C.c_int
    lib.mx5_attention_bench_coordinates.argtypes = [C.POINTER(C.c_int)]
    lib.mx5_attention_bench_coordinates.restype = None
    if lib.mx5_attention_bench_args_size() != C.sizeof(Args):
        raise RuntimeError('attention bench argument ABI mismatch')
    coordinates = (C.c_int * 4)()
    lib.mx5_attention_bench_coordinates(coordinates)
    dim, query, b, hkv, length, bmin = [int(key[v]) for v in ['dim', 'q', 'b', 'hkv', 'length', 'b_min']]
    norm, kind, capacity = int(key['qk_norm']), int(key['prefetch_kind']), int(key.get('capacity', 1088))
    if list(coordinates) != [dim, query, kind, int(meta['threads'])]:
        raise RuntimeError('attention key does not match compiled coordinates')
    if (b not in (1, 2, 4, 8, 16) or hkv < 1 or bmin not in (1, 2, 4, 8) or norm not in (0, 1, 2)
            or (norm and dim != 128) or capacity != 1088 or not 1 <= length <= capacity):
        raise ValueError('invalid attention key')
    if norm==2 and (hkv!=16 or query!=1 or meta.get('row_norm_heads')!=16):
        raise ValueError('whole-row attention requires its own 16-head one-query binary')
    if norm!=2 and meta.get('row_norm_heads',0):
        raise ValueError('per-head/no-norm key cannot use a row-normalizing binary')
    heads = b * hkv
    copy_bytes = heads * capacity * dim * 4
    active_bytes = heads * ((length + 31) // 32) * 32 * dim * 4
    # Count the blocks read, not unused capacity, toward the cold working set.
    copies = max(1, (256 * 1024**2 + active_bytes - 1) // active_bytes)
    iterations = max(requested_iterations, copies + 1)
    generator = t.Generator().manual_seed(4217 + ordinal)
    raw = (t.randn(heads, query + 2, dim, generator=generator) * .2).bfloat16().cuda()
    keys = (t.randn(heads, capacity, dim, generator=generator) * .2).bfloat16().cuda()
    values = (t.randn(heads, capacity, dim, generator=generator) * .2).bfloat16().cuda()
    weights = (1 + .1 * t.randn(2, dim*(hkv if norm==2 else 1), generator=generator)).bfloat16().cuda()
    angle = t.linspace(-2., 2., dim // 2).repeat(2)
    cosine, sine = [f(angle).repeat(capacity, 1).bfloat16().cuda() for f in (t.cos, t.sin)]
    ki, vi = layout_indices(dim, capacity, t)
    packed_k = t.empty(copies, heads, capacity * dim, dtype=t.bfloat16, device='cuda')
    packed_v = t.empty_like(packed_k)
    packed_k[:, :, ki] = keys.reshape(1, heads, -1)
    packed_v[:, :, vi] = values.reshape(1, heads, -1)
    ctx = t.empty(heads, query, dim, dtype=t.bfloat16, device='cuda')
    partial = t.empty(heads * 34 * query * (dim + 2), device='cuda')
    tickets = t.zeros(heads, dtype=t.int32, device='cuda')
    counter = t.zeros(8, dtype=t.int64, device='cuda')
    stamps = t.zeros(iterations, dtype=t.int64, device='cuda')
    resources = (C.c_int * 4)()
    ops = Operands(raw.data_ptr(), weights[0].data_ptr(), weights[1].data_ptr(), cosine.data_ptr(), sine.data_ptr(),
                   packed_k.data_ptr(), packed_v.data_ptr(), ctx.data_ptr(), partial.data_ptr(), tickets.data_ptr(),
                   capacity, length - 1, 1e-6)
    args = Args(ops, counter.data_ptr(), stamps.data_ptr(), resources, b, hkv, iterations, copies)
    stream = C.c_void_p(t.cuda.current_stream().cuda_stream)
    observed = []
    for _ in range(2):
        counter.zero_(); tickets.zero_(); ctx.fill_(float('nan'))
        result = lib.mx5_attention_bench(C.byref(args), bmin, norm, stream)
        if result:
            raise RuntimeError(f'attention bench launch failed rc={result}, key={key}')
        t.cuda.synchronize()
        if counter[0].item() != 104 * iterations or counter[1:].count_nonzero().item():
            raise RuntimeError('attention barrier count mismatch')
        if tickets.count_nonzero().item():
            raise RuntimeError('unreset attention LA tickets')
        observed.append(ctx.clone())
    q = transform_reference(raw[:, :query], weights[0], cosine[length - 1], sine[length - 1], norm, t)
    new_key = transform_reference(raw[:, query], weights[1], cosine[length - 1], sine[length - 1], norm, t)
    expected_k = keys.clone(); expected_k[:, length - 1] = new_key
    expected_v = values.clone(); expected_v[:, length - 1] = raw[:, query + 1]
    score = t.einsum('hqd,htd->hqt', q.float(), expected_k[:, :length].float()) / dim**.5
    expected_ctx = t.einsum('hqt,htd->hqd', score.softmax(-1), expected_v[:, :length].float()).bfloat16()
    error = (ctx.float() - expected_ctx.float()).abs()
    if not t.all(error <= .006 + .016 * expected_ctx.float().abs()).item():
        raise RuntimeError(f'repeated attention context mismatch, max error={error.max().item()}')
    if not t.equal(observed[0].view(t.int16), observed[1].view(t.int16)):
        raise RuntimeError('repeated attention context differs bitwise')
    # Every copy was visited. Exact history/value checks also catch writes beyond the owned block.
    for copy in range(copies):
        actual_k = packed_k[copy, :, ki].reshape(heads, capacity, dim)
        actual_v = packed_v[copy, :, vi].reshape(heads, capacity, dim)
        if not t.equal(actual_v.view(t.int16), expected_v.view(t.int16)):
            raise RuntimeError('attention V cache bits differ')
        for region in (slice(None, length - 1), slice(length, None)):
            if not t.equal(actual_k[:, region].view(t.int16), keys[:, region].view(t.int16)):
                raise RuntimeError('attention changed historical K cache bits')
        delta = (actual_k[:, length - 1].float() - new_key.float()).abs()
        if not t.all(delta <= .016 + .016 * new_key.float().abs()).item():
            raise RuntimeError('attention new K differs from the FP32 transform oracle')
    ticks = stamps.cpu().tolist()
    durations = sorted(y - x for x, y in zip(ticks, ticks[1:]))
    if not durations or durations[0] <= 0:
        raise RuntimeError('attention stage timestamps did not advance')
    return dict(key=key, round=round_number, executor='mx_static_stage_bench', implementation='mx5_attention_v2',
                median_ns=statistics.median(durations), p90_ns=durations[len(durations) * 9 // 10],
                iterations=iterations, copies=copies, working_set_bytes=active_bytes * copies,
                active_working_set_bytes=active_bytes * copies, allocated_kv_bytes=copy_bytes * copies,
                mtreg=resources[0], local_bytes=resources[1], static_smem_bytes=resources[2],
                resident_ctas=resources[3], threads=coordinates[3], source_sha256=meta['source_sha256'],
                binary_sha256=binary, base_head=meta['base_head'], prefetch_kind=kind,
                validation=dict(context_atol=.006, context_rtol=.016, new_k_atol=.016, new_k_rtol=.016,
                                historical_kv_bits=True, repeated_context_bits=True, barrier_and_LA_counts=True))


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
