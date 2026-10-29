#!/usr/bin/env python3
"""Fresh-process diagnostic timing of native router GEMM plus top-k/dispatch."""
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


class Dispatch(C.Structure):
    _fields_ = [(name, C.c_void_p) for name in ('logits', 'n_active', 'active', 'row_count', 'rows',
        'weights', 'u_gate_up', 'u_down')]


class Router(C.Structure):
    _fields_ = [(name, C.c_void_p) for name in ('a', 'w', 'residual', 'out', 'partial', 'tickets',
        'a_tokens', 'residual_tokens')] + [('a_stride', C.c_int), ('residual_stride', C.c_int),
        ('epsilon', C.c_float), ('router_ticket', C.c_void_p), ('dispatch', Dispatch)]


class Args(C.Structure):
    _fields_ = [('router', Router), ('counter', C.c_void_p), ('stamps', C.c_void_p),
        ('resources', C.POINTER(C.c_int))] + [(name, C.c_int) for name in ('n', 'k', 'b', 'iterations', 'copies')]


class LegacyTopK(C.Structure):
    _fields_ = [(name, C.c_void_p) for name in ('partial_logits', 'partial_indices', 'indices', 'weights')] + [
        (name, C.c_uint) for name in ('tokens', 'experts', 'parts')] + [
        (name, C.c_uint64) for name in ('row_stride', 'part_stride', 'rank_stride')] + [('weights_fp32', C.c_bool)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--keys', type=Path, required=True)
    p.add_argument('--legacy', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--round', type=int, choices=(1, 2, 3), required=True)
    p.add_argument('--iterations', type=int, default=128)
    a = p.parse_args()
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1' or a.out.exists() or a.iterations < 3:
        p.error('guard, fresh output and at least three iterations are required')
    import torch as t
    from tilemega.serving.weights import _packed_gpu
    t.set_num_threads(4)
    stream = t.cuda.current_stream().cuda_stream
    legacy = C.CDLL(str(a.legacy.resolve()))
    topk = legacy.mx5_legacy_topk
    topk.argtypes = [LegacyTopK, C.c_void_p]
    topk.restype = C.c_int
    with a.out.open('x') as f:
        for ordinal, key in enumerate(json.loads(a.keys.read_text())):
            so = Path(key['library'])
            if hashlib.sha256(so.read_bytes()).hexdigest() != key['binary_sha256']:
                raise ValueError('router benchmark binary changed')
            lib = C.CDLL(str(so))
            if lib.mx5_moe_router_bench_args_size() != C.sizeof(Args):
                raise ValueError('router benchmark ABI differs')
            fn = lib.mx5_moe_router_bench
            fn.argtypes = [C.POINTER(Args), C.c_void_p]
            fn.restype = C.c_int
            b, tn = key['b'], key['tn']
            generator = t.Generator().manual_seed(2030 + ordinal)
            x_cpu = t.randn((b, 2048), generator=generator).bfloat16()
            w_cpu = (t.randn((128, 2048), generator=generator) / math.sqrt(2048)).bfloat16()
            copies = math.ceil(256 * 1024**2 / (128 * 2048 * 2))
            iterations = max(a.iterations, copies + 2)
            recipe = dict(kind='tile_pages', layout='xcore_mma', tile_n=tn, tile_k=32,
                source=dict(kind='alias', source='matrix'))
            packed = _packed_gpu(recipe, lambda _: w_cpu).cuda().repeat(copies)
            x = x_cpu.cuda()
            logits = t.full((b, 128), -93, dtype=t.bfloat16, device='cuda')
            tiles = 128 // tn
            partial = t.empty((tiles, (104 + tiles - 1) // tiles + 1, b, tn + 1), device='cuda')
            tickets = t.zeros(tiles, dtype=t.int32, device='cuda')
            route_ticket = t.zeros(16, dtype=t.int32, device='cuda')
            active = t.full((128,), -1, dtype=t.int32, device='cuda')
            counts = active.clone()
            mapping = t.full((128, b, 2), -1, dtype=t.int32, device='cuda')
            weights = t.empty((b, 8), dtype=t.float32, device='cuda')
            na = t.zeros(1, dtype=t.int32, device='cuda')
            ug, ud = na.clone(), na.clone()
            dispatch = Dispatch(*[v.data_ptr() for v in (logits, na, active, counts, mapping, weights, ug, ud)])
            router = Router(x.data_ptr(), packed.data_ptr(), None, logits.data_ptr(), partial.data_ptr(), tickets.data_ptr(),
                None, None, 2048, 0, 1e-6, route_ticket.data_ptr(), dispatch)
            counter = t.zeros(8, dtype=t.int64, device='cuda')
            stamps = t.zeros(iterations, dtype=t.int64, device='cuda')
            resources = (C.c_int * 4)()
            args = Args(router, counter.data_ptr(), stamps.data_ptr(), resources, 128, 2048, b, 0, copies)
            rc = fn(C.byref(args), stream)
            if rc or resources[1] or resources[2] or resources[3] != 1:
                raise RuntimeError(f'router resource query rejected {key}: {rc}, {list(resources)}')
            args.iterations = iterations
            previous = None
            durations = []
            for _ in range(2):
                counter.zero_()
                tickets.zero_()
                route_ticket.zero_()
                rc = fn(C.byref(args), stream)
                if rc:
                    raise RuntimeError(f'router bench launch failed: {rc}, {key}')
                t.cuda.synchronize()
                current = [v.cpu().clone() for v in (logits, active, counts, mapping, weights, na, ug, ud)]
                if previous is not None and not all(t.equal(x, y) for x, y in zip(previous, current)):
                    raise AssertionError('router output/dispatch is not deterministic')
                previous = current
                values = stamps.cpu().tolist()
                durations = [v - u for u, v in zip(values, values[1:])]
                if not all(v > 0 for v in durations) or not bool(t.all(tickets == 0) and t.all(route_ticket == 0)):
                    raise AssertionError('timestamps or router ticket reset failed')
            expected = ((x_cpu.float() @ w_cpu.float().T) * t.rsqrt(x_cpu.float().square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
            t.testing.assert_close(logits.cpu().float(), expected.float(), rtol=.012, atol=.02)
            indices = t.empty((b, 8), dtype=t.int32, device='cuda')
            reference_weights = t.empty_like(weights)
            pidx = t.arange(128, dtype=t.int32, device='cuda').repeat(b, 1)
            plog = logits.float()
            lp = LegacyTopK(plog.data_ptr(), pidx.data_ptr(), indices.data_ptr(), reference_weights.data_ptr(),
                b, 128, 16, 128, 8, 1, True)
            if topk(lp, stream):
                raise RuntimeError('legacy top-k reference launch failed')
            t.cuda.synchronize()
            if not t.equal(weights, reference_weights):
                raise AssertionError('router weights differ from actual legacy body')
            chosen = indices.cpu()
            experts = sorted(set(chosen.flatten().tolist()))
            if na.item() != len(experts) or active[:len(experts)].cpu().tolist() != experts:
                raise AssertionError('active expert list differs')
            if ug.item() != len(experts) * 24 or ud.item() != len(experts) * 32:
                raise AssertionError('dispatch unit counts differ from compiled TN64/64')
            actual_mapping, actual_counts = mapping.cpu(), counts.cpu()
            for expert_index, expert in enumerate(experts):
                pairs = [(token, rank) for token in range(b) for rank in range(8) if chosen[token, rank].item() == expert]
                if actual_counts[expert_index].item() != len(pairs) or actual_mapping[expert_index, :len(pairs)].tolist() != [list(v) for v in pairs]:
                    raise AssertionError('dispatch rows differ')
            row = dict(key=key, round=a.round, PID=os.getpid(), executor='mx_static_stage_bench', implementation='MxRouterGemmV2',
                copies=copies, addressed_rotating_weight_bytes=copies * 128 * 2048 * 2,
                median_ns=statistics.median(durations), p90_ns=sorted(durations)[math.ceil(.9 * len(durations)) - 1],
                spread_ns=max(durations) - min(durations), iterations=iterations,
                resources=dict(MTreg=resources[0], stack=resources[1], static_smem=resources[2], resident=resources[3]),
                FP32_reference_passed=True, legacy_topk_weights_bitwise=True, dispatch_verified=True, repeat_bitwise=True,
                diagnostic_only=True, qualified=False, final=False)
            f.write(json.dumps(row) + '\n')
            f.flush()
            print('MOE_ROUTER_STAGE_KEY_PASS', ordinal, b, key['stream_k'], key['pw'], row['median_ns'], flush=True)
            del packed, x, logits, partial, tickets, route_ticket, active, counts, mapping, weights, na, ug, ud, counter, stamps, plog, pidx, indices, reference_weights
            t.cuda.empty_cache()


if __name__ == '__main__':
    main()
