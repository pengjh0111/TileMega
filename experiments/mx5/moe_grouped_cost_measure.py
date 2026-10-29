#!/usr/bin/env python3
"""Source-bound grouped-stage cost samples; aggregate only three verified processes."""
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


class Unit(C.Structure):
    _fields_ = [(name, C.c_uint) for name in ('active', 'expert', 'tile', 'count', 'input_row')]


class Combine(C.Structure):
    _fields_ = [(name, C.c_void_p) for name in ('partials', 'weights', 'residual', 'output', 'tickets')]


class Operands(C.Structure):
    _fields_ = [(name, C.c_void_p) for name in ('a', 'w', 'out', 'active', 'row_count', 'rows')] + [
        ('units', C.c_uint), ('expert_stride', C.c_uint), ('epsilon', C.c_float), ('unit', Unit), ('combine', Combine)]


class Args(C.Structure):
    _fields_ = [('grouped', Operands), ('counter', C.c_void_p), ('stamps', C.c_void_p),
        ('resources', C.POINTER(C.c_int))] + [(name, C.c_int) for name in ('n', 'k', 'b', 'iterations', 'copies')]


def reference_rows(batch):
    active = {1: 8, 2: 16, 4: 29, 8: 52, 16: 82}[batch]
    rows = [[] for _ in range(active)]
    for token in range(batch):
        for rank in range(8):
            rows[(token * 8 + rank) % active].append((token, rank))
    assert sorted(pair for row in rows for pair in row) == [(b, r) for b in range(batch) for r in range(8)]
    assert max(map(len, rows)) <= batch and max(map(len, rows)) - min(map(len, rows)) <= 1
    return rows



def validate_cost_inputs(epoch_path, keys_path):
    def checked(record):
        path = Path(record['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('cost input changed: ' + str(path))
        return path
    epoch = json.loads(epoch_path.read_text())
    root = Path(epoch['root'])
    if epoch.get('kind') != 'mx5_moe_grouped_source_bound_cost_epoch' or epoch.get('final') is not False:
        raise ValueError('expected a source-bound grouped cost epoch')
    if epoch['R_cap'] != {'t512': 248, 't576': 220}:
        raise ValueError('cost epoch changes the prompt register budget')
    checked(epoch['compiler'])
    if checked(epoch['driver']).resolve() != Path(__file__).resolve():
        raise ValueError('cost driver does not match its epoch')
    for name, digest in epoch['inputs'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError('frozen cost source changed: ' + name)
    for query in epoch['query_records']:
        checked(query)
    for key in json.loads(keys_path.read_text()):
        if key['source_digest'] != epoch['source_digest']:
            raise ValueError('key belongs to a different source epoch')
        build = json.loads(checked(key['build_record']).read_text())
        if build['status'] != 'CPU_resource_fit' or build['source_digest'] != epoch['source_digest']:
            raise ValueError('cost key lacks a fitted source-bound build')
        for name in ('threads', 'gate_up', 'tn', 'wn', 'd', 'pw'):
            if key[name] != build[name]:
                raise ValueError('key/build coordinate differs: ' + name)
        if checked(build['binary']).resolve() != Path(key['library']).resolve() or build['binary']['sha256'] != key['binary_sha256']:
            raise ValueError('key/build binary differs')
        checked(build['MIR'])
        matched = next((row for row in epoch['eligible_keys'] if row['tag'] == build['tag'] and row['b'] == key['b']), None)
        if matched is None or any(matched[name] != key[name] for name in ('n', 'k', 'n_active', 'threads', 'gate_up', 'tn', 'wn', 'd', 'pw', 'r_next')):
            raise ValueError('cost key lacks its actual device query')
        if matched['local_bytes'] or matched['static_smem'] or matched['resident'] != 1:
            raise ValueError('cost key has rejected device resources')
    return dict(source_digest=epoch['source_digest'], epoch_sha256=hashlib.sha256(epoch_path.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--keys', type=Path, required=True)
    p.add_argument('--epoch', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--round', type=int, choices=(1, 2, 3), required=True)
    p.add_argument('--iterations', type=int, default=128)
    a = p.parse_args()
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        p.error('run through scripts/mx_gpu_job.sh')
    if a.out.exists() or a.iterations < 3:
        p.error('fresh output and at least three iterations are required')
    scope = validate_cost_inputs(a.epoch, a.keys)
    import torch as t
    from tilemega.serving.weights import _packed_gpu
    t.set_num_threads(4)
    stream = t.cuda.current_stream().cuda_stream
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as output:
        for ordinal, key in enumerate(json.loads(a.keys.read_text())):
            library = Path(key['library'])
            if hashlib.sha256(library.read_bytes()).hexdigest() != key['binary_sha256']:
                raise ValueError('changed grouped benchmark binary')
            lib = C.CDLL(str(library))
            if lib.mx5_moe_grouped_bench_args_size() != C.sizeof(Args):
                raise ValueError('grouped benchmark ABI differs')
            fn = lib.mx5_moe_grouped_bench
            fn.argtypes = [C.POINTER(Args), C.c_void_p]
            fn.restype = C.c_int
            b, n, k, tn, wn, gu = (int(key[name]) for name in ('b', 'n', 'k', 'tn', 'wn', 'gate_up'))
            rows = reference_rows(b)
            active_count = len(rows)
            copies = max(1, math.ceil(256 * 1024**2 / (active_count * n * k * 2)))
            generator = t.Generator().manual_seed(2030 + ordinal)
            # Copies cover the actually addressed expert bytes, not unused stack rows.
            w_cpu = (t.randn((128, n, k), generator=generator) / math.sqrt(k)).bfloat16()
            recipe = dict(kind='tile_pages', layout='xcore_mma', tile_n=tn, tile_k=32,
                source=dict(kind='alias', source='matrix'))
            packed = t.stack([_packed_gpu(recipe, lambda _, e=e: w_cpu[e]) for e in range(128)]).cuda().repeat(copies, 1)
            counts = t.tensor([len(row) for row in rows], dtype=t.int32, device='cuda')
            active = t.arange(active_count, dtype=t.int32, device='cuda')
            mapping = t.zeros((active_count, b, 2), dtype=t.int32)
            for expert, pairs in enumerate(rows):
                mapping[expert, :len(pairs)] = t.tensor(pairs, dtype=t.int32)
            mapping_gpu = mapping.cuda()
            x_cpu = t.randn((b, k) if gu else (active_count, b, k), generator=generator).bfloat16()
            x = x_cpu.cuda()
            out = t.full((active_count, b, n // 2) if gu else (b, 8, n), -93, dtype=t.bfloat16, device='cuda')
            residual = t.randn((b, 2048), generator=generator).bfloat16().cuda()
            merged = t.empty_like(residual)
            weights = t.full((b, 8), 1 / 8, dtype=t.float32, device='cuda')
            ticket = t.zeros(2048 // tn, dtype=t.int32, device='cuda')
            counter = t.zeros(8, dtype=t.int64, device='cuda')
            iterations = max(a.iterations, copies + 2)
            stamps = t.zeros(iterations, dtype=t.int64, device='cuda')
            resources = (C.c_int * 4)()
            cp = Combine(out.data_ptr(), weights.data_ptr(), residual.data_ptr(), merged.data_ptr(), ticket.data_ptr())
            ops = Operands(*[value.data_ptr() for value in (x, packed, out, active, counts, mapping_gpu)],
                active_count * (n // tn), n * k, 1e-6, Unit(), cp)
            args = Args(ops, counter.data_ptr(), stamps.data_ptr(), resources, n, k, b, 0, copies)
            rc = fn(C.byref(args), stream)
            if rc or resources[1] or resources[2] or resources[3] != 1:
                raise RuntimeError(f'grouped resource query rejected {key}: {rc}, {list(resources)}')
            args.iterations = iterations
            measurements = []
            previous = previous_merged = None
            for repeat in range(2):
                counter.zero_()
                ticket.zero_()
                rc = fn(C.byref(args), stream)
                if rc:
                    raise RuntimeError(f'grouped stage launch rejected {key}: {rc}')
                t.cuda.synchronize()
                if not bool(t.all(ticket == 0)):
                    raise AssertionError('combine tickets did not reset')
                values = stamps.cpu().tolist()
                durations = [v - u for u, v in zip(values, values[1:])]
                if not all(value > 0 for value in durations):
                    raise AssertionError('non-monotonic benchmark timestamps')
                measurements = durations
                actual = out.cpu()
                if previous is not None and not t.equal(actual, previous):
                    t.save(dict(previous=previous, actual=actual, key=key, rows=rows), a.out.with_suffix('.repeat_failure.pt'))
                    raise AssertionError('grouped result is not deterministic')
                if not gu and previous_merged is not None and not t.equal(merged, previous_merged):
                    raise AssertionError('combine result is not deterministic')
                previous = actual
                previous_merged = merged.clone() if not gu else None
            for expert, pairs in enumerate(rows):
                index = t.tensor(pairs, dtype=t.int64)
                source = x_cpu[index[:, 0]] if gu else x_cpu[expert, :len(pairs)]
                reference = source.float() @ w_cpu[expert].float().T
                if gu:
                    reference *= t.rsqrt(source.float().square().mean(-1, keepdim=True) + 1e-6)
                    pair = reference.bfloat16().reshape(len(pairs), -1, 2, 16)
                    gate, up = pair[:, :, 0].float(), pair[:, :, 1].float()
                    expected = ((gate / (1 + (-gate).exp())).bfloat16().float() * up).bfloat16().reshape(len(pairs), -1)
                    actual = previous[expert, :len(pairs)]
                else:
                    expected = reference.bfloat16()
                    actual = previous[index[:, 0], index[:, 1]]
                t.testing.assert_close(actual.float(), expected.float(), rtol=.012, atol=.02)
            if not gu:
                total = t.zeros((b, n), dtype=t.float32)
                for rank in range(8):
                    total = total + previous[:, rank].float() * (1 / 8)
                expected_merged = (total.bfloat16().float() + residual.cpu().float()).bfloat16()
                if not t.equal(merged.cpu(), expected_merged):
                    raise AssertionError('combine rounding or rank order differs')
            row = dict(key=key, round=a.round, PID=os.getpid(), executor='mx_static_stage_bench',
                implementation='mx5_moe_grouped_v2', body_type='MxGroupedGemm', seed=2030 + ordinal, n_active=active_count, row_counts=[len(row) for row in rows],
                copies=copies, addressed_rotating_weight_bytes=copies * active_count * n * k * 2,
                median_ns=statistics.median(measurements), p90_ns=sorted(measurements)[math.ceil(.9 * len(measurements)) - 1],
                spread_ns=max(measurements) - min(measurements), iterations=iterations,
                resources=dict(MTreg=resources[0], stack=resources[1], static_smem=resources[2], resident=resources[3]),
                reference_passed=True, repeat_bitwise=True, cost_protocol=True, source_bound=True, source_digest=scope['source_digest'],
                epoch_sha256=scope['epoch_sha256'], driver_sha256=scope['driver_sha256'],
                raw_sample_only=True, final=False)
            output.write(json.dumps(row) + '\n')
            output.flush()
            print('MOE_GROUPED_STAGE_KEY_PASS', ordinal, b, gu, tn, wn, row['median_ns'], flush=True)
            del packed, x, out, active, counts, mapping_gpu, residual, merged, weights, ticket, counter, stamps, previous_merged
            t.cuda.empty_cache()


if __name__ == '__main__':
    main()
