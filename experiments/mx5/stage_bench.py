#!/usr/bin/env python3
"""One fresh-process batch of MX-5 stage keys; raw output stays in runs/."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics


REFERENCE_BACKEND = 'torch_cpu'
REFERENCE_PRECISION = 'float32'
REFERENCE_ORACLE_VERSION = 'gemm_cpu_fp32_v1'
REFERENCE_ORACLE_DESCRIPTION = (
    'CPU BF16 sources; CPU FP32 matmul/RMS/SiLU; original BF16 rounding points; device comparison'
)


def cpu_fp32_reference(x, w, residual, op, norm, t):
    if any(value.device.type != 'cpu' or value.dtype != t.bfloat16 for value in (x, w, residual)):
        raise ValueError('reference inputs must be CPU BF16 source tensors')
    b = x.shape[0]
    reference = x.float() @ w.float().T
    if norm: reference *= t.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    reference = reference.bfloat16()
    if op == 1: reference = (reference.float() + residual.float()).bfloat16()
    elif op == 2:
        pair = reference.reshape(b, -1, 2, 16)
        gate, up = pair[:, :, 0].float(), pair[:, :, 1].float()
        reference = ((gate / (1 + (-gate).exp())).bfloat16().float() * up).bfloat16().reshape(b, -1)
    return reference


class Operands(C.Structure):
    _fields_ = [(name, C.c_void_p) for name in ['a', 'w', 'residual', 'out', 'partial', 'tickets',
                                              'a_tokens', 'residual_tokens']] + [
        ('a_stride', C.c_int), ('residual_stride', C.c_int), ('epsilon', C.c_float)]


class Args(C.Structure):
    _fields_ = [('gemm', Operands), ('counter', C.c_void_p), ('stamps', C.c_void_p),
               ('resources', C.POINTER(C.c_int))] + [
        (name, C.c_int) for name in ['n', 'k', 'b', 'iterations', 'copies']]


def validate_metadata(key, meta):
    if meta.get('bench') is not True or meta.get('body') != 'gemm' or meta.get('returncode') != 0:
        raise RuntimeError('expected a successful GEMM stage-bench build')

    def integer(value, label):
        if type(value) is not int:
            raise RuntimeError(f'{label} must be an integer')
        return value

    compiled = {name: integer(meta.get(name), f'build {name}') for name in ['tn', 'depth', 'threads']}
    compiled['pw'] = compiled['depth'] if meta.get('pw') is None else integer(meta['pw'], 'build pw')
    names = dict(MX5_TEST_TN='tn', MX5_TEST_DEPTH='depth', MX5_TEST_THREADS='threads', MX5_BENCH_PW='pw')
    command = meta.get('command')
    if not isinstance(command, list) or not all(isinstance(arg, str) for arg in command):
        raise RuntimeError('missing build command')
    definitions = {}
    arguments = iter(command)
    for argument in arguments:
        if argument == '-D':
            argument = next(arguments, '')
        elif argument.startswith('-D'):
            argument = argument[2:]
        else:
            continue
        macro, _, value = argument.partition('=')
        if macro not in names:
            continue
        name = names[macro]
        if name in definitions:
            raise RuntimeError(f'duplicate build definition: {macro}')
        try:
            definitions[name] = int(value)
        except ValueError as error:
            raise RuntimeError(f'invalid build definition: {macro}') from error
    # An omitted MX5_BENCH_PW uses MX5_TEST_DEPTH in the compiled source.
    definitions.setdefault('pw', definitions.get('depth'))
    for name, actual in compiled.items():
        if definitions.get(name) != actual:
            raise RuntimeError(f'build metadata/command mismatch for {name}')
        if name == 'tn' or name in key:
            declared = integer(key.get(name), f'key {name}')
            if declared != actual:
                raise RuntimeError(f'bench key/build mismatch for {name}: key={declared}, build={actual}')
    return compiled


def load_bench(key):
    path = Path(key['library']).resolve()
    meta = json.loads(path.with_suffix('.json').read_text())
    compiled = validate_metadata(key, meta)
    binary = hashlib.sha256(path.read_bytes()).hexdigest()
    if meta.get('binary_sha256') != binary:
        raise RuntimeError('bench binary identity mismatch: ' + str(path))
    lib = C.CDLL(str(path))
    lib.mx5_bench_args_size.argtypes = []
    lib.mx5_bench_args_size.restype = C.c_int
    if lib.mx5_bench_args_size() != C.sizeof(Args):
        raise RuntimeError('bench argument ABI mismatch')
    lib.mx5_bench.argtypes = [C.POINTER(Args), C.c_int, C.c_int, C.c_int, C.c_int, C.c_void_p]
    lib.mx5_bench.restype = C.c_int
    return lib, meta, binary, compiled


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
            lib, meta, binary, compiled = load_bench(key)
            n, k, b, tn = [int(key[v]) for v in ['n', 'k', 'b', 'tn']]
            wn, op, norm, part = [int(key[v]) for v in ['wn', 'op', 'norm', 'part']]
            copies = max(1, (256 * 1024**2 + n * k * 2 - 1) // (n * k * 2))
            iterations = max(a.iterations, copies + 1)
            generator = t.Generator().manual_seed(2017 + ordinal)
            x_cpu = (t.randn(b, k, generator=generator) * .1).bfloat16()
            w_cpu = (t.randn(n, k, generator=generator) * .1).bfloat16()
            x, w = x_cpu.cuda(), w_cpu.cuda()
            packed = _packed_gpu(dict(kind='tile_pages', layout='xcore_mma', tile_n=tn, tile_k=32,
                                     source=dict(kind='alias', source='w')), lambda _: w).contiguous().repeat(copies)
            residual_cpu = (t.randn(b, n, generator=generator) * .1).bfloat16()
            residual = residual_cpu.cuda()
            out = t.empty(b, n // 2 if op == 2 else n, dtype=t.bfloat16, device='cuda')
            tiles = n // tn
            slots = (104 + tiles - 1) // tiles + 1
            partial = t.empty(tiles * slots * b * (tn + 1), device='cuda')
            tickets = t.zeros(tiles, dtype=t.int32, device='cuda')
            counter = t.zeros(8, dtype=t.int64, device='cuda')
            stamps = t.zeros(iterations, dtype=t.int64, device='cuda')
            resources = (C.c_int * 4)()
            ops = Operands(*[v.data_ptr() for v in [x, packed, residual, out, partial, tickets]],
                           None, None, k, n, 1e-6)
            args = Args(ops, counter.data_ptr(), stamps.data_ptr(), resources, n, k, b, iterations, copies)
            for repeat in range(2):
                counter.zero_(); tickets.zero_()
                result = lib.mx5_bench(C.byref(args), wn, op, norm, part, stream)
                if result: raise RuntimeError(f'bench launch failed rc={result}, key={key}')
                t.cuda.synchronize()
            if counter[0].item() != 104 * iterations: raise RuntimeError('barrier count mismatch')
            if tickets.abs().sum().item(): raise RuntimeError('unreset LA tickets')
            reference = cpu_fp32_reference(x_cpu, w_cpu, residual_cpu, op, norm, t).to(out.device)
            if not t.all((out.float() - reference.float()).abs() <= .02 + .016 * reference.float().abs()).item():
                raise RuntimeError('repeated-stage output mismatch')
            ticks = stamps.cpu().tolist(); durations = sorted(y - x for x, y in zip(ticks, ticks[1:]))
            if not durations or durations[0] <= 0:
                raise RuntimeError('GEMM stage timestamps did not advance')
            row = dict(key=key, round=a.round, executor='mx_static_stage_bench', implementation='mx5_gemm_v2',
                       median_ns=statistics.median(durations), p90_ns=durations[len(durations) * 9 // 10],
                       iterations=iterations, copies=copies, working_set_bytes=packed.numel() * 2,
                       mtreg=resources[0], local_bytes=resources[1], static_smem_bytes=resources[2],
                       resident_ctas=resources[3], threads=meta['threads'], source_sha256=meta['source_sha256'],
                       binary_sha256=binary, base_head=meta['base_head'],
                       prefetch=compiled['pw'], reference_backend=REFERENCE_BACKEND,
                       reference_precision=REFERENCE_PRECISION, reference_oracle_version=REFERENCE_ORACLE_VERSION,
                       reference_oracle_description=REFERENCE_ORACLE_DESCRIPTION,
                       reference_torch_version=t.__version__, reference_num_threads=t.get_num_threads())
            output.write(json.dumps(row, sort_keys=True) + '\n'); output.flush()
            print(f'round={a.round} key={ordinal} median_ns={row["median_ns"]}', flush=True)


if __name__ == '__main__': main()
