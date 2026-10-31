"""FP64 attention, original V2 transforms, cache history and exact repeats."""
import ctypes as C
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pytest

from test_mx_decode_attention_v2 import Ops, Args, layout_indices

pytestmark = pytest.mark.skipif(
    os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1' or not os.environ.get('MX6_ATTN_SO'),
    reason='requires guarded MX6_ATTN_SO and its profile')


def test_attention_matrix():
    import torch as t
    t.set_num_threads(1)
    profile = json.loads(os.environ['MX6_ATTN_PROFILE'])
    dim, query, row_heads = (profile[k] for k in ('dim', 'query', 'row_heads'))
    library = Path(os.environ['MX6_ATTN_SO'])
    lib = C.CDLL(str(library))
    launch = lib.mx6_attention
    launch.argtypes = [C.POINTER(Args)] + [C.c_int] * 3 + [C.c_void_p, C.POINTER(C.c_int), C.c_int]
    launch.restype = C.c_int
    transform = lib.mx6_attention_reference_transform
    transform.argtypes = [C.POINTER(Ops), C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_void_p]
    transform.restype = C.c_int
    stream = t.cuda.current_stream().cuda_stream
    cap = 1088
    ki, vi = [t.from_numpy(a.reshape(-1)).long().cuda() for a in layout_indices(dim, cap)]
    angle = t.arange(cap).float()[:, None] * t.linspace(.002, .02, dim // 2)[None, :]
    angle = t.cat((angle, angle), dim=1)
    cosine, sine = angle.cos().bfloat16().cuda(), angle.sin().bfloat16().cuda()
    rows = []
    executed = rejected = 0

    def record(value):
        rows.append(value)
        if os.environ.get('MX6_ATTN_OUT'):
            with Path(os.environ['MX6_ATTN_OUT']).open('a') as output:
                output.write(json.dumps(value) + '\n')

    for norm, batch, hkv, length in itertools.product(
            (False, True) if dim == 128 else (False,), (1, 16),
            (row_heads,) if row_heads else (4, 16), (1, 31, 32, 33, 65, 257, 1088)):
        heads = batch * hkv
        generator = t.Generator().manual_seed(63000 + dim + query + batch + hkv + length)
        raw = (t.randn(heads, query + 2, dim, generator=generator) * .2).bfloat16().cuda()
        following = (t.randn(heads, query + 2, dim, generator=generator) * .2).bfloat16().cuda()
        key = (t.randn(heads, cap, dim, generator=generator) * .2).bfloat16().cuda()
        value = (t.randn(heads, cap, dim, generator=generator) * .2).bfloat16().cuda()
        width = row_heads * dim if row_heads and norm else dim
        weights = (1 + .1 * t.randn(2, width, generator=generator)).bfloat16().cuda()

        def operands(inp, past, pk=None, pv=None, ctx=None, partial=None, tickets=None):
            ptr = lambda tensor: tensor.data_ptr() if tensor is not None else None
            return Ops(inp.data_ptr(), weights[0].data_ptr(), weights[1].data_ptr(),
                       cosine.data_ptr(), sine.data_ptr(), ptr(pk), ptr(pv), ptr(ctx),
                       ptr(partial), ptr(tickets), cap, past, 1e-6)

        def reference(inp, past, history_k, history_v):
            q = t.empty(heads, query, dim, dtype=t.bfloat16, device='cuda')
            k = t.empty(heads, dim, dtype=t.bfloat16, device='cuda')
            p = operands(inp, past)
            assert transform(C.byref(p), q.data_ptr(), k.data_ptr(), heads, int(norm), stream) == 0
            expected_k, expected_v = history_k.clone(), history_v.clone()
            expected_k[:, past] = k
            expected_v[:, past] = inp[:, query + 1]
            scores = t.einsum('hqd,htd->hqt', q.double(), expected_k[:, :past + 1].double()) / dim ** .5
            expected = t.einsum('hqt,htd->hqd', scores.softmax(-1),
                                expected_v[:, :past + 1].double()).bfloat16()
            return expected, expected_k, expected_v

        expected, expected_k, expected_v = reference(raw, length - 1, key, value)
        history = reference(following, length, expected_k, expected_v) if length < cap else None
        for kind, split in itertools.product((0, 1, 2), (1, 2, 4)):
            if split > 1 and heads * split > 104:
                continue
            base = dict(dim=dim,query=query,row_norm_heads=row_heads if norm else 0,
                        qk_norm=norm,batch=batch,hkv=hkv,length=length,prefetch=kind,split=split,
                        PID=os.getpid(),binary_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                        reference='FP64 attention over independently transformed original-V2 BF16 Q/K',
                        contract='bf16',threads=512,ctas=104,timing_eligible=False)
            outputs = []
            for repeat in range(2):
                pk = t.empty(heads, cap * dim, dtype=t.bfloat16, device='cuda')
                pv = t.empty_like(pk)
                pk[:, ki] = key.reshape(heads, -1)
                pv[:, vi] = value.reshape(heads, -1)
                ctx = t.full((heads, query, dim), float('nan'), dtype=t.bfloat16, device='cuda')
                partial = t.empty(heads * split * query * (dim + 2), device='cuda') if split > 1 else None
                tickets = t.zeros(heads, dtype=t.int32, device='cuda') if split > 1 else None
                calls = t.zeros(104, 8, dtype=t.int32, device='cuda')
                p = operands(raw, length - 1, pk, pv, ctx, partial, tickets)
                a = Args(p, calls.data_ptr(), batch, hkv)
                resources = (C.c_int * 4)()
                status = launch(C.byref(a), int(norm), kind, split, stream, resources, 1)
                if status == 8:
                    assert repeat == 0
                    record(dict(base,status='resource_rejected_not_correctness_pass',MTreg=resources[0],
                                local=resources[1],static=resources[2],resident=resources[3]))
                    rejected += 1
                    break
                assert status == 0, dict(base,query_status=status)
                assert resources[1] == resources[2] == 0 and resources[3] == 1
                results = []
                for step, (inp, past, wanted) in enumerate(
                        [(raw, length - 1, (expected, expected_k, expected_v))] +
                        ([(following, length, history)] if history else [])):
                    calls.zero_()
                    p = operands(inp, past, pk, pv, ctx, partial, tickets)
                    a = Args(p, calls.data_ptr(), batch, hkv)
                    assert launch(C.byref(a), int(norm), kind, split, stream, None, 0) == 0
                    t.cuda.synchronize()
                    want, wk, wv = wanted
                    assert bool((calls == 1).all()), base
                    if tickets is not None:
                        assert bool((tickets == 0).all()), base
                    assert t.equal(pk[:, ki].view(t.int16), wk.reshape(heads, -1).view(t.int16)), ('K bits',base,step)
                    assert t.equal(pv[:, vi].view(t.int16), wv.reshape(heads, -1).view(t.int16)), ('V bits',base,step)
                    error = (ctx.float() - want.float()).abs()
                    assert bool((error <= .006 + .016 * want.float().abs()).all()), (base,step,float(error.max()))
                    results.append(ctx.cpu().view(t.int16))
                outputs.append(results)
            if len(outputs) == 2:
                assert all(t.equal(a,b) for a,b in zip(outputs[0],outputs[1])), base
                record(dict(base,status='pass',MTreg=resources[0],local=resources[1],static=resources[2],
                            resident=resources[3],repeated_bitwise=True,cache_history=history is not None,
                            original_transform_cache_bits=True,fp64_context_pass=True))
                executed += 1
    assert executed > 0
    print('MX6_ATTENTION_MATRIX',profile,'executed',executed,'resource_rejected',rejected,flush=True)
