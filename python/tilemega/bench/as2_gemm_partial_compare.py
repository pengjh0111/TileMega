"""CPU diagnostic for observed two/three/seven-part FP32 GEMM reductions.

The device probe writes disjoint FP32 partials. This comparison does not
verify a device combine, CTG decomposition, synchronization, or model execution.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tilemega.bench.as2_models import RUNS, controlled
from tilemega.bench.as2_native import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('case', 'partial', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    for name in ('case', 'partial', 'out'):
        setattr(a, name, controlled(getattr(a, name), [RUNS]))
    if a.out.exists():
        raise FileExistsError(a.out)
    case = json.loads((a.case / 'case.json').read_text())
    m, n, k = case['shape_mnk']
    down_slice=(m,n,k)==(8,4096,14336)
    if (m not in (1,16,64) and not down_slice) or not ((n == 2048 and k in (6144, 8192)) or (n == 3072 and k == 8192) or (n == 1024 and k == 4096) or (m==16 and n==4096 and k==4096) or down_slice):
        raise ValueError('Require the observed, bounded Stream-K diagnostic')
    parts = 7 if n == 1024 else 2 if n == 3072 else 3
    if case.get('partial_count')==7:
        if not (((m,n)==(16,2048) and k in (6144,8192)) or (m,n,k)==(16,4096,4096)) or case.get('source_shape_mnk')!=[1024,n,k]:
            raise ValueError('Seven-part residual probe requires the observed B16 prefill slice')
        parts=7
    if case.get('partial_count')==3 and ((m,n,k)==(16,4096,4096) or down_slice):
        if case.get('source_shape_mnk')!=[1024,n,k]:
            raise ValueError('Three-part Q probe requires the observed B16 prefill slice')
        parts=3
    for name, sha in case['files'].items():
        if digest(a.case / name) != sha:
            raise ValueError('Observed operand/reference identity changed')
    path = a.partial / 'partial_f32.bin'
    if path.stat().st_size != parts * m * n * 4:
        raise ValueError('Partial extent mismatch')
    partial = np.fromfile(path, dtype='<f4').reshape(parts, m, n)
    if not np.isfinite(partial).all():
        raise ValueError('Nonfinite FP32 partial')
    # Installed CANN's Stream-K epilogue adds increasing K slices, then RINT.
    reduced = partial[0].copy()
    for part in partial[1:]:
        reduced = np.add(reduced, part, dtype=np.float32)
    value = torch.from_numpy(reduced).bfloat16()
    tail=case.get('observed_tail_column_begin')
    raw_partial_full_difference=None
    if tail is not None:
        if tail!=3584 or not ((m,n,k)==(16,4096,4096) or down_slice):raise ValueError('Unregistered same-input tail experiment')
        full_reference=torch.from_file(str(a.case/'reference.bin'),size=m*n,dtype=torch.bfloat16).reshape_as(value)
        raw_partial_full_difference=int((value!=full_reference).sum())
        regular=torch.from_file(str(a.case/'model_native.bin'),size=m*n,dtype=torch.bfloat16).reshape_as(value)
        value[:,:tail]=regular[:,:tail]
    checks = []
    for name in ('reference.bin', 'model_native.bin'):
        if name == 'model_native.bin' and name not in case['files']:
            continue
        gold = torch.from_file(str(a.case / name), shared=False,
                               size=m * n, dtype=torch.bfloat16).reshape_as(value)
        delta = value.float() - gold.float()
        checks.append(dict(reference=name, elements=value.numel(),
            bitwise_different=int((value.view(torch.int16) != gold.view(torch.int16)).sum()),
            max_abs=float(delta.abs().max()), rmse=float(delta.square().mean().sqrt())))
    chunk = (k + parts - 1) // parts
    result = dict(scope='same-operand reduction-order diagnostic only',
        case_sha256=digest(a.case / 'case.json'), partial_sha256=digest(path),
        partial_k_ranges=[[i * chunk, min((i + 1) * chunk, k)] for i in range(parts)],
        combine='CPU FP32 increasing K slice sum, BF16 RINT',
        composite_tail_begin=tail,raw_partial_full_width_differences=raw_partial_full_difference,
        observed_reference_matches=checks[0]['bitwise_different'] == 0,
        device_combine_verified=False, tilemega_pipeline=False,
        full_model_admitted=False, checks=checks)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return 0 if result['observed_reference_matches'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
