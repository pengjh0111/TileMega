"""CPU diagnostic for the observed three-part FP32 GEMM reduction.

The device probe writes three disjoint FP32 partials. This comparison does not
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
    if m not in (1, 64) or n != 2048 or k not in (6144, 8192):
        raise ValueError('Require the observed, bounded Stream-K diagnostic')
    for name, sha in case['files'].items():
        if digest(a.case / name) != sha:
            raise ValueError('Observed operand/reference identity changed')
    path = a.partial / 'partial_f32.bin'
    if path.stat().st_size != 3 * m * n * 4:
        raise ValueError('Partial extent mismatch')
    partial = np.fromfile(path, dtype='<f4').reshape(3, m, n)
    if not np.isfinite(partial).all():
        raise ValueError('Nonfinite FP32 partial')
    # Installed CANN's Stream-K epilogue adds increasing K slices, then RINT.
    reduced = np.add(np.add(partial[0], partial[1], dtype=np.float32),
                     partial[2], dtype=np.float32)
    value = torch.from_numpy(reduced).bfloat16()
    checks = []
    for name in ('reference.bin', 'model_native.bin'):
        gold = torch.from_file(str(a.case / name), shared=False,
                               size=m * n, dtype=torch.bfloat16).reshape_as(value)
        delta = value.float() - gold.float()
        checks.append(dict(reference=name, elements=value.numel(),
            bitwise_different=int((value.view(torch.int16) != gold.view(torch.int16)).sum()),
            max_abs=float(delta.abs().max()), rmse=float(delta.square().mean().sqrt())))
    chunk = (k + 2) // 3
    result = dict(scope='same-operand reduction-order diagnostic only',
        case_sha256=digest(a.case / 'case.json'), partial_sha256=digest(path),
        partial_k_ranges=[[i * chunk, min((i + 1) * chunk, k)] for i in range(3)],
        combine='CPU FP32 (partial0 + partial1) + partial2, BF16 RINT',
        device_combine_verified=False, tilemega_pipeline=False,
        full_model_admitted=False, checks=checks)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
