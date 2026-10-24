#!/usr/bin/env python3
"""Prepare one fixed exact-integer Qwen B16 QKV input/reference set on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def bf16(value):
    bits = np.asarray(value, dtype='<f4').view('<u4')
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype('<u2')


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--out', type=Path, required=True)
    args = p.parse_args();assert not args.out.exists(), 'preserve prepared inputs'
    args.out.mkdir(parents=True)
    k = np.arange(1024, dtype=np.uint32)[None, :]
    row = np.arange(16, dtype=np.uint32)[:, None]
    scales = (1 << (row % 3)).astype(np.int32)
    quota_table = np.array([0, 1, 2, 3, 4, 3, 2, 1], dtype=np.uint32)
    quota = quota_table[((k // 8) + row) % 8]
    selected = ((k % 8) ^ (row % 8)) < quota
    mix = (k + 17) * np.uint32(0x85ebca6b) ^ (row + 1) * np.uint32(0x9e3779b9)
    mix ^= mix >> 16
    activation = np.where(selected, np.where(mix & 1, -2, 2) * scales, 0).astype(np.int32)
    n = np.arange(4096, dtype=np.uint32)[:, None]
    mix = (n + 1) * np.uint32(0x9e3779b9) ^ (k + 11) * np.uint32(0x85ebca6b)
    mix ^= mix >> 16;mix *= np.uint32(0x7feb352d);mix ^= mix >> 15
    weight = ((mix % 3).astype(np.int32) - 1)
    norm = (activation.reshape(16, 128, 8).astype(np.int64) ** 2).sum(axis=2).astype('<f4')
    squares = norm.sum(axis=1, dtype=np.float32)
    assert np.array_equal(squares, (1024 * scales[:, 0] ** 2).astype(np.float32))
    dot = activation @ weight.T
    expected = (dot.astype(np.float32) / scales.astype(np.float32)).astype('<f4')
    reference = bf16(expected)
    # All products/partial sums are small exact integers; normalization is 1,1/2,1/4.
    assert np.max(np.abs(dot)) < 2**24 and np.max(np.abs(expected)) <= 512
    assert np.array_equal((reference.astype(np.uint32) << 16).view(np.float32), expected)
    assert set(np.unique(weight)) == {-1, 0, 1}
    assert set(np.unique(activation)) == {-8, -4, -2, 0, 2, 4, 8}
    assert np.all(np.isfinite(expected)) and np.unique(expected).size > 64
    # A wrong K correspondence and wrong N correspondence must be observable.
    assert np.count_nonzero(activation @ np.roll(weight, 8, axis=1).T != dot) > 60000
    assert np.count_nonzero(np.roll(dot, 1, axis=1) != dot) > 60000
    arrays = {'activation.bf16': bf16(activation), 'weight-rowmajor.bf16': bf16(weight),
              'norm8.f32': norm, 'expected.f32': expected, 'expected.bf16': reference}
    for name, array in arrays.items():array.tofile(args.out / name)
    metadata = dict(schema=1, geometry=dict(batch=16,n=4096,k=1024,tn=32,tk=512,slots=2,warps_n=1,warps_k=4),
        epsilon=0, activation_values=[-8,-4,-2,0,2,4,8], weight_values=[-1,0,1],
        row_scales=scales[:,0].tolist(), norm_layout='one FP32 sum of squares per eight activation elements;16x128',
        packing='input weight row-major[N,K]; CUDA host packs each task/tile through exact G::LayoutB with bijection check',
        reference='exact int32 dense dot, exact power-of-two row normalization, BF16 round-to-nearest-even',
        max_abs_dot=int(np.max(np.abs(dot))), max_abs_expected=float(np.max(np.abs(expected))),
        source_sha256=sha(__file__), numpy_version=np.__version__,
        files={name:dict(bytes=(args.out/name).stat().st_size,sha256=sha(args.out/name)) for name in arrays})
    (args.out/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(dict(out=str(args.out), files=len(arrays), max_abs_dot=metadata['max_abs_dot'], max_abs_expected=metadata['max_abs_expected'])))


if __name__=='__main__':main()
