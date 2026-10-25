#!/usr/bin/env python3
"""Rewrite older native dense coordinates as schema 3 (prompt §5 D4, §6 D5/D6).

Schema 2 dropped `dn_vector_sums`: InputNorm statistics are accumulated by the
GEMM consumers. Schema 3 drops the per-stage arena offset and preissue fields,
which D5/D6 derive by rule, and adds the DT-1 `weight_stream` switch. Every
other coordinate is copied unchanged, so a pairing keeps the same geometry.
"""
import argparse
import json
from pathlib import Path

DERIVED = ('workspace_offset', 'preissue_smem_bytes', 'prefetch_l2_bytes', 'preissue_late')


def upgrade(value, weight_stream):
    schema = value.get('schema')
    if schema == 3:
        return dict(value, weight_stream=weight_stream)
    if schema not in (1, 2):
        raise ValueError('unknown native coordinate schema')
    result = {k: v for k, v in value.items() if k != 'dn_vector_sums'}
    stages = [*result['gemms'].values(), result['attention']]
    for stage in stages:
        if any(stage.get(k) for k in DERIVED):
            raise ValueError('schema 3 derives arena offsets and preissue; refusing nonzero values')
        for k in DERIVED:
            stage.pop(k, None)
    for g in result['gemms'].values():
        g.setdefault('register_stages', 1)
    result['schema'] = 3
    result['weight_stream'] = weight_stream
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--weight-stream', action='store_true', help='enable the D6 stream (DT-1 arm)')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    value = upgrade(json.loads(a.input.read_text()), a.weight_stream)
    a.output.write_text(json.dumps(value, indent=2) + '\n')


if __name__ == '__main__':
    main()
