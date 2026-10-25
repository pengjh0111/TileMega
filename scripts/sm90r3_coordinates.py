#!/usr/bin/env python3
"""Rewrite schema-1 native dense coordinates as schema 2 (prompt §5 D4).

Schema 2 drops `dn_vector_sums`: InputNorm statistics are accumulated by the
GEMM consumers, so the deferred-norm sum layout no longer exists. Every other
coordinate is copied unchanged, so a pairing keeps the same geometry.
"""
import argparse
import json
from pathlib import Path


def upgrade(value):
    if value.get('schema') == 2:
        return value
    if value.get('schema') != 1:
        raise ValueError('unknown native coordinate schema')
    result = {k: v for k, v in value.items() if k != 'dn_vector_sums'}
    result['schema'] = 2
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.write_text(json.dumps(upgrade(json.loads(a.input.read_text())), indent=2) + '\n')


if __name__ == '__main__':
    main()
