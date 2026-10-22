#!/usr/bin/env python3
"""Identify compiler-equivalent register-lookahead variants before timing."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from check_sm90r2_rebuild import encodings
from sm90d1_common import ROOT, write

sys.path.insert(0, str(ROOT / 'python'))
from tilemega.build.identity import verify


def program_key(binary):
    binary = Path(binary).resolve()
    identity = verify(binary)
    plan = json.loads(Path(str(binary) + '.plan.json').read_text())
    if identity['trace'] or 'native_dense' not in plan:
        raise ValueError('native measurement keys require an uninstrumented native plan')
    assembly = subprocess.check_output([os.getenv('CUOBJDUMP', '/usr/local/cuda/bin/cuobjdump'),
        '--dump-sass', str(binary)], text=True, timeout=120)
    instructions = hashlib.sha256(json.dumps(list(encodings(assembly).values()),
        sort_keys=True).encode()).hexdigest()
    coordinates = copy.deepcopy(plan['native_dense'])
    # Register lookahead changes neither packing nor the launch. Drop it only
    # after binding the complete device instruction encodings above.
    for gemm in coordinates['gemms'].values():
        gemm.pop('register_stages', None)
    signature = dict(instructions=instructions, coordinates=coordinates,
        grid=plan['grid'], shared_bytes=identity['shared_memory_bytes'],
        batch=plan['runtime_batch'], capacity=plan['capacity'],
        export_sha256=plan['export_sha256'])
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    return dict(program_key=digest, signature=signature,
        binary_sha256=identity['binary_sha256'],
        evidence='verified instructions, launch resources and non-lookahead coordinates')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('retain previous program-key evidence')
    result = program_key(a.binary)
    write(a.out, result)
    print(result['program_key'])


if __name__ == '__main__':
    main()
