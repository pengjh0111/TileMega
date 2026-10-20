#!/usr/bin/env python3
"""Query actual device occupancy for already-built diagnostic libraries."""
import argparse
import ctypes as C
from pathlib import Path

from sm90d1_common import RUN, read, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    from tilemega.serving.baseline_common import require_guard
    from tilemega.serving.plan import PlanLibrary
    import torch
    require_guard()
    torch.cuda.init()
    rows = []
    for path in sorted((RUN / 'build').glob('*/*/*.so.identity.json')):
        identity = read(path)
        heads = {read(RUN / 'compiler-v2/receipt.json')['head']}
        if (RUN / 'compiler-p3/receipt.json').exists():
            heads.add(read(RUN / 'compiler-p3/receipt.json')['head'])
        if (RUN / 'compiler-pg-profile/receipt.json').exists():
            heads.add(read(RUN / 'compiler-pg-profile/receipt.json')['head'])
        if (RUN / 'compiler-ready/receipt.json').exists():
            heads.add(read(RUN / 'compiler-ready/receipt.json')['head'])
        if identity['source']['head'] not in heads:
            continue
        binary = Path(str(path).removesuffix('.identity.json'))
        try:
            library = PlanLibrary(binary)
            grid, residency, error = library.info.grid, library.info.residency, None
        except (RuntimeError, ValueError) as exception:
            grid, residency, error = None, None, str(exception)
        rows.append(dict(binary=str(binary), binary_sha256=sha(binary),
            trace=identity['trace'], grid=grid, query_error=error,
            runtime_queried_residency=residency,
            shared_memory_bytes=identity['shared_memory_bytes'], resources=identity['resources'],
            note='runtime occupancy query; measured CTA-to-SM mapping additionally available in stage traces'))
    write(a.out, rows)


if __name__ == '__main__':
    main()
