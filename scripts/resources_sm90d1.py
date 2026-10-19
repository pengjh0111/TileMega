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
        if identity['source']['head'] != read(RUN / 'compiler-v2/receipt.json')['head']:
            continue
        binary = Path(str(path).removesuffix('.identity.json'))
        library = PlanLibrary(binary)
        rows.append(dict(binary=str(binary), binary_sha256=sha(binary),
            trace=identity['trace'], grid=library.info.grid,
            runtime_queried_residency=library.info.residency,
            shared_memory_bytes=identity['shared_memory_bytes'], resources=identity['resources'],
            note='runtime occupancy query; measured CTA-to-SM mapping additionally available in stage traces'))
    write(a.out, rows)


if __name__ == '__main__':
    main()
