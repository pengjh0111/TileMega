#!/usr/bin/env python3
"""Reject calibration-only local temporaries absent from native Consume."""
import argparse
from pathlib import Path
import re
import subprocess
import sys

from sm90d1_common import ROOT, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--ptxas-log', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(ROOT / 'python'))
    from tilemega.build.identity import parse_resources
    resources = parse_resources(a.ptxas_log.read_text())
    sass = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '--dump-sass',
                                    str(a.binary.resolve())], text=True)
    functions = {x.splitlines()[0].strip(): x for x in sass.split('Function : ')[1:]}
    rows = []
    for name, facts in resources.items():
        match = re.match(r'_Z8MainloopILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)EE', name)
        if not match:
            continue
        if name not in functions:
            raise ValueError('mainloop kernel is absent from the audited binary')
        tn, tk, wn, depth = map(int, match.groups())
        local = len(re.findall(r'\b(?:LDL|STL)\b', functions[name]))
        compatible = not (local or facts.get('spill') or facts.get('stack_bytes', 0))
        rows.append(dict(tile_n=tn, tile_k=tk, warps_n=wn, warps_k=4 // wn,
            register_stages=depth, registers=facts['registers'], local_instructions=local,
            stack_bytes=facts.get('stack_bytes', 0),
            spill_store_bytes=facts.get('spill_store_bytes', 0),
            spill_load_bytes=facts.get('spill_load_bytes', 0),
            production_compatible=compatible))
    if not rows:
        raise ValueError('no mainloop resource entries found')
    write(a.out, dict(binary_sha256=sha(a.binary), ptxas_sha256=sha(a.ptxas_log),
        kernels=rows, method='native no-local Consume surrogate; reject extra stack/local/spills',
        compatible_cases=','.join(f"{r['tile_n']}x{r['tile_k']}x{r['warps_n']}x{r['register_stages']}"
                                  for r in rows if r['production_compatible']),
        compatible=sum(r['production_compatible'] for r in rows), total=len(rows)))
    print(f"{sum(r['production_compatible'] for r in rows)}/{len(rows)} compatible mainloops")


if __name__ == '__main__':
    main()
