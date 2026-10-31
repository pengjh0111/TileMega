#!/usr/bin/env python3
"""Per-stage SASS size (prompt §4 A0.1 item 7).

A stage body that runs once per layer sits on the critical path cold: if the
expert weight stream has evicted its code, every instruction fetch miss goes to
DRAM. This measures how much code each stage type has, so the D0 marks can be
read against it.
"""
import argparse
import csv
import json
import re
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

# A demangled device function is attributed to the stage whose body it belongs
# to; order matters, the first match wins.
STAGES = (('Sm90DenseMoeFusedGemm', 'expert_fused'),
          ('Sm90DenseRouterFinish', 'router_finish'),
          ('Sm90DenseMoeGemm', 'moe'),
          ('Sm90DenseAttentionBody', 'attention'),
          ('Sm90DenseHeadFinish', 'head_finish'),
          ('Sm90DenseGemm', 'gemm'),
          ('Sm90DenseEmbedding', 'embedding'),
          ('sm90_dense_native_kernel', 'kernel'))


def functions(binary):
    """(name, bytes) per device function from cuobjdump's SASS listing."""
    raw = subprocess.check_output(['cuobjdump', '-sass', str(binary)], text=True,
                                  errors='replace')
    result, name, count = [], None, 0
    for line in raw.splitlines():
        header = re.match(r'\s*Function : (.+)', line)
        if header:
            if name is not None:
                result.append((name, count * 16))
            name, count = header.group(1).strip(), 0
            continue
        # One SASS instruction per numbered line; each is 16 bytes on sm_90.
        if name is not None and re.match(r'\s*/\*[0-9a-f]{4}\*/', line):
            count += 1
    if name is not None:
        result.append((name, count * 16))
    return result


def stage_of(name):
    for marker, stage in STAGES:
        if marker in name:
            return stage
    return 'other'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, action='append', required=True)
    p.add_argument('--label', action='append', required=True,
                   help='one label per --binary, e.g. the model and batch')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if len(a.binary) != len(a.label):
        p.error('one --label per --binary')
    rows = []
    for binary, label in zip(a.binary, a.label):
        totals = {}
        for name, size in functions(binary):
            stage = stage_of(name)
            entry = totals.setdefault(stage, {'sass_bytes': 0, 'functions': 0, 'largest': 0})
            entry['sass_bytes'] += size
            entry['functions'] += 1
            entry['largest'] = max(entry['largest'], size)
        for stage, entry in sorted(totals.items(), key=lambda kv: -kv[1]['sass_bytes']):
            rows.append(dict(label=label, stage=stage, **entry))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['label', 'stage', 'sass_bytes', 'functions',
                                                    'largest'])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'rows': len(rows), 'out': str(a.out),
                      'total_bytes': {r['label']: sum(x['sass_bytes'] for x in rows
                                                      if x['label'] == r['label'])
                                      for r in rows}}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
