#!/usr/bin/env python3
"""Verify the selected native program survives rebuilding in another directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from tilemega.build.identity import sha, verify
from tilemega.bench.compare_artifacts import parse_sass


def encodings(text):
    result = {}
    for section in text.split('Function : ')[1:]:
        name = section.splitlines()[0].strip()
        if name in result:
            raise ValueError('ambiguous device architecture')
        words = re.findall(r'/\* (0x[0-9a-f]+) \*/', section)
        if not words:
            raise ValueError('missing device instruction encodings')
        result[name] = hashlib.sha256(''.join(words).encode()).hexdigest()
    if not result:
        raise ValueError('no device functions')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('retain previous rebuild evidence; choose a new output directory')
    a.out.mkdir(parents=True)
    identities, plans, sources, assembly = [], [], [], []
    for label, binary in [('reference', a.reference), ('candidate', a.candidate)]:
        identity = verify(binary)
        if identity['trace']:
            raise ValueError('instrumented libraries cannot verify a selected artifact')
        identities.append(identity)
        plans.append(json.loads(Path(str(binary) + '.plan.json').read_text()))
        sources.append(sha(str(binary) + '.cu'))
        text = subprocess.check_output([os.getenv('CUOBJDUMP', '/usr/local/cuda/bin/cuobjdump'),
                                        '--dump-sass', str(binary.resolve())], text=True, timeout=120)
        (a.out / (label + '.sass')).write_text(text)
        assembly.append(text)
    old, new = [parse_sass(text) for text in assembly]
    old_encoding, new_encoding = [encodings(text) for text in assembly]
    record = dict(schema=1, reference_sha256=identities[0]['binary_sha256'],
        candidate_sha256=identities[1]['binary_sha256'],
        generated_source_sha256=sources,
        same_coordinates=plans[0]['native_dense'] == plans[1]['native_dense'],
        same_workload=all(plans[0][k] == plans[1][k]
                          for k in ('export_sha256', 'runtime_batch', 'capacity')),
        same_source=sources[0] == sources[1], same_instructions=old == new,
        same_instruction_encoding=old_encoding == new_encoding,
        changed_functions=sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k)),
        instruction_encoding_sha256=old_encoding,
        evidence='verified static rebuild comparison; not a performance result')
    record['pass'] = all(record[k] for k in ('same_coordinates', 'same_workload',
        'same_source', 'same_instructions', 'same_instruction_encoding'))
    (a.out / 'comparison.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items()
                      if k not in ('changed_functions', 'instruction_encoding_sha256')}))
    return 0 if record['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
