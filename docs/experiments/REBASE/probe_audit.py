#!/usr/bin/env python3
"""Audit signed intervention pairs and the unchanged control kernels.

The audit does not filter samples or normalize timing bands. --dump-sass
regenerates disassembly from this machine's binaries; without it, the archived
disassembly and raw process logs suffice. Run again after collection finishes.
"""
import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'JOINT2'))
import analyze as joint


def functions(raw):
    headers = list(re.finditer(r'^\s*Function : (.+)$', raw, re.M))
    result = {}
    for i, header in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(raw)
        block = raw[header.start():end]
        last = block.rfind('}')
        result[header.group(1)] = block[:last + 1] if last >= 0 else block
    return result


def write_table(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t',
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dump-sass', action='store_true')
    args = parser.parse_args()
    root = HERE / 'raw/real_s128'
    out = HERE / 'probe_audit'
    out.mkdir(exist_ok=True)
    disassembly = {}
    for arm in ('full', 'nofence', 'l1nosync'):
        path = out / (arm + '.sass')
        if args.dump_sass:
            binary = root / 'bin' / ('eft__' + arm)
            command = ['/usr/local/cuda/bin/cuobjdump', '--dump-sass', str(binary)]
            with path.open('w') as stream:
                subprocess.run(command, stdout=stream, check=True)
            (out / (arm + '.command.json')).write_text(json.dumps(dict(
                command=command, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()
            ), indent=2) + '\n')
        disassembly[arm] = functions(path.read_text())
    comparisons = []
    for arm in ('nofence', 'l1nosync'):
        for name, block in disassembly[arm].items():
            comparisons.append(dict(arm=arm, function=name, bytes=len(block),
                                    identical_full=int(disassembly['full'].get(name) == block)))
    write_table(out / 'kernel_identity.tsv', comparisons)
    pairs = []
    controls = []
    specs = json.loads((root / 'specs.json').read_text())
    for config in dict.fromkeys(k.split('__')[0] for k in specs):
        for i in range(25):
            times = {}
            metadata = {}
            for arm in ('full', 'nofence', 'nowait', 'neither', 'l1nosync'):
                log = root / 'measure' / (config + '__' + arm) / f'r{i}.log'
                if not log.with_suffix('.json').exists():
                    continue
                times[arm] = joint.timing(log)
                metadata[arm] = json.loads(log.with_suffix('.json').read_text())
                controls.append(dict(config=config, arm=arm, round=i,
                                     started_ns=metadata[arm]['started_ns'],
                                     l05_ms=times[arm]['l05_ms'], l1_ms=times[arm]['l1_ms'],
                                     l2_ms=times[arm]['l2_ms'], log=str(log)))
            if 'full' not in times or 'l1nosync' not in times:
                continue
            difference = times['full']['l1_ms'] - times['l1nosync']['l1_ms']
            pairs.append(dict(config=config, round=i, barrier_ms=difference,
                              nonpositive=int(difference <= 0), zero=int(difference == 0),
                              full_l1_ms=times['full']['l1_ms'],
                              l1nosync_l1_ms=times['l1nosync']['l1_ms'],
                              start_separation_s=abs(metadata['full']['started_ns'] -
                                                     metadata['l1nosync']['started_ns']) / 1e9))
    write_table(out / 'barrier_pairs.tsv', pairs)
    write_table(out / 'unchanged_controls.tsv', controls)
    print(f'PROBE_AUDIT processes={len(controls)}/1250 pairs={len(pairs)}/250 '
          f'nonpositive={sum(r["nonpositive"] for r in pairs)} '
          f'zero={sum(r["zero"] for r in pairs)}')
    for row in comparisons:
        if 'tilemega_l1_kernel' in row['function'] or 'tilemega_l2_kernel' in row['function']:
            print('KERNEL_IDENTITY', row)


if __name__ == '__main__':
    main()
