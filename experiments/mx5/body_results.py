#!/usr/bin/env python3
"""Append compiler resource evidence without rewriting earlier records."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


FIELDS = ['record_type', 'campaign', 'round', 'executor', 'implementation', 'body',
          'threads', 'coordinates', 'kernel', 'mtreg', 'stack_bytes', 'static_smem_bytes',
          'compile_seconds', 'base_head', 'source_sha256', 'binary_sha256']
RESOURCE = re.compile(r'Function properties for\s+(\S+)\s*\n\s*(\d+) bytes stack frame\s*\n'
                      r'\s*maca info\s*:\s*Used\s+(\d+) MTregisters,\s*\d+ STregisters,\s*'
                      r'(\d+) bytes shared mem')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, action='append', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    existing = []
    if args.out.exists():
        with args.out.open(newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != FIELDS: raise RuntimeError('resource schema changed')
            existing = list(reader)
    keys = {(r['campaign'], r['kernel']) for r in existing}
    additions = []
    for directory in args.campaign:
        records = sorted(directory.glob('*.json'))
        if not records: raise RuntimeError('no compilation records: ' + str(directory))
        for record in records:
            item = json.loads(record.read_text())
            if 'binary_sha256' not in item or item['returncode'] != 0:
                raise RuntimeError('unsuccessful compilation: ' + str(record))
            binary = record.with_suffix('.so')
            if hashlib.sha256(binary.read_bytes()).hexdigest() != item['binary_sha256']:
                raise RuntimeError('binary identity changed: ' + str(binary))
            log = record.with_suffix('.log').read_text()
            # cucc merges compiler stdout/stderr without line serialization;
            # its warning summary sometimes splits a mangled kernel name.
            counts = set(re.findall(r'^(\d+) warnings? generated when compiling for host\.$', log, re.M))
            for count in counts:
                log = re.sub(re.escape(count) + r' warnings? generated when compiling for [^.]+\.\n', '', log)
            resources = RESOURCE.findall(log)
            if not resources: raise RuntimeError('no resource diagnostics: ' + str(record))
            family = item.get('body', 'gemm')
            coordinates = json.dumps({k: item[k] for k in ['tn', 'depth', 'dim', 'query', 'pw']
                                      if item.get(k) is not None},
                                     sort_keys=True, separators=(',', ':'))
            for kernel, stack, registers, smem in resources:
                row = dict(record_type='stage_bench_build' if item.get('bench') else 'unit_kernel_build',
                           campaign=str(directory), round='build1',
                           executor='mx_static_stage_bench' if item.get('bench') else 'unit_cuda',
                           implementation='mx5_' + family + '_v2', body=family,
                           threads=str(item.get('threads', 576)), coordinates=coordinates, kernel=kernel,
                           mtreg=registers, stack_bytes=stack, static_smem_bytes=smem,
                           compile_seconds=f'{item["seconds"]:.6f}', base_head=item['base_head'],
                           source_sha256=item['source_sha256'], binary_sha256=item['binary_sha256'])
                key = (str(directory), kernel)
                old = next((r for r in existing if (r['campaign'], r['kernel']) == key), None)
                if old:
                    if old != row: raise RuntimeError('historical resource row differs: ' + str(key))
                elif key not in keys:
                    additions.append(row); keys.add(key)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator='\n')
        if not existing: writer.writeheader()
        writer.writerows(additions)
    print(json.dumps(dict(existing=len(existing), appended=len(additions))))


if __name__ == '__main__': main()
