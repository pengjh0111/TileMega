#!/usr/bin/env python3
"""Read append-only resource CSVs, including immutable hashed CSV parts."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

PART_TYPE = 'resource_csv_part'


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def schema_sha256(fields: list[str]) -> str:
    return sha256(json.dumps(fields, separators=(',', ':')).encode())


def part_rows(main: Path, index: dict, fields: list[str]):
    metadata = json.loads(index['resource_scope'])
    if metadata['kind'] != 'mx6_resource_csv_part' or metadata['schema_version'] != 1:
        raise ValueError('unsupported resource CSV part index')
    name = metadata['path']
    if Path(name).name != name or not name.endswith('.csv.gz'):
        raise ValueError('resource CSV part must be a sibling .csv.gz file')
    if metadata['fieldnames_sha256'] != schema_sha256(fields):
        raise ValueError('resource CSV part schema differs from its main file')
    compressed = (main.parent / name).read_bytes()
    if sha256(compressed) != metadata['sha256']:
        raise ValueError('resource CSV part compressed SHA256 differs')
    raw = gzip.decompress(compressed)
    if sha256(raw) != metadata['uncompressed_sha256']:
        raise ValueError('resource CSV part uncompressed SHA256 differs')
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8'), newline=''))
    if reader.fieldnames != fields:
        raise ValueError('resource CSV part header differs')
    count = 0
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError('resource CSV part row width differs')
        if row['record_type'] == PART_TYPE:
            raise ValueError('nested resource CSV parts are forbidden')
        if row['status'] != 'CPU_resource_rejected_never_launched' or row['actual_query'] != 'False':
            raise ValueError('only unlaunched CPU resource rejections may be partitioned')
        count += 1
        yield row
    if count != metadata['rows']:
        raise ValueError('resource CSV part row count differs')


def resource_rows(main: Path):
    """Expand only indexed parts; historical short-schema main rows stay intact."""
    main = Path(main)
    with main.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if fields is None:
            raise ValueError('resource CSV header is absent')
        for row in reader:
            if row.get('record_type') == PART_TYPE:
                yield from part_rows(main, row, fields)
            else:
                yield row


def seal_rejections(main: Path, fields: list[str], rows: list[dict], checkpoint: str) -> dict:
    """Create and verify a new part before its index is appended to the main CSV."""
    main = Path(main)
    if not rows:
        raise ValueError('empty resource CSV part')
    if any(row['status'] != 'CPU_resource_rejected_never_launched' or
           str(row['actual_query']) != 'False' for row in rows):
        raise ValueError('queried/measured resource rows must stay in the main CSV')
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    raw = buffer.getvalue().encode('utf-8')
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    sequence = 1
    while (main.parent / f'{main.stem}_cpu_part{sequence:04d}.csv.gz').exists():
        sequence += 1
    part = main.parent / f'{main.stem}_cpu_part{sequence:04d}.csv.gz'
    with part.open('xb') as stream:
        stream.write(compressed)
    metadata = dict(kind='mx6_resource_csv_part', schema_version=1, path=part.name,
                    sha256=sha256(compressed), uncompressed_sha256=sha256(raw), rows=len(rows),
                    fieldnames_sha256=schema_sha256(fields),
                    scope='complete CPU rejection rows; never queried or executed; immutable part')
    index = dict(checkpoint=checkpoint, record_type=PART_TYPE, program_version=3,
                 purpose='resource_probe', actual_query=False, evidence='verified',
                 status='sealed_CPU_rejection_CSV_part',
                 resource_scope=json.dumps(metadata, sort_keys=True, separators=(',', ':')))
    normalized = [{key: str(value) if value is not None else '' for key, value in row.items()}
                  for row in rows]
    normalized = [{key: row.get(key, '') for key in fields} for row in normalized]
    if list(part_rows(main, index, fields)) != normalized:
        raise ValueError('sealed resource CSV part failed exact field round trip')
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--out', type=Path, help='write an expanded derived CSV, normally under runs/')
    args = parser.parse_args()
    count = parts = 0
    with args.csv.open(newline='') as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        parts = sum(row.get('record_type') == PART_TYPE for row in reader)
    output = args.out.open('x', newline='') if args.out else None
    try:
        writer = csv.DictWriter(output, fields, lineterminator='\n') if output else None
        if writer:
            writer.writeheader()
        for row in resource_rows(args.csv):
            if writer:
                writer.writerow(row)
            count += 1
    finally:
        if output:
            output.close()
    print(json.dumps(dict(rows=count, verified_parts=parts, expanded_output=str(args.out) if args.out else None)))


if __name__ == '__main__':
    main()
