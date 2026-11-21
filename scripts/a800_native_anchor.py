#!/usr/bin/env python3
"""Recover a native incumbent and its exact inputs from a formal A800 receipt."""
import argparse
import csv
import json
from pathlib import Path

from a800_baselines import RUN
from a800_native import artifact, sha
from tilemega.bench.scheduler import write_json


def incumbent(model, batch, table):
    with table.open() as stream:
        rows = [row for row in csv.DictReader(stream)
                if row['model'] == model and int(row['batch']) == batch]
    row, = rows
    if row['status'] != 'measured':
        raise ValueError('the cell has no accepted native timing')
    identities = ('target_sha256', 'decode_sha256', 'prefill_sha256')
    receipts = sorted(path for directory in (RUN/'native').glob(f'*/{model}/B{batch}/*')
                      for stage in ('measure','candidate-timing')
                      for path in directory.glob(f'{stage}-*.json'))
    for path in receipts:
        receipt = json.loads(path.read_text())
        if not receipt['pass'] or any(receipt[k] != row[k] for k in identities):
            continue
        timing = Path(receipt['evidence'])/f'B{batch}/measurements.json'
        if sha(timing) not in row['receipt_sha256'].split():
            continue
        result = json.loads(timing.read_text())
        compiled = result['configuration']['identities']
        decode, decode_identity, plan = artifact(Path(compiled['decode']['binary_path']))
        if decode_identity['binary_sha256'] != row['decode_sha256']:
            raise ValueError('accepted native binary changed')
        prefill = 'prompt'
        if row['prefill_sha256'] != prefill:
            library, prefill_identity, _ = artifact(Path(compiled['prefill']['binary_path']))
            if prefill_identity['binary_sha256'] != row['prefill_sha256']:
                raise ValueError('accepted prefill binary changed')
            prefill = str(library)
        return dict(model=model,batch=batch,prefill=prefill,decode=str(decode),job=receipt['job'],
                    correctness_scope=str(path.parent),timing=str(timing),timing_sha256=sha(timing),
                    coordinates=plan['native_dense'],**{key:row[key] for key in identities})
    raise ValueError(f'no available artifact with the formal receipt for {model} B{batch}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--table', type=Path, default=RUN/'collected/A800_llm.csv')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('choose a new incumbent receipt path')
    write_json(args.out,incumbent(args.model,args.batch,args.table))
