#!/usr/bin/env python3
"""Audit the complete stage-price gate against the archived ORACLE universe."""
import csv
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent / 'gemm_price_gate')
parser.add_argument('--stage-entry', action='store_true')
args=parser.parse_args()
directory = args.directory.resolve()
repo = directory.parents[3]
rows = list(csv.DictReader((directory/'prices.tsv').open(), delimiter='\t'))
checks = 0
for dtype in ('bf16', 'f32'):
    for model, stages in (('gqa2', 14), ('mha4', 28)):
        with (repo/f'docs/experiments/ORACLE/raw/screen_{model}.tsv').open() as f:
            oracle = list(csv.reader(f, delimiter='\t'))[1:]
        expected = {f'{r[0]}x{r[1]}x{r[2]}s{r[3]}k{r[4]}' for r in oracle if r[8]=='PASS'}
        actual = [r for r in rows if r['dtype']==dtype and r['model']==model]
        assert len(expected)==len(actual)==1077
        assert {r['config'] for r in actual}==expected
        for r in actual:
            assert r['status']=='PASS' and int(r['gemm_stages'])==stages
            assert int(r['seq_points'])==5 and int(r['price_bit_checks'])==stages*5*2
            checks += int(r['price_bit_checks'])
assert len(rows)==4308 and checks==904680
assert 'ISL_CONTEXT remaining=0' in (directory/'status.txt').read_text()
if args.stage_entry:
    assert (directory/'status.txt').read_text().count('status=PASS stage_entry_bits_equal=1')==4
print(json.dumps(dict(status='PASS',groups=len(rows),price_bit_checks=checks,
    stage_entry_bit_checks=checks if args.stage_entry else 0,
    prices_sha256=hashlib.sha256((directory/'prices.tsv').read_bytes()).hexdigest(),
    scope='GEMM only; scalar and solver evidence recorded separately'),indent=2))
