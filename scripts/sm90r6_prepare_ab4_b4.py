#!/usr/bin/env python3
"""Prepare and independently solve the four R6 aggregate B4 cells."""
import argparse
import concurrent.futures
from pathlib import Path
import subprocess

from sm90d1_common import ROOT, read, sha, write


def prepare(out, search, compiler):
    original = read(ROOT / 'runs/sm90r6/ab5/domains-v1/cells.json')
    seeds = {
        'qwen3_1_7b': ROOT / 'configs/serving/sm90r3/qwen3_1_7b_b4.json',
        'llama3_1_8b': ROOT / 'configs/serving/sm90r3/llama3_1_8b_b4.json',
        'qwen3_30b_a3b': ROOT / 'configs/serving/sm90r5/ref_r4/qwen3_30b_a3b_b4.json',
        'olmoe_1b_7b': ROOT.parent / 'TileMega-r5/runs/sm90r5/a1/olmoe-B4/winner.tmexec.json',
    }
    out.mkdir(parents=True, exist_ok=False)
    cells = []
    for model, source in seeds.items():
        parent = next(c for c in original if c['model'] == model and c['batch'] == 1)
        if sha(parent['export']) != parent['export_sha256']:
            raise ValueError('original decode export changed')
        directory = out / (model + '-B4')
        directory.mkdir()
        value = read(source)
        seed = directory / 'seed.json'
        write(seed, value.get('coordinates', value))
        cell = dict(id=model + '-B4', model=model, tp=1, batch=4, arm='AB4b', budget_ms=600000,
            seed=str(seed), seed_sha256=sha(seed), seed_source=str(source), seed_source_sha256=sha(source),
            export=parent['export'], export_sha256=parent['export_sha256'],
            export_provenance='original R6 batch-independent decode export; runtime batch supplied separately')
        cells.append(cell)
        write(directory / 'cells.json', [cell])
    write(out / 'cells.json', cells)
    paths = [search, compiler, ROOT / 'configs/targets/sm_90_h800.json',
             ROOT / 'scripts/sm90r6_search_arms.cpp', Path(__file__)]
    write(out / 'source.json', dict(protocol='R6 AB4b/AB4c B4 two-arm supplement',
        source_hashes={str(p.resolve()): sha(p) for p in paths},
        seed_rule='R3 dense B4 finals; REF-R4 Qwen MoE B4; R5 A1 OLMoE B4 winner, matching original R6 families',
        timing_protocol='three rotating rounds over AB4b and AB4c; 3 fresh processes per arm; 1 warmup plus 3 samples'))
    return cells


def solve(cell, out, search):
    directory = out / cell['id']
    command = list(map(str, [search, ROOT / 'configs/targets/sm_90_h800.json', ROOT,
                            directory / 'cells.json', directory / 'search.jsonl']))
    write(directory / 'search-command.json', command)
    with (directory / 'search.log').open('w') as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    import json
    rows = [json.loads(line) for line in (directory / 'search.jsonl').read_text().splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]['finalists']) != 4 or rows[0]['input'] != cell:
        raise ValueError('incomplete aggregate search receipt')
    write(directory / 'receipt.json', rows[0])
    print(cell['id'] + ' aggregate search complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--search', type=Path, required=True)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    out, search, compiler = args.out.resolve(), args.search.resolve(), args.compiler.resolve()
    cells = prepare(out, search, compiler)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda c: solve(c, out, search), cells))
