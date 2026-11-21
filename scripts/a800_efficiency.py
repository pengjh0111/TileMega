#!/usr/bin/env python3
"""Calculate the matched 40-cell A800 memory roofline from frozen inputs."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calculate(native, baseline, freeze, roofline, target):
    guarantee = json.loads(freeze.read_text())
    if digest(baseline) != guarantee['table_sha256']:
        raise ValueError('the baseline matrix differs from its frozen receipt')
    current = rows(native)
    if len(current) != 40 or any(row['status'] != 'measured' for row in current):
        raise ValueError('efficiency requires the complete measured TileMega matrix')
    bounds = {}
    for row in rows(roofline):
        if row['system'] == 'TileMega' and row['tp'] == '1' and row['basis'] == 'measured':
            key = row['model'], row['batch']
            if key in bounds:
                raise ValueError(f'duplicate TP1 roofline data for {key}')
            bounds[key] = row
    comparisons = defaultdict(list)
    for row in rows(baseline):
        if row['status'] == 'measured':
            comparisons[row['model'], row['batch']].append(row)
    bandwidth = json.loads(target.read_text())['sm80_native_constants']['dram_gbps']
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('require a measured A800 HBM bandwidth')
    result = {}
    for row in current:
        key = row['model'], row['batch']
        bound = bounds[key]
        fastest = min(comparisons[key], key=lambda r: float(r['tpot_ms']))
        if any(r['config_sha256'] != row['config_sha256'] or r['prompt_sha256'] != row['prompt_sha256']
               for r in comparisons[key]) or bound['model_config_sha256'] != row['config_sha256']:
            raise ValueError(f'mismatched workload for {key}')
        tpot, best = (float(r['tpot_ms']) for r in (row, fastest))
        bytes_step = float(bound['bytes_step'])
        floor = bytes_step/bandwidth/1e6
        if key in result or min(tpot,best,floor)<=0 or not all(map(math.isfinite,(tpot,best,floor))):
            raise ValueError(f'duplicate or invalid roofline result {key}')
        ratio = best/tpot
        result[key] = dict(model=key[0], batch=int(key[1]), bytes_step=bytes_step,
            a800_hbm_gbps=bandwidth, memory_floor_ms=floor, tilemega_tpot_ms=tpot,
            fastest_baseline=fastest['framework'], baseline_tpot_ms=best, speedup=ratio,
            tilemega_efficiency=floor/tpot, baseline_efficiency=floor/best,
            comparison='tie' if abs(ratio-1)<=0.005 else 'win' if ratio>1 else 'loss',
            native_decode_sha256=row['decode_sha256'], source='matched SM90R6 logical bytes')
    if set(result)!=set(bounds) or set(result)!=set(comparisons):
        raise ValueError('native, frozen and roofline matrices cover different cells')
    return list(result.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT/'docs/results'
    for flag, path in (('native',base/'A800_llm.csv'), ('baseline',base/'A800_baselines.csv'),
                       ('freeze',base/'A800_baselines_frozen.json'),
                       ('roofline',base/'SM90R6_hw_bound.csv'),
                       ('target',ROOT/'configs/targets/sm_80_a800_sxm4_80gb_registry.json')):
        parser.add_argument(f'--{flag}', default=path, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = calculate(args.native,args.baseline,args.freeze,args.roofline,args.target)
    with args.out.open('x',newline='') as stream:
        writer = csv.DictWriter(stream, list(result[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(result)
    print(json.dumps(dict(cells=len(result), raw_wins=sum(r['speedup']>1 for r in result),
        wins=sum(r['comparison']=='win' for r in result),
        ties=sum(r['comparison']=='tie' for r in result),
        losses=sum(r['comparison']=='loss' for r in result),
        geomean={key:math.exp(math.fsum(math.log(row[key]) for row in result)/len(result))
                 for key in ('speedup','tilemega_efficiency','baseline_efficiency')}), indent=2))


if __name__ == '__main__':
    main()
