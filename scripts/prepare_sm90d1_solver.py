#!/usr/bin/env python3
"""Export solver classes and convert the frozen per-GEMM table to that API."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import csv
import os
import subprocess

from sm90d1_common import ROOT, RUN, read, write


def prepare(cell):
    directory = RUN / 'solver' / cell
    command = read(directory / 'solve-command.json')
    command[0] = str(RUN / 'compiler-class-map2/tilemega')
    command[3] = str(directory / 'map-only.cu')
    command += ['--diagnostic-class-map', str(directory / 'classes.tsv')]
    with (directory / 'class-map.log').open('w') as output:
        subprocess.run(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                       stdout=output, stderr=subprocess.STDOUT, check=True, timeout=180)
    config = read(ROOT / f'configs/serving/sm90d1/reference/{cell}.json')
    plan = read(RUN / f"build/reference2/{config['model']}-B1/decode.so.plan.json")
    groups = defaultdict(list)
    for row in csv.DictReader((directory / 'classes.tsv').open(), delimiter='\t'):
        groups[int(row['class'])].append(int(row['gemm']))
    geometries, conflicts = [], []
    for index, ids in sorted(groups.items()):
        values = [{k:v for k,v in plan['gemms'][i].items() if k != 'index'} for i in ids]
        if any(value != values[0] for value in values):
            conflicts.append(dict(operator_class=index, gemms=ids, geometries=values))
        geometries.append(values[0])
    write(directory / 'class-map-conversion.json', dict(
        classes=len(groups), gemms=len(plan['gemms']), conflicts=conflicts))
    if conflicts:
        return dict(cell=cell, status='unrepresentable_per_class_geometry', conflicts=len(conflicts))
    cases = directory / 'evaluate-classes.json'
    write(cases, dict(cases=[dict(kappa=plan['kappa'],
        residency=int(plan['requested_residency'] or 1), geometries=geometries)]))
    command = read(directory / 'evaluate-command.json')
    command[command.index('--evaluate-configs')+1] = str(cases)
    command[3] = str(directory / 'evaluate-classes.cu')
    write(directory / 'evaluate-class-command.json', command)
    with (directory / 'evaluate-classes.log').open('w') as output:
        result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                stdout=output, stderr=subprocess.STDOUT, timeout=900)
    return dict(cell=cell, classes=len(groups), gemms=len(plan['gemms']),
                valid_evaluation_exit=result.returncode)


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(prepare, cell) for cell in
                   ('qwen3_0_6b_b1','llama3_2_1b_b1','llama3_1_8b_b1')]
        for future in as_completed(futures):
            print(future.result(), flush=True)
