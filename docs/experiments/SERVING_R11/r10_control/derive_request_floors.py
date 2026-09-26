#!/usr/bin/env python3
"""Evaluate exact CG request floors for the eight immutable R10 controls."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PLANS = Path('/root/r11_work/r10_control/plans')
TOOL = Path('/root/r11_work/build/tools/tilemega')
TARGET = HERE / 'calibration/target_serving.json'


def selected_cg(cell: Path, ranks: dict[str, int]) -> Path:
    rank = ranks[cell.name]
    cg = cell / f'plan.so.top{rank}.mlir'
    if not cg.is_file():
        raise FileNotFoundError(cg)
    return cg


def main() -> None:
    destination = HERE / 'floors'; destination.mkdir(parents=True, exist_ok=True)
    with (HERE / 'plans_summary.tsv').open(newline='') as stream:
        ranks = {row['cell']: int(row['top3_rank']) for row in
                 csv.DictReader(stream, delimiter='\t')}
    rows = []
    for model in ('llama', 'qwen3'):
        for batch in (1, 16):
            floors = {}
            for phase, lo, hi in (('prefill', 0, 0), ('decode', 64, 1086)):
                cg = selected_cg(PLANS / f'{model}_{phase}_B{batch}', ranks)
                name = f'{model}_{phase}_B{batch}'
                output = destination / f'{name}.json'
                steps = destination / f'{name}.tsv'
                command = [str(TOOL), 'inspect', 'request-floor', str(cg), str(TARGET),
                           str(batch), str(lo), str(hi), str(output), str(steps)]
                with (destination / f'{name}.log').open('w') as log:
                    subprocess.run(command, cwd=ROOT, stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
                value = json.loads(output.read_text())
                floors[phase] = value['sum_floor_seconds']
                rows.append(dict(model=model, batch=batch, phase=phase,
                    sum_floor_seconds=floors[phase], cg=str(cg),
                    cg_sha256=hashlib.sha256(cg.read_bytes()).hexdigest(),
                    target_sha256=hashlib.sha256(TARGET.read_bytes()).hexdigest()))
            rows.append(dict(model=model, batch=batch, phase='request',
                sum_floor_seconds=sum(floors.values()), cg='', cg_sha256='',
                target_sha256=hashlib.sha256(TARGET.read_bytes()).hexdigest()))
    with (HERE / 'request_floors.tsv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    print(HERE / 'request_floors.tsv')


if __name__ == '__main__':main()
