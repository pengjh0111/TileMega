#!/usr/bin/env python3
"""Bind stage attribution to the exact R11 release artifacts."""
import argparse
import csv
import os
from pathlib import Path
import subprocess

from a100_baselines import sha
from a800_r10_prepare import MODELS
from a800_r9_fused_trace import release_bandwidth
from a800_r11_diagnostics import trace_cases
from a800_r11_gate import read
from a800_r11_qwen_traces import prepare_trace
from tilemega.bench.scheduler import write_json


def cell(row):
    return f'{row["model"]}/B{row["batch"]}'


def matrix_rows(matrix):
    with (matrix/'A800_R11_llm.csv').open(newline='') as stream:
        rows = {cell(row):row for row in csv.DictReader(stream)}
    if len(rows) != 40:
        raise ValueError('R11 stage attribution requires the complete final matrix')
    return rows


def previous(history, rows):
    return {key:trace for key, trace in history.items() if key in rows
            and trace['original_sha256'] == rows[key]['decode_sha256'] and abs(trace['perturbation_pct']) <= 5}


def prepare(args, rows, covered):
    for key, row in rows.items():
        if key in covered or row['model'] not in MODELS:
            continue
        spec = read(args.final/'prepared'/key/'finalists.json')
        if sha(spec['control']) != spec['control_sha256']:
            raise ValueError(f'{key} release trace has a different R10 control')
        anchor = read(spec['control'])
        if anchor['decode_sha256'] != row['decode_sha256']:
            candidates = spec['candidates']
            fallback = args.final/'fallback'/key/'finalists.json'
            if fallback.is_file():
                alternate = read(fallback)
                if alternate['control_sha256'] != spec['control_sha256']:
                    raise ValueError(f'{key} fallback uses another R10 control')
                candidates = [*candidates, *alternate['candidates']]
            [winner] = [arm for arm in candidates if arm['binary_sha256']==row['decode_sha256']]
            target = winner.get('target', spec['target'])
            anchor = dict(anchor, decode=winner['binary'], decode_sha256=winner['binary_sha256'],
                          coordinates=winner['coordinates'], artifact_target=target, target=target,
                          target_sha256=winner.get('target_sha256', spec['target_sha256']))
        if anchor['target_sha256'] != row['target_sha256'] or sha(anchor['decode']) != row['decode_sha256']:
            raise ValueError(f'{key} stage analysis must use the final binary and calibration')
        folder = args.scope/'overlap'/row['model']/f'B{row["batch"]}-final-release'
        plan = folder/'overlap.rounds.json'
        if plan.exists():
            continue
        try:
            probe = prepare_trace(args.scope.parents[2], args.lock, folder/'release', anchor, 'release')
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            print(key, type(error).__name__, str(error), flush=True)
            continue
        write_json(plan, dict(model=row['model'], batch=int(row['batch']), scope='final-release',
                              decode_sha256=row['decode_sha256'], probes=[probe]))
        print(key, plan, flush=True)


def collect(args, rows, covered):
    scope = args.scope/'overlap'
    errors = []
    if (scope/'accounting.tsv').exists():
        for job, spec in trace_cases(scope):
            if spec.get('scope')!='final-release':
                continue
            key = cell(spec)
            [probe] = spec['probes']
            run = args.scope.parent/'optimization/decode-traces'/job/probe['arm']
            plan = Path(probe['trace_plan'])
            if read(plan)['original_sha256'] != rows[key]['decode_sha256']:
                raise ValueError(f'{key} attribution trace belongs to another final artifact')
            path = args.out/f'release-{job}.json'
            try:
                release_bandwidth(argparse.Namespace(run=run, plan=plan, out=path,
                                                     expert_trace=None, expected_tokens=None))
                covered[key] = dict(read(path), job=job, origin='R11')
            except (ValueError, OSError) as error:
                errors.append(dict(cell=key, job=job, reason=str(error)))
    target = args.scope/'collected-calibration/registry.json'
    hbm = read(target)['sm80_native_constants']['dram_gbps']
    result = {}
    for key, trace in covered.items():
        row = rows[key]
        tpot = float(row['r11_tpot_ms'])
        baseline = float(row['fastest_baseline_tpot_ms'])
        stages = [dict(stage, byte_floor_ns=stage['logical_bytes']/hbm,
                       excess_ns=stage['stage_span_ns']-stage['logical_bytes']/hbm,
                       fraction_of_request_tpot=stage['stage_span_ns']/(tpot*1e6)) for stage in trace['rows']]
        result[key] = dict(job=trace['job'], source=trace['origin'], decode_sha256=trace['original_sha256'],
                           perturbation_pct=trace['perturbation_pct'], stages=stages,
                           target_gap_ns=max(0,tpot-baseline/1.20)*1e6,
                           required_efficiency=1.20*float(row['baseline_efficiency']))
    write_json(args.out/'release-attribution.json', dict(cells=result, missing=sorted(rows.keys()-result.keys()),
                                                        rejected=errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare','collect'))
    for name in ('scope','final','history','lock','out'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('assemble attribution on the login node without CUDA devices')
    args.out.mkdir(parents=True, exist_ok=True)
    rows = matrix_rows(args.final/'matrix')
    covered = previous(read(args.history), rows)
    {'prepare':prepare,'collect':collect}[args.mode](args, rows, covered)


if __name__ == '__main__':
    main()
