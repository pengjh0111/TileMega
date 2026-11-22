#!/usr/bin/env python3
"""Build, sample, and import missing SM80 MMA geometries from a coverage audit."""
import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from a100_baselines import ROOT, sha
from a800_baselines import CUDA_RESOURCES, calibrated_device, device_policy
from tilemega.bench.scheduler import write_json


def read(path):
    return json.loads(path.read_text())


def cases(domain, sms):
    groups = defaultdict(set)
    for rows in domain.values():
        for row in rows:
            if row['measured']:
                continue
            grid, tn, tk, wn, stages = map(int, row['mma_key'].split(':'))
            if grid not in (sms, 2*sms):
                raise ValueError(f'unexpected SM80 calibration grid {grid}')
            groups[grid//sms, stages].add((tn, tk, wn, stages))
    return groups


def census(args):
    missing = {}
    for cell, rows in read(args.domain).items():
        for row in rows:
            if row['measured']:
                continue
            key = row['mma_key']
            record = missing.setdefault(key, dict(family_options=0, cells=set(), families=set()))
            record['family_options'] += row['family_options']
            record['cells'].add(cell)
            record['families'].add(row['family'])
    with args.out.open('x', newline='') as stream:
        fields = ('mma_key', 'family_options', 'cells', 'families')
        writer = csv.DictWriter(stream, fields, lineterminator='\n')
        writer.writeheader()
        for key in sorted(missing, key=lambda value: tuple(map(int, value.split(':')))):
            row = missing[key]
            writer.writerow(dict(mma_key=key, family_options=row['family_options'],
                                 cells=len(row['cells']), families=' '.join(sorted(row['families']))))
    print(f'{len(missing)} missing MMA calibration keys', flush=True)


def build(args):
    if os.environ.get('SLURM_JOB_ID'):
        raise ValueError('compile mainloop probes on the login node')
    base = read(args.target)
    resources = base['resources']
    sms = resources['num_sms']
    if not resources['max_smem_per_sm']//3 < args.dual_shared <= resources['max_smem_per_sm']//2:
        raise ValueError('use the linked shared footprint of a dual-resident production artifact')

    source = ROOT/'scripts/sm90r2_mainloop_calibration.cu'
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = []
    chunks = []
    for (residency, stages), available in sorted(cases(read(args.domain), sms).items(), reverse=True):
        shapes = sorted(available)
        block = args.cases_per_binary or len(shapes)
        for offset in range(0, len(shapes), block):
            label = f'r{residency}-s{stages}'
            if args.cases_per_binary:
                label += f'-{offset//block}'
            chunks.append((residency, label, shapes[offset:offset+block]))
    for residency, label, shapes in chunks:
        scope = args.out/label
        scope.mkdir(exist_ok=True)
        binary = scope/'mainloop'
        entries = scope/'cases.inc'
        entries.write_text(''.join(
            f'  Case<{tn},{tk},{wn},{r}>(p,frequency,times,errors);\n'
            for tn, tk, wn, r in shapes))
        options = ['-std=c++17', '-O3', '-arch=sm_80', '-DTILEMEGA_ARCH_ID=800',
                   f'-DTILEMEGA_A800_MMA_CASE_FILE="{entries}"',
                   '--expt-relaxed-constexpr', '--expt-extended-lambda', '-Xptxas=-v',
                   f'-I{ROOT}/include', f'-I{ROOT}/third_party/cutlass/include']
        if residency == 2:
            options.append(f'-DTILEMEGA_A800_DUAL_MMA_SHARED={args.dual_shared}')
        ptxas = scope/'ptxas.log'
        with ptxas.open('w') as stream:
            subprocess.run(['nvcc', *options, str(source), '-o', str(binary)],
                           stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=670)
        audit = scope/'audit.json'
        subprocess.run([sys.executable, str(ROOT/'scripts/audit_sm90r2_mainloop_calibration.py'),
                        '--binary', str(binary), '--ptxas-log', str(ptxas), '--out', str(audit)],
                       check=True, timeout=135)
        accepted = read(audit)
        if len(accepted['kernels']) != len(shapes):
            raise ValueError(f'{scope.name} did not compile every requested case')
        outputs.append(dict(label=scope.name, grid=residency*sms, binary=str(binary),
                            binary_sha256=sha(binary), audit=str(audit), audit_sha256=sha(audit),
                            compatible_cases=accepted['compatible_cases'],
                            requested=len(shapes), resource_rejected=len(shapes)-accepted['compatible']))
    if not outputs:
        raise ValueError('no missing geometries were requested')
    write_json(args.out/'plan.json', dict(schema=1, source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), source_sha256=sha(source),
        driver_sha256=sha(Path(__file__)), target=str(args.target), target_sha256=sha(args.target),
        domain_sha256=sha(args.domain), jobs=outputs, dual_shared=args.dual_shared))


def frozen(plan_path, expected):
    plan = read(plan_path)
    if sha(plan_path) != expected or sha(Path(__file__)) != plan['driver_sha256']:
        raise ValueError('queued calibration driver or plan changed')
    target = Path(plan['target'])
    if sha(target) != plan['target_sha256']:
        raise ValueError('queued calibration target changed')
    for job in plan['jobs']:
        if sha(Path(job['binary'])) != job['binary_sha256'] or sha(Path(job['audit'])) != job['audit_sha256']:
            raise ValueError(f"queued calibration binary or audit changed for {job['label']}")
    return plan, read(target)


def sample(args, plan, target):
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise ValueError('mainloop probes require the strict guard and GPU lock')
    for job in plan['jobs']:
        cases_csv = job['compatible_cases']
        if not cases_csv:
            continue
        case = [cases_csv] if job['grid'] != target['resources']['num_sms'] else ['--cases', cases_csv]
        with (args.out/f"{job['label']}.out").open('x') as out:
            subprocess.run([job['binary'], *case], stdout=out, check=True, timeout=90)
        print(job['label'], len(cases_csv.split(',')), flush=True)


def run(args, plan, target):
    policy = device_policy()
    device = calibrated_device(policy, target)
    if device['driver'] != target['device_provenance']['driver']:
        raise ValueError('the allocation driver differs from the calibration registry')
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out/'admitted-device.json', device)
    policy_file = args.out/'policy.json'
    write_json(policy_file, policy)
    env = dict(os.environ, TILEMEGA_GPU_PID_MODE='strict', TILEMEGA_TIMING='0',
               TILEMEGA_DEVICE_INDEX=str(device['index']), TILEMEGA_NUMA_NODE=device['numa_node'],
               TILEMEGA_GPU_LOCK=policy['gpu_lock_path'])
    command = [sys.executable, '-m', 'tilemega.bench.gpu_guard', '--policy', str(policy_file),
               '--out', str(args.out/'guard'), '--needs-free-mib', '1024', '--timeout-s', '655', '--',
               'taskset', '--cpu-list', device['cpus'], sys.executable,
               str(ROOT/'scripts/run_r16_with_device_identity.py'),
               '--out', str(args.out/'device'), '--guard-policy', str(policy_file),
               '--expected-uuid', device['uuid'], '--expected-pci', device['pci'],
               '--numa-node', device['numa_node'], '--allow-no-measurements', '--',
               sys.executable, str(Path(__file__).resolve()), 'sample', '--plan', str(args.plan),
               '--plan-sha256', args.plan_sha256, '--out', str(args.out)]
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def collect(args, plan, target):
    if os.environ.get('SLURM_JOB_ID'):
        raise ValueError('import calibration on the login node')
    guard_file = args.out/'guard/guard_result.json'
    device_file = args.out/'device/physical-device-identity.json'
    policy_file = args.out/'policy.json'
    admitted_file = args.out/'admitted-device.json'
    guard, device, policy, admitted = map(read, (guard_file, device_file, policy_file, admitted_file))
    if (guard['code'] or guard['acceptance'] != 'strict' or not guard['ownership_verified']
            or device['nvml_uuid'] != policy['gpu_uuid']
            or device['guard_policy_sha256'] != sha(policy_file)
            or device['compute_capability'] != [8, 0]
            or admitted['uuid'] != device['nvml_uuid']
            or admitted['driver'] != target['device_provenance']['driver']
            or any(device[attribute] != target['resources'][field]
                   for field, attribute in CUDA_RESOURCES.items())):
        raise ValueError('calibration must use the guarded and equivalent A800 target')
    sms = target['resources']['num_sms']
    table = target['sm80_native_constants']['mma_mainloop']
    accepted = {}
    for job in plan['jobs']:
        if not job['compatible_cases']:
            continue
        path = args.out/f"{job['label']}.out"
        dual = job['grid'] == 2*sms
        with path.open() as stream:
            rows = [json.loads(line) for line in stream] if dual else list(csv.DictReader(stream))
        expected = set(job['compatible_cases'].split(','))
        seen = set()
        for row in rows:
            geometry = tuple(int(row[k]) for k in ('tile_n','tile_k','warps_n','register_stages'))
            case = 'x'.join(map(str, geometry))
            key = ':'.join(map(str, (job['grid'], *geometry)))
            if (case not in expected or case in seen or int(row['grid']) != job['grid']
                    or int(row['warps_k'])*geometry[2] != 4 or int(row['samples']) <= 0
                    or key in accepted or key in table['median_ns']):
                raise ValueError(f'invalid or duplicate calibration row for {key}')
            if dual:
                if (row['status'] != 'measured' or row['overlap_sms'] != sms
                        or row['dynamic_shared'] != plan['dual_shared'] or row['local_bytes']
                        or row['input_norm'] is not True):
                    raise ValueError(f'dual-resident measurement failed its resource checks: {key}')
                median,p99 = (row[k] for k in ('median_ns','p99_ns'))
            else:
                if row['metric'] != 'native_ldmatrix_mma_k16' or row['unit'] != 'ns':
                    raise ValueError(f'incorrect probe metric for {key}')
                median,p99 = (float(row[k]) for k in ('median','p99'))
            if not 0 < median <= p99 < math.inf:
                raise ValueError(f'invalid MMA statistics for {key}')
            accepted[key] = (median,p99)
            seen.add(case)
        if seen != expected:
            raise ValueError(f"{job['label']} missed compatible shapes {sorted(expected-seen)}")
    if not accepted:
        raise ValueError('no guarded calibration measurements')
    for key, (median,p99) in sorted(accepted.items()):
        table['median_ns'][key] = median
        table['p99_ns'][key] = p99
    table.setdefault('supplements', []).append(dict(
        parent_profile_sha256=plan['target_sha256'], method='isolated SM80 MMA per K16 and split-K warp group',
        plan_sha256=args.plan_sha256, guard_result_sha256=sha(guard_file),
        device_receipt_sha256=sha(device_file), admission_sha256=sha(admitted_file),
        gpu_uuid=device['nvml_uuid'],
        source_commit=plan['source_commit'], shape_keys=sorted(accepted), processes_per_series=1,
        evidence=str(args.out), dual_shared=plan['dual_shared']))
    write_json(args.output_target, target)
    print(f'{len(accepted)} guarded MMA measurements; {len(table["median_ns"])} total', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    b = sub.add_parser('build')
    for flag in ('domain','target','out'):
        b.add_argument(f'--{flag}', required=True, type=Path)
    b.add_argument('--dual-shared', required=True, type=int)
    b.add_argument('--cases-per-binary', type=int)
    listing = sub.add_parser('census')
    listing.add_argument('--domain', required=True, type=Path)
    listing.add_argument('--out', required=True, type=Path)
    for mode in ('run','sample','collect'):
        stage = sub.add_parser(mode)
        stage.add_argument('--out', type=Path, required=True)
        stage.add_argument('--plan', type=Path, required=True)
        stage.add_argument('--plan-sha256', required=True)
        if mode == 'collect':
            stage.add_argument('--output-target', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'build':
        if args.cases_per_binary is not None and args.cases_per_binary < 1:
            parser.error('--cases-per-binary must be positive')
        build(args)
    elif args.mode == 'census':
        census(args)
    else:
        plan, target = frozen(args.plan, args.plan_sha256)
        {'run':run, 'sample':sample, 'collect':collect}[args.mode](args, plan, target)


if __name__ == '__main__':
    main()
