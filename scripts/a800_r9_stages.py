#!/usr/bin/env python3
"""Prepare and run the frozen R9 dense stage width array."""
import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

from a800_baselines import RUN
from a100_baselines import ROOT, sha
from prepare_a800_dense_service import production_shape
from tilemega.bench.scheduler import write_json


def read(path):
    return json.loads(Path(path).read_text())


def source():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def geometry(n, k, tn, tk, wn, stages, slots, residency, family, spec, batch):
    resources = spec['resources']
    grid = resources['num_sms'] * residency
    tasks = (n+tn-1)//tn
    if tn > n or (n % tn and (n % 32 or tasks > 3*grid)):
        return None, 'unsupported N tile or padded task count'
    side = max(2, (tasks+grid-1)//grid) if family in ('o', 'down') else 0
    if side > 3:
        return None, 'residual storage supports three tasks per CTA'
    limit = min(resources['max_dynamic_smem_per_cta'],
                resources['max_smem_per_sm']//residency)
    activation = 8 if batch <= 8 else 16
    # Change K or warp partition only when the incumbent shape cannot be linked.
    for depth, warps_n in itertools.product(
            dict.fromkeys((tk, 64, 128, 256, 512, 1024, 2048)),
            dict.fromkeys((wn, 4, 2, 1))):
        warps_k = 4//warps_n
        shared = 1536+2*slots*depth*(tn+activation)+64*warps_k*tn+2*side*batch*tn
        if (n < tn or tn % (8*warps_n) or tn//warps_n > 256 or k % depth
                or depth < 16*warps_k*stages or shared > limit):
            continue
        return {'tk': depth, 'warps_n': warps_n, 'static_shared_budget_bytes': shared}, ''
    return None, 'no compatible K tile and warp partition fit the shared memory budget'


def prepare(args):
    if os.environ.get('SLURM_JOB_ID'):
        raise ValueError('prepare linked stage artifacts on the login node')
    config = read(args.config)
    spec = read(args.target)
    anchors = read(args.anchors)
    args.out.mkdir(parents=True, exist_ok=True)
    cohort = []
    for model, family in itertools.product(config['models'], config['families']):
        batch = config['batch']
        anchor_path = Path(anchors[f'{model}/B{batch}'])
        anchor = read(anchor_path)
        if anchor['model'] != model or anchor['batch'] != batch:
            raise ValueError('anchor identity differs from the cohort index')
        coords = anchor['coordinates']
        incumbent = coords['gemms'][family]
        generated = Path(anchor['decode']+'.cu').read_text()
        _, n, k, tn, tk, _, wn, _, _, _ = production_shape(
            generated, family, batch, incumbent)
        groups = []
        for residency in config['residencies']:
            arms, flags = [], []
            for scale, slots in itertools.product(config['width_multipliers'], config['slots']):
                width = scale*tn
                g, reason = geometry(n, k, width, tk, wn, incumbent['register_stages'],
                                     slots, residency, family, spec, batch)
                label = f'{model}-b{batch}-{family}-u16-rows0-copy0-slots{slots}'
                if g:
                    flags.extend(('--ring', f"{family}:{slots}:{g['tk']}:{width}:{g['warps_n']}"))
                    label += f"-tk{g['tk']}-tn{width}-wn{g['warps_n']}"
                arms.append(dict(case=label, multiplier=scale, slots=slots, tn=width,
                                 geometry=g, preparation_rejection=reason))
            folder = args.out/f'{model}-b{batch}-{family}-r{residency}'
            identity = folder/'probe.identity.json'
            if flags:
                if not identity.exists():
                    subprocess.run([sys.executable, str(ROOT/'scripts/prepare_a800_dense_service.py'),
                                    '--anchor', str(anchor_path), '--out', str(folder),
                                    '--residency', str(residency), '--families', family,
                                    '--width-scan', *flags], check=True)
                linked = read(identity)
                expected = {a['case'] for a in arms if a['geometry']}
                if (set(linked['cases']) != expected or linked['source_sha'] != source()
                        or linked['anchor_sha256'] != sha(anchor_path)):
                    raise ValueError(f'{folder} has a stale linked probe')
            groups.append(dict(residency=residency, arms=arms, binary=str(folder/'probe'),
                               identity_sha256=sha(identity) if flags else None))
        cohort.append(dict(model=model, batch=batch, family=family, groups=groups,
                           anchor_sha256=sha(anchor_path)))
    output = args.out/'scan.plan.json'
    if output.exists():
        raise ValueError('retain the existing immutable cohort plan')
    write_json(output, dict(schema=1, source_sha=source(), target=str(args.target.resolve()),
                            target_sha256=sha(args.target), config_sha256=sha(args.config),
                            script_sha256=sha(Path(__file__)), cohort=cohort))
    print(output, sha(output), len(cohort), flush=True)


def load_plan(args):
    plan = read(args.plan)
    if (plan['schema'] != 1 or plan['script_sha256'] != sha(Path(__file__))
            or sha(Path(plan['target'])) != plan['target_sha256']
            or sha(args.plan) != args.plan_sha256 or plan['source_sha'] != source()):
        raise ValueError('the prepared width scan inputs changed')
    return plan


def evidence(job, label):
    return RUN/'calibration'/'r2-supplement-service'/str(job)/label


def run(args, plan):
    job = int(os.environ['SLURM_JOB_ID'])
    index = args.index
    if not 0 <= index < len(plan['cohort']):
        raise ValueError('array index is out of range')
    for group in plan['cohort'][index]['groups']:
        if group['identity_sha256']:
            label = f'r9-stage-r{group["residency"]}'
            subprocess.run([sys.executable, str(ROOT/'scripts/run_a800_dense_service.py'),
                            '--binary', group['binary'], '--identity-sha256', group['identity_sha256'],
                            '--target', plan['target'], '--target-sha256', plan['target_sha256'],
                            '--run-label', label], check=True, timeout=375)
    print(json.dumps(dict(job=job, index=index, plan_sha256=args.plan_sha256)), flush=True)


def collect(args, plan):
    jobs = read(args.jobs)
    results = []
    for index, item in enumerate(plan['cohort']):
        job = jobs[str(index)]
        base = {key: item[key] for key in ('model', 'batch', 'family')}
        uuids = set()
        for group in item['groups']:
            residency = group['residency']
            folder = evidence(job, f'r9-stage-r{residency}')
            if group['identity_sha256']:
                guard = read(folder/'guard/guard_result.json')
                device = read(folder/'job/physical-device-identity.json')
                identity = Path(group['binary']).with_suffix('.identity.json')
                if (not guard['ownership_verified'] or guard['code'] or guard['acceptance'] != 'strict'
                        or sha(identity) != group['identity_sha256']
                        or read(folder/'artifact.json') != read(identity)
                        or read(folder/'frozen.json')['identity_sha256'] != group['identity_sha256']
                        or device['guard_policy_sha256'] != sha(folder/'policy.json')
                        or device['nvml_uuid'] != read(folder/'policy.json')['gpu_uuid']):
                    raise ValueError(f'unaccepted stage guard for {job}')
                uuids.add(device['nvml_uuid'])
            for arm in group['arms']:
                row = dict(**base, job=job, residency=residency, tn=arm['tn'],
                           slots=arm['slots'], multiplier=arm['multiplier'],
                           status='resource_rejected', reason=arm['preparation_rejection'])
                if arm['geometry']:
                    observation = read(folder/f"{arm['case']}.json")
                    row['status'] = observation['status']
                    if row['status'] == 'measured':
                        if (observation['tn'] != arm['tn'] or observation['slots'] != arm['slots']
                                or observation['requested_residency'] != residency
                                or observation['stage_span_ns'] <= 0
                                or not 0 < observation['active_ctas'] <= observation['grid']):
                            raise ValueError('results differ from the immutable stage arm')
                        if abs(observation['logical_weight_bytes']/observation['stage_span_ns']
                               - observation['effective_gbps']) > .001:
                            raise ValueError('invalid effective bandwidth')
                        row.update({key: observation[key] for key in
                                    ('stage_span_ns', 'active_ctas', 'overlapping_sms', 'effective_gbps',
                                     'logical_weight_bytes', 'padded_n', 'clipped_output_bitwise_equal')})
                        row.update(tk=arm['geometry']['tk'], warps_n=arm['geometry']['warps_n'])
                    elif row['status'] == 'resource_rejected':
                        row['reason'] = 'linked kernel cannot attain requested residency'
                    else:
                        raise ValueError(f"unexpected stage status {row['status']}")
                results.append(row)
        if len(uuids) != 1:
            raise ValueError(f'dense sweep entry {index} did not share one GPU allocation')
    fields = list(dict.fromkeys(itertools.chain.from_iterable(row.keys() for row in results)))
    with args.out.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    make = modes.add_parser('prepare')
    for arg in ('config', 'target', 'anchors', 'out'):
        make.add_argument(f'--{arg}', required=True, type=Path)
    for mode in ('run', 'collect'):
        command = modes.add_parser(mode)
        command.add_argument('--plan', required=True, type=Path)
        command.add_argument('--plan-sha256', required=True)
        if mode == 'run':
            command.add_argument('--index', required=True, type=int)
        else:
            command.add_argument('--jobs', required=True, type=Path)
            command.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args)
    else:
        {'run': run, 'collect': collect}[args.mode](args, load_plan(args))


if __name__ == '__main__':
    main()
