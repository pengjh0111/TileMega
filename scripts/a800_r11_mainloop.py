#!/usr/bin/env python3
"""Report guarded mainloop outcomes with exact linked launch resources."""
import argparse
import json
from pathlib import Path

from a100_baselines import inspect_inputs, sha
from a100_results import write_csv
from a800_baselines import INVENTORY
from a800_native import artifact
from a800_r10_collect_round import guarded_stats
from a800_r11_collect_round import collect
from a800_r11_diagnostics import trace_cases
from a800_r11_finish import completed
from a800_r11_gate import read
from a800_r11_screen import arm_name, arms


def architecture(path, expected):
    _, identity, manifest = artifact(Path(path))
    if expected != identity['binary_sha256']:
        raise ValueError('mainloop arm no longer matches the measured decode binary')
    [kernel] = [row for name,row in identity['resources'].items() if 'sm90_dense_native_kernel' in name]
    c = manifest['native_dense']
    threads = manifest.get('register_contract', {}).get('threads')
    return dict(threads=threads or (128 if c.get('multistage') else 128+32*c.get('producer_warps',1)),
                thread_count_source='register_contract' if threads else 'coordinates', multistage=c.get('multistage',False),
                single_join=c.get('multistage_single_join',False), residency=c['residency'],
                slots=json.dumps({k:v['slots'] for k,v in c['gemms'].items()},sort_keys=True),
                linked_registers=kernel['registers'], shared_bytes=identity['shared_memory_bytes'])


def roles_for(probe, fingerprint):
    if probe is None:
        return []
    anchor = Path(probe['anchor'])
    identity = Path(probe['identity'])
    if (read(anchor)['decode_sha256'] != fingerprint or read(identity)['anchor_sha256'] != sha(anchor)
            or sha(identity.with_name('probe')) != read(identity)['binary_sha256']):
        raise ValueError('role measurement used a different mainloop body')
    rows = guarded_stats(Path(probe['run']), {'identity_sha256':sha(identity)}, probe['gpu_uuid'])
    return [row for row in rows if row['status']=='measured']


def role_evidence(scope, runs):
    samples = {}
    for job, case in trace_cases(scope):
        if case.get('scope') != 'mainloop':
            continue
        devices = set()
        parents = {str(case[key]) for key in ('retry_of', 'outage_retry') if key in case}
        for probe in case['probes']:
            run = runs/'calibration/r2-supplement-service'/job/probe['arm']
            if not (run/'job/physical-device-identity.json').is_file():
                continue
            gpu = read(run/'job/physical-device-identity.json')['nvml_uuid']
            devices.add(gpu)
            digest = read(probe['anchor'])['decode_sha256']
            evidence = samples.setdefault(digest, [])
            if evidence and not any(row['job'] in parents for row in evidence):
                raise ValueError('bind a repeated role observation to its original job')
            evidence.append(dict(anchor=probe['anchor'], identity=probe['identity'],
                                 job=job, run=str(run), gpu_uuid=gpu))
        if len(devices)>1:
            raise ValueError('compared role samples must run in one single-GPU allocation')
    return samples


def make(path, runs, inputs, roles):
    for phase, pair, job in completed(path):
        if phase != 'mainloop' or not (runs/'optimization/decode-pairs'/job/'result.json').exists():
            continue
        document = read(pair['finalists'])
        control = read(document['control'])
        candidates = {arm_name(row['rank']): row for row in arms(document, phase)}
        observed = [fact for fact, _, _ in collect(runs, job, inputs) if fact['arm'] in candidates]
        accepted = [fact for fact in observed if fact['status'] == 'faster']
        winner = min(accepted,key=lambda row:row['tpot_ms'])['arm'] if accepted else 'control'
        control_fact = dict(observed[0], arm='control', tpot_ms=observed[0]['control_tpot_ms'],
                            improvement_pct=0, status='control', decode_sha256=control['decode_sha256'],
                            pass_rate='', passed_processes='', processes='')
        for fact in (control_fact,*observed):
            arm = fact['arm']
            digest = fact['decode_sha256']
            name = 'control' if arm=='control' else candidates[arm]['label']
            binary = control['decode'] if arm=='control' else candidates[arm]['binary']
            launch = architecture(binary,digest)
            fields = ('stage','producer_issue_fraction','consumer_wait_fraction',
                      'epilogue_fraction','grid_barrier_fraction','trace_perturbation')
            accepted_roles = {}
            for role in roles.get(digest, []):
                for sample in roles_for(role, digest):
                    accepted_roles.setdefault(sample['stage'], dict(sample, role_job=role['job'],
                                              role_gpu_uuid=role['gpu_uuid']))
            for sample in accepted_roles.values() or [{}]:
                yield dict(model=fact['model'], batch=fact['batch'], job=job, arm=name, decisive=arm==winner,
                    tpot_ms=fact['tpot_ms'], improvement_pct=fact['improvement_pct'], status=fact['status'],
                    decode_sha256=digest, gpu_uuid=fact['gpu_uuid'],
                    pass_rate=fact['pass_rate'], passed_processes=fact['passed_processes'],
                    correctness_processes=fact['processes'],
                    role_job=sample.get('role_job', ''), role_gpu_uuid=sample.get('role_gpu_uuid', ''), **launch,
                    **{field:sample.get(field,'') for field in fields})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('waves','runs','out'):
        parser.add_argument(f'--{field}', type=Path, required=True)
    parser.add_argument('--role-scope', type=Path)
    args=parser.parse_args()
    evidence = role_evidence(args.role_scope,args.runs) if args.role_scope else {}
    write_csv(args.out,list(make(args.waves,args.runs,inspect_inputs(INVENTORY),evidence)))


if __name__=='__main__':
    main()
