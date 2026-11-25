#!/usr/bin/env python3
"""Prepare matched stage traces for eligible Qwen router and output trials."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from a100_baselines import inspect_inputs, sha
from a800_baselines import INVENTORY
from a800_r11_collect_round import collect
from a800_r11_finish import completed
from a800_r11_gate import read
from a800_r11_screen import arm_name, arms
from tilemega.bench.scheduler import write_json


def selected_arms(candidates, facts):
    eligible = {fact['arm']: fact for fact in facts if fact['pass']}
    families = {}
    for row in candidates:
        arm = arm_name(row['rank'])
        if arm not in eligible:
            continue
        fact = eligible[arm]
        if row['binary_sha256'] != fact['decode_sha256']:
            raise ValueError('the Qwen trace must use the measured artifact')
        family = 'router' if row['label'].startswith('router-') else 'output'
        if family not in families or fact['tpot_ms'] < families[family][0]['tpot_ms']:
            families[family] = fact, row
    return list(families.values())


def prepare_trace(root, lock, folder, anchor, kind='overlap'):
    binding = folder/'anchor.json'
    folder.mkdir(parents=True, exist_ok=True)
    if binding.exists() and read(binding) != anchor:
        raise ValueError('the selected trace anchor changed')
    if not binding.exists():
        write_json(binding, anchor)
    trace = folder/'trace.plan.json'
    if not trace.exists():
        probe = ['--release-ledger', '--reuse-exact-control'] if kind == 'release' else ['--stage-overlap']
        subprocess.run([sys.executable, str(root/'scripts/a800_r9_fused_trace.py'), 'prepare',
                        '--anchor', str(binding), '--out', str(folder), *probe,
                        '--build-lock', str(lock)], cwd=root, check=True, timeout=1400)
    payload = read(trace)
    if (payload['original_sha256'] != anchor['decode_sha256']
            or payload['target_sha256'] != anchor['target_sha256'] or payload['trace_kind'] != kind):
        raise ValueError('trace artifacts do not match their timing provenance')
    args = ['--model', anchor['model'], '--batch', str(anchor['batch']), '--prefill', anchor['prefill'],
            '--reference', payload['arms']['control']['path'], '--trace', payload['arms']['trace']['path'],
            '--target', anchor['target'], '--timeout-s',
            '1080' if anchor['model']=='qwen3_30b_a3b' else '570', '--independent-arms',
            '--expected-checkpoint-sha256', anchor['checkpoint_sha256'], '--run-label', folder.name]
    command = [sys.executable, str(root/'scripts/a800_trace_decode.py'), 'plan', *args]
    text = subprocess.check_output(command, cwd=root, timeout=110)
    proof = folder/'stage.plan.json'
    if proof.exists() and proof.read_bytes() != text:
        raise ValueError('the prepared trace admission changed')
    if not proof.exists():
        proof.write_bytes(text)
    return dict(arm=folder.name, args=args, proof=str(proof), proof_sha256=sha(proof), trace_plan=str(trace))


def prepare_output_roles(folder, anchor, label):
    coordinates = anchor['coordinates']
    if (coordinates['gemms']['o'].get('split_k', 1) != 1 or coordinates.get('l2_weight_preissue')
            or coordinates.get('producer_warps', 1) != 1):
        return None
    role = folder/f'role-{label}'
    binary = role/'probe'
    identity = role/'probe.identity.json'
    binding = folder/label/'anchor.json'
    if not identity.exists() and not role.exists():
        preparer = Path(__file__).resolve().with_name('prepare_a800_dense_service.py')
        command = [sys.executable, str(preparer), '--anchor', str(binding), '--out', str(role),
                   '--families', 'o', '--role-clock', '--reserve-program-shared']
        for key, flag in (('multistage', '--multistage'), ('multistage_single_join', '--single-join'),
                          ('residual_copy_issue', '--residual-copy'), ('weight_evict_first', '--weight-evict-first')):
            if coordinates.get(key):
                command.append(flag)
        if coordinates.get('epilogue_rows') == 'live':
            command.append('--live')
        if coordinates.get('copy_issue_group'):
            command.extend(('--copy', str(coordinates['copy_issue_group'])))
        with (folder/f'role-{label}.log').open('a') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=380)
    if not identity.exists():
        return None
    linked = read(identity)
    if linked['anchor_sha256'] != sha(binding) or linked['binary_sha256'] != sha(binary):
        raise ValueError('Qwen output role probe changed after linking')
    return dict(arm=role.name, kind='role', limit=405, identity=str(identity),
                binary_sha256=linked['binary_sha256'],
                args=['--binary', str(binary), '--identity-sha256', sha(identity),
                      '--target', anchor['target'], '--target-sha256', anchor['target_sha256'],
                      '--run-label', role.name])


def prepare(root, lock, folder, manifest, facts, job):
    control = read(manifest['control'])
    arms = selected_arms(manifest['candidates'], facts)
    if not arms:
        return None
    control_fact = dict(job=job, decode_sha256=control['decode_sha256'],
                        tpot_ms=facts[0]['control_tpot_ms'], status='control', improvement_pct=0)
    probes = [dict(prepare_trace(root, lock, folder/'control', control), timing=control_fact)]
    anchors = {'control': control}
    for fact, row in arms:
        target = row.get('target', manifest['target'])
        digest = row.get('target_sha256', manifest['target_sha256'])
        anchor = dict(control, decode=row['binary'], decode_sha256=row['binary_sha256'],
                      coordinates=row['coordinates'], target=target, artifact_target=target, target_sha256=digest)
        probes.append(dict(prepare_trace(root, lock, folder/row['label'], anchor), timing=fact))
        anchors[row['label']] = anchor
    for label, anchor in anchors.items():
        role = prepare_output_roles(folder, anchor, label)
        if role is not None:
            probes.append(role)
    value = dict(model=control['model'], batch=control['batch'], scope='qwen-winners', timing_job=job, probes=probes)
    plan = folder/'overlap.rounds.json'
    if plan.exists() and read(plan) != value:
        raise ValueError('retain the first job binding for these Qwen traces')
    if not plan.exists():
        write_json(plan, value)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'waves', 'runs', 'out', 'lock'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('prepare stage trace artifacts on the login node without CUDA devices')
    finished = [(phase, plan, job) for phase, plan, job in completed(args.waves)
                if phase in ('qwen', 'qwenout')]
    if not finished:
        return
    inputs = inspect_inputs(INVENTORY)
    for phase, plan, job in finished:
        if not (args.runs/'optimization/decode-pairs'/job/'result.json').is_file():
            continue
        folder = args.out/plan['model']/f'B{plan["batch"]}-winners-{job}'
        if (folder/'overlap.rounds.json').exists():
            continue
        manifest = Path(plan['finalists'])
        if sha(manifest) != plan['finalists_sha256']:
            raise ValueError('the completed Qwen binding changed')
        binding = read(manifest)
        candidates = arms(binding, phase)
        names = {arm_name(row['rank']) for row in candidates}
        facts = [fact for fact, _, _ in collect(args.runs, job, inputs) if fact['arm'] in names]
        if facts:
            print(prepare(args.root, args.lock, folder, dict(binding, candidates=candidates), facts, job),
                  flush=True)


if __name__ == '__main__':
    main()
