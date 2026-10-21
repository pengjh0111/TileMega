#!/usr/bin/env python3
"""Audit one native solver ranking cohort against accepted paired measurements."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

from scipy.stats import kendalltau
from sm90d1_common import ROOT, read, sha, write
from verify_sm90r2_correctness import verify_receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--candidate', type=Path, nargs='+', required=True)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--states', type=Path, nargs='+', required=True)
    p.add_argument('--probe', type=Path, default=ROOT / 'build/sm90r2_price_probe')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out = a.out.resolve()
    if a.out.exists() or not a.out.is_relative_to(ROOT / 'runs/sm90r2'):
        raise ValueError('choose a new evidence directory under runs/sm90r2')
    a.out.mkdir(parents=True)
    config = read(a.config)
    hf = read(ROOT / f"configs/serving/sm90d1/inputs/{config['model']}/config.json")
    calibration = read(a.target)['sm90_dense_decode']
    pairs = list(csv.DictReader(a.pairs.open()))
    states = [(path, read(path)) for path in a.states]
    rows, seen, devices, missing = [], set(), set(), []
    for binary in a.candidate:
        binary = binary.resolve()
        identity = read(str(binary) + '.identity.json')
        digest = sha(binary)
        if digest in seen:
            raise ValueError('duplicate candidate artifact')
        seen.add(digest)
        if digest != identity['binary_sha256'] or identity['trace']:
            raise ValueError('ranking needs unchanged uninstrumented artifacts')
        source = identity['source']['source_files']
        if any(source.get(k) != v for k, v in calibration['native_headers'].items()):
            raise ValueError('candidate differs from the calibrated native program')
        observed = [r for r in pairs if r['candidate_sha256'] == digest]
        if not observed:
            missing.append(str(binary))
            continue
        required = 3 if config['model'] == 'llama3_1_8b' else 1
        receipts, process_ids, timings = [], set(), []
        for r in observed:
            selected = read(Path(r['candidate_run']) / 'job/selected-config.json')
            if selected['config_sha256'] != sha(a.config):
                raise ValueError('ranking measurements mix workloads')
            guard_path = Path(r['candidate_run']).parents[1] / 'guard_result.json'
            if (sha(guard_path) != r['guard_sha256'] or read(guard_path)['code']
                    or r['paired_guard'] != 'True'):
                raise ValueError('ranking row lacks an accepted complete pair')
            receipt = str(Path(r['candidate_run']).parent / 'paired.json')
            if receipt in receipts:
                raise ValueError('duplicate paired measurement')
            receipts.append(receipt)
            process_ids.add((receipt, r['candidate_pid']))
            timings.append(float(r['candidate_ms']))
            devices.add(r['uuid'])
        correctness = None
        for state_path, state in states:
            for job, record in state.items():
                selected_path = Path(record.get('out', '/nonexistent')) / 'job/selected-config.json'
                if record['status'] != 'done' or not selected_path.exists():
                    continue
                selected = read(selected_path)
                if selected['stage'] == 'correctness' and selected['decode_sha256'] == digest:
                    correctness = verify_receipt(state_path, job, binary, a.config)
        if correctness is None:
            raise ValueError('candidate has no accepted full-length correctness receipt')
        plan = read(str(binary) + '.plan.json')
        shapes = {g['family']: [g['n'], g['k']] for g in plan['gemms']}
        work = dict(shapes=[shapes[k] for k in ('qkv', 'o', 'gate_up', 'down', 'lm_head')],
            batch=config['batch'], layers=hf['num_hidden_layers'],
            head_dim=hf.get('head_dim', hf['hidden_size'] // hf['num_attention_heads']),
            queries=hf['num_attention_heads'] // hf['num_key_value_heads'],
            qk_norm=hf['model_type'] == 'qwen3')
        stem = a.out / digest[:16]
        coordinates = Path(str(stem) + '.tmexec.json')
        workload = Path(str(stem) + '.workload.json')
        write(coordinates, plan['native_dense']); write(workload, work)
        command = [str(a.probe.resolve()), str(a.target.resolve()), str(workload), str(coordinates)]
        prediction = json.loads(subprocess.check_output(command, text=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='')))
        write(Path(str(stem) + '.prediction.json'), prediction)
        predicted_ms = prediction['predictions'][0]['request']['step_ns'] / 1e6
        measured_ms = statistics.median(timings)
        rows.append(dict(tag=binary.parents[1].name, binary_sha256=digest,
            predicted_ms=predicted_ms, measured_ms=measured_ms,
            step_error_pct=(predicted_ms / measured_ms - 1) * 100,
            processes=len(process_ids), protocol_complete=len(process_ids) >= required,
            process_range_pct=(max(timings)-min(timings))/measured_ms*100,
            ownership_verified=all(r['ownership_verified'] == 'True' for r in observed),
            correctness=correctness, paired_receipts=receipts))
    if len(devices) > 1:
        raise ValueError('ranking cohort must use one physical GPU')
    complete = len(rows) >= 8 and not missing and all(r['protocol_complete'] for r in rows)
    tau = float(kendalltau([r['predicted_ms'] for r in rows],
                          [r['measured_ms'] for r in rows]).statistic) if len(rows) > 1 else None
    if tau is not None and not math.isfinite(tau):
        tau = None
    top3 = sorted(rows, key=lambda r: r['predicted_ms'])[:3]
    fastest = min(rows, key=lambda r: r['measured_ms']) if rows else None
    top3_contains_fastest = fastest in top3 if fastest else False
    # This receipt never promotes a fit: class errors and the other six cells
    # remain separate E-3 requirements, even when this cohort ranks correctly.
    result = dict(model=config['model'], batch=config['batch'], candidates=rows, missing=missing,
        complete=complete, kendall_tau=tau, top3_contains_fastest=top3_contains_fastest,
        ranking_pass=complete and tau is not None and tau >= .7 and top3_contains_fastest,
        class_validation='not evaluated', solver_gate_pass=False,
        target_sha256=sha(a.target), pairs_sha256=sha(a.pairs),
        probe_sha256=sha(a.probe), script_sha256=sha(__file__), gpu_uuids=sorted(devices))
    write(a.out / 'ranking.json', result)
    print({k: result[k] for k in ('model', 'batch', 'complete', 'kendall_tau',
        'top3_contains_fastest', 'ranking_pass')})


if __name__ == '__main__':
    main()
