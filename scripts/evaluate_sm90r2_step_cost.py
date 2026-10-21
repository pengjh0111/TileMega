#!/usr/bin/env python3
"""Compare native request predictions with one explicitly selected paired cohort."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess

from sm90d1_common import ROOT, read, sha, write
from verify_sm90r2_correctness import verify_receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--pair-prefix', required=True)
    p.add_argument('--states', type=Path, nargs='+', required=True)
    p.add_argument('--probe', type=Path, default=ROOT / 'build/sm90r2_price_probe')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args();a.out = a.out.resolve()
    if a.out.exists() or not a.out.is_relative_to(ROOT / 'runs/sm90r2'):
        raise ValueError('choose a new evidence directory')
    calibration = read(a.target)['sm90_dense_decode']
    groups = {}
    for row in csv.DictReader(a.pairs.open()):
        if row['pair'].startswith(a.pair_prefix):
            groups.setdefault(row['pair'], []).append(row)
    if not groups:
        raise ValueError('the requested paired cohort is empty')
    a.out.mkdir(parents=True)
    states = [(path, read(path)) for path in a.states]
    results = []
    for pair, rows in sorted(groups.items()):
        hashes = {r['candidate_sha256'] for r in rows}
        devices = {r['uuid'] for r in rows}
        if len(hashes) != 1 or len(devices) != 1:
            raise ValueError('one cell must use one candidate and physical GPU')
        digest = hashes.pop();selected = None;processes = set();times = []
        for row in rows:
            run = Path(row['candidate_run'])
            current = read(run / 'job/selected-config.json')
            guard = run.parents[1] / 'guard_result.json'
            if sha(guard) != row['guard_sha256'] or read(guard)['code'] or row['paired_guard'] != 'True':
                raise ValueError('step validation requires accepted whole pairs')
            if selected and current['config_sha256'] != selected['config_sha256']:
                raise ValueError('paired cohort mixes workloads')
            selected = current
            identity = (str(run), row['candidate_pid'])
            if identity in processes:
                raise ValueError('duplicate fresh-process observation')
            processes.add(identity);times.append(float(row['candidate_ms']))
        config_path = Path(selected['config']);config = read(config_path)
        command = selected['command'];binary = Path(command[command.index('--decode') + 1])
        ident = read(str(binary) + '.identity.json')
        if sha(binary) != digest or ident['binary_sha256'] != digest or ident['trace']:
            raise ValueError('request prediction must bind an unmodified uninstrumented binary')
        if any(ident['source']['source_files'].get(k) != v for k,v in calibration['native_headers'].items()):
            raise ValueError('request prediction uses a different calibrated native implementation')
        correctness = None
        for state_path,state in states:
            for job,record in state.items():
                path = Path(record.get('out') or '/nonexistent') / 'job/selected-config.json'
                if record['status'] != 'done' or not path.exists():
                    continue
                check = read(path)
                if check['stage'] == 'correctness' and check['decode_sha256'] == digest:
                    correctness = verify_receipt(state_path, job, binary, config_path)
        if correctness is None:
            raise ValueError('request candidate has no accepted C-1/C-2 receipt')
        hf = read(ROOT / f"configs/serving/sm90d1/inputs/{config['model']}/config.json")
        plan = read(str(binary) + '.plan.json');shapes = {g['family']:[g['n'],g['k']] for g in plan['gemms']}
        work = dict(shapes=[shapes[k] for k in ('qkv','o','gate_up','down','lm_head')],
            batch=config['batch'],layers=hf['num_hidden_layers'],
            head_dim=hf.get('head_dim',hf['hidden_size']//hf['num_attention_heads']),
            queries=hf['num_attention_heads']//hf['num_key_value_heads'],qk_norm=hf['model_type']=='qwen3')
        stem = a.out / pair
        coordinates = Path(str(stem) + '.tmexec.json');workload = Path(str(stem) + '.workload.json')
        write(coordinates,plan['native_dense']);write(workload,work)
        output = json.loads(subprocess.check_output([str(a.probe.resolve()),str(a.target.resolve()),
            str(workload),str(coordinates)],text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='')))
        write(Path(str(stem)+'.prediction.json'),output)
        prediction = output['predictions'][0]['request']['step_ns']/1e6
        observed = statistics.median(times);error = (prediction/observed-1)*100
        complete = len(processes) >= (3 if config['model']=='llama3_1_8b' else 1)
        results.append(dict(model=config['model'],batch=config['batch'],pair=pair,
            prediction_ms=prediction,measured_ms=observed,error_pct=error,
            processes=len(processes),protocol_complete=complete,
            step_pass=complete and abs(error)<=10,
            process_range_pct=(max(times)-min(times))/observed*100,
            binary_sha256=digest,ownership_verified=all(r['ownership_verified']=='True' for r in rows)))
    with (a.out/'steps.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    write(a.out/'receipt.json',dict(target_sha256=sha(a.target),pairs_sha256=sha(a.pairs),
        probe_sha256=sha(a.probe),script_sha256=sha(__file__),pair_prefix=a.pair_prefix,
        cells=len(results),step_pass=all(r['step_pass'] for r in results),
        solver_gate_pass=False,evidence='request check only; class and ranking gates remain separate'))
    print(json.dumps(results))


if __name__=='__main__':main()
