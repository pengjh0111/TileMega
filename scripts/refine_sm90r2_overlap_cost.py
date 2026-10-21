#!/usr/bin/env python3
"""Refine native overlap prices with paired request and stage observations.

All supplied rankings become training evidence. This never promotes a fit or
counts their post-fit ordering toward E-3.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess

import numpy as np
from scipy.optimize import least_squares
from sm90d1_common import ROOT, read, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage-fit', type=Path, required=True)
    p.add_argument('--ranking', type=Path, nargs='+', required=True)
    p.add_argument('--probe', type=Path, default=ROOT / 'build/sm90r2_price_probe')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out = a.out.resolve()
    if a.out.exists() or not a.out.is_relative_to(ROOT / 'runs/sm90r2'):
        raise ValueError('retain previous fits; choose a new diagnostic directory')
    target_path = a.stage_fit / 'target.json'
    target = read(target_path); cal = target['sm90_dense_decode']
    receipt = read(a.stage_fit / 'receipt.json')
    if sha(target_path) != receipt['target_sha256'] or cal['fit_status'] != 'initial_overlap_fitted':
        raise ValueError('an unchanged stage-only overlap fit is required')
    a.out.mkdir(parents=True)
    probe = a.out / 'price-probe'; shutil.copy2(a.probe.resolve(), probe)
    parameters = ('preissue_efficiency', 'tail_efficiency', 'issue_ns', 'issue_service_scale')
    old = {k: dict(v['coefficients']) for k, v in cal['body_fits'].items()}
    for fit in cal['body_fits'].values():
        for name in parameters: fit['coefficients'][name] = 0
    zero = a.out / 'zero-overlap.target.json'; write(zero, target)

    def predict(work, coordinates, stem, target_file=zero):
        wp, cp = a.out / (stem + '.workload.json'), a.out / (stem + '.coordinates.json')
        write(wp, work); write(cp, coordinates)
        data = json.loads(subprocess.check_output([str(probe), str(target_file), str(wp),
            str(cp), '--request-samples'], text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='')))
        for row in data['predictions']:
            total = statistics.mean(s['prediction']['step_ns'] for s in row['request_samples'])
            if abs(total-row['request']['step_ns']) > .05:
                raise ValueError('exported samples differ from request integration')
        write(a.out / (stem + '.predictions.json'), data)
        return data['predictions']

    def feature(s, weight):
        return dict(key=s['key'], weight=weight, limit=s['overlap_limit_ns'],
            early=s['preissue_service_ns'], late=s['late_service_ns'],
            count=s['issue_count'], service=s['issue_service_ns'])

    tags = [r['tag'] for r in receipt['artifacts']]
    work = read(a.stage_fit / 'workload.json'); coords = read(a.stage_fit / 'coordinates.json')
    stages = predict(work, coords, 'stage-input')
    rows = []
    for source in csv.DictReader((a.stage_fit / 'training.csv').open()):
        candidate, control = [stages[tags.index(source[k])]['past575']['stages']
                              for k in ('tag', 'baseline_tag')]
        ca = next(s for s in candidate if s['kind'] == source['kind'])
        co = next(s for s in control if s['kind'] == source['kind'])
        rows.append(dict(source='stage', label=source['tag'] + ':' + source['kind'],
            observed_ns=float(source['observed_delta_ns']), base_ns=ca['stage_ns']-co['stage_ns'],
            sigma_ns=1000., terms=[feature(ca, 1), feature(co, -1)],
            group='stage-input', candidate_index=tags.index(source['tag']),
            control_index=tags.index(source['baseline_tag']), kind=source['kind']))

    inputs = [('stage-input', work, coords)]
    for number, path in enumerate(a.ranking):
        ranking = read(path)
        if (ranking['retrospective'] or not ranking['pair_prefix'] or
                len(ranking['gpu_uuids']) != 1 or ranking['blank_pairs'] < 2):
            raise ValueError('request calibration requires a same-card paired cohort with blanks')
        cell = f"{ranking['model']}-B{ranking['batch']}"
        artifacts = ranking['candidates']
        if not all(r['protocol_complete'] and r['correctness']['pass_'] for r in artifacts):
            raise ValueError('incomplete request or correctness protocol')
        plans = []
        for row in artifacts:
            binary = ROOT / 'runs/sm90r2/build' / row['tag'] / cell / 'decode.so'
            identity = read(str(binary) + '.identity.json')
            if sha(binary) != row['binary_sha256'] or identity['trace'] or any(
                    identity['source']['source_files'].get(k) != v for k, v in cal['native_headers'].items()):
                raise ValueError('request and stage calibration use different native programs')
            proof = row['correctness']
            for name, digest in proof['receipt_sha256'].items():
                if sha(Path(proof['source_attempt']) / name) != digest:
                    raise ValueError('correctness evidence changed')
            for pair_path in row['paired_receipts']:
                pair = read(pair_path)
                guard = read(Path(pair_path).parents[1] / 'guard_result.json')
                if pair['status'] != 'completed' or guard['code'] or pair['baseline_tag'] != 'B15Z':
                    raise ValueError('request pair is incomplete or has a different control')
            plans.append(read(str(binary) + '.plan.json'))
        baseline = next(i for i, row in enumerate(artifacts) if row['tag'] == 'B15Z')
        hf = read(ROOT / f"configs/serving/sm90d1/inputs/{ranking['model']}/config.json")
        shape = {g['family']: [g['n'], g['k']] for g in plans[baseline]['gemms']}
        work = dict(shapes=[shape[k] for k in ('qkv', 'o', 'gate_up', 'down', 'lm_head')],
            batch=ranking['batch'], layers=hf['num_hidden_layers'],
            head_dim=hf.get('head_dim', hf['hidden_size']//hf['num_attention_heads']),
            queries=hf['num_attention_heads']//hf['num_key_value_heads'], qk_norm=hf['model_type']=='qwen3')
        coords = [plan['native_dense'] for plan in plans]
        stem = f'request-{number}'; predictions = predict(work, coords, stem)
        inputs.append((stem, work, coords))
        groups = {}
        for i, row in enumerate(artifacts): groups.setdefault(row['instruction_sha256'], []).append(i)
        for indices in groups.values():
            i = indices[0]
            if baseline in indices: continue
            values = [artifacts[j] for j in indices]
            measured = statistics.median(t for r in values for t in r['process_medians_ms'])
            control = statistics.median(r['baseline_ms'] for r in values)
            terms = []
            for index, sign in ((i, 1), (baseline, -1)):
                for sample in predictions[index]['request_samples']:
                    terms += [feature(s, sign*s['instances']/16) for s in sample['prediction']['stages']]
            rows.append(dict(source='request', label=cell + ':' + '/'.join(r['tag'] for r in values),
                observed_ns=(measured-control)*1e6,
                base_ns=predictions[i]['request']['step_ns']-predictions[baseline]['request']['step_ns'],
                sigma_ns=control*1e4*max(.5, ranking['blank_process_range_pct'] or 0,
                                        *(r['decision_threshold_pct'] for r in values)),
                terms=terms, group=stem, candidate_index=i, control_index=baseline, kind='request'))

    keys = sorted({t['key'] for r in rows for t in r['terms'] if t['early'] or t['late'] or t['count']})
    key_indices = {key: i for i, key in enumerate(keys)}
    prior = np.tile([.05, .05, .3, .1], len(keys))
    def deltas(x):
        result = []
        for row in rows:
            cost = row['base_ns']
            for t in row['terms']:
                if t['key'] not in key_indices: continue
                i = 4*key_indices[t['key']]
                cost += t['weight']*(t['count']*1000*x[i+2]+t['service']*x[i+3]-
                    min(t['limit'], t['early']*x[i]+t['late']*x[i+1]))
            result.append(cost)
        return np.array(result)
    observed = np.array([r['observed_ns'] for r in rows]); sigma = np.array([r['sigma_ns'] for r in rows])
    optimum = least_squares(lambda x: np.r_[(deltas(x)-observed)/sigma, .05*(x-prior)], prior,
        bounds=(np.zeros(len(prior)), np.tile([1, 1, 5, 2], len(keys))), loss='soft_l1')
    for i, key in enumerate(keys):
        fit = cal['body_fits'][key]
        fit['coefficients'].update(zip(parameters, [float(optimum.x[4*i]), float(optimum.x[4*i+1]),
                                    float(optimum.x[4*i+2]*1000), float(optimum.x[4*i+3])]))
        fit['overlap_evidence'] = dict(evidence='inferred',
            method='bounded joint stage/request-delta fit; supplied cohorts are training only',
            stage_rows=sum(r['source']=='stage' and any(t['key']==key for t in r['terms']) for r in rows),
            request_rows=sum(r['source']=='request' and any(t['key']==key for t in r['terms']) for r in rows))
    cal['fit_status'] = 'initial_overlap_fitted'
    cal['pending'] = ['fresh independent E-3 class/step/ranking validation after this refit',
                      'unmeasured issuer and L2-overlap classes', 'E-4 measured selection']
    output = a.out / 'target.json'; write(output, target)
    fitted = {stem: predict(work, coords, stem+'-fitted', output) for stem, work, coords in inputs}
    values = deltas(optimum.x)
    table = []
    for row, value in zip(rows, values):
        ca, co = [fitted[row['group']][row[k]] for k in ('candidate_index', 'control_index')]
        if row['source']=='request': native = ca['request']['step_ns']-co['request']['step_ns']
        else:
            native = (next(s['stage_ns'] for s in ca['past575']['stages'] if s['kind']==row['kind'])-
                      next(s['stage_ns'] for s in co['past575']['stages'] if s['kind']==row['kind']))
        if abs(native-value) > .05: raise ValueError('Python refinement and native prediction disagree')
        table.append(dict(source=row['source'], label=row['label'], observed_delta_ns=row['observed_ns'],
            base_delta_ns=row['base_ns'], predicted_delta_ns=float(value), residual_ns=float(value-row['observed_ns']),
            uncertainty_ns=row['sigma_ns'], evidence='training only'))
    with (a.out/'training.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(table)
    write(a.out/'receipt.json', dict(source_target_sha256=sha(target_path), target_sha256=sha(output),
        probe_sha256=sha(probe), script_sha256=sha(__file__), stage_receipt_sha256=sha(a.stage_fit/'receipt.json'),
        request_cohorts=[dict(path=str(p), sha256=sha(p)) for p in a.ranking], keys=keys,
        previously_unpriced_issuer_keys=[k for k in keys if not old[k]['issue_ns'] and not old[k]['issue_service_scale']],
        training_rows=len(rows), solver_gate_pass=False, evidence='inferred training fit; no independent validation'))
    print(json.dumps(dict(training_rows=len(rows), fitted_keys=keys, solver_gate_pass=False)))


if __name__ == '__main__': main()
