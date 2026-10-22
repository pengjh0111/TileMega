#!/usr/bin/env python3
"""Audit six native stage classes without counting a training fit as validation."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess

from sm90d1_common import ROOT, read, sha, write
from verify_sm90r2_correctness import verify_receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--probe', type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--ranking', type=Path)
    source.add_argument('--control', type=Path,
                        help='frozen stage-control identity and accepted C-1/C-2 dependency')
    p.add_argument('--ledger', type=Path, required=True)
    p.add_argument('--overhead', type=Path, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.ranking:
        ranking = read(a.ranking)
    else:
        control = read(a.control)
        if control['kind'] != 'sm90r2_stage_validation':
            raise ValueError('expected an explicit stage validation control')
        config = read(control['config'])
        proof = verify_receipt(Path(control['state']), control['job'],
                               Path(control['binary']), Path(control['config']))
        expected = ROOT / f"runs/sm90r2/build/{a.tag}/{config['model']}-B{config['batch']}/decode.so"
        if Path(control['binary']).resolve() != expected.resolve():
            raise ValueError('stage control names another artifact')
        ranking = dict(model=config['model'], batch=config['batch'],
            target_sha256=control['target_sha256'], probe_sha256=control['probe_sha256'],
            candidates=[dict(tag=a.tag, binary_sha256=proof['binary_sha256'], correctness=proof)])
    if sha(a.target) != ranking['target_sha256'] or sha(a.probe) != ranking['probe_sha256']:
        raise ValueError('class and request validation must use one frozen price model')
    model, batch = ranking['model'], ranking['batch']
    candidates = [r for r in ranking['candidates'] if r['tag'] == a.tag]
    if len(candidates) != 1 or not candidates[0]['correctness']['pass_']:
        raise ValueError('choose one ranking candidate with accepted correctness')
    binary = ROOT / f'runs/sm90r2/build/{a.tag}/{model}-B{batch}/decode.so'
    instrumented = ROOT / f'runs/sm90r2/build/{a.tag}-stage/{model}-B{batch}/decode.so'
    calibration = read(a.target)['sm90_dense_decode']
    for path, trace in ((binary, False), (instrumented, True)):
        identity = read(str(path) + '.identity.json')
        if sha(path) != identity['binary_sha256'] or bool(identity['trace']) != trace:
            raise ValueError('class observations require unchanged control and trace artifacts')
        if any(identity['source']['source_files'].get(k) != v
               for k, v in calibration['native_headers'].items()):
            raise ValueError('class observation differs from the calibrated native program')
    if sha(binary) != candidates[0]['binary_sha256'] or sha(str(binary) + '.cu') != sha(str(instrumented) + '.cu'):
        raise ValueError('stage-only trace must add only instrumentation to the ranked program')
    tag = a.tag + '-stage'
    def selected(path):
        return [r for r in csv.DictReader(path.open()) if r['model'] == model
                and int(r['batch']) == batch and r['tag'] == tag]
    ledger, overhead = selected(a.ledger), selected(a.overhead)
    if len(overhead) != 1 or not ledger:
        raise ValueError('one matched trace/control overhead observation is required')
    if len({r['kind'] for r in ledger}) != len(ledger):
        raise ValueError('duplicate stage classes are ambiguous')
    if any(abs(float(r['closure_max_fraction'])) > .03 for r in ledger):
        raise ValueError('stage ledger does not close')
    hf = read(ROOT / f'configs/serving/sm90d1/inputs/{model}/config.json')
    plan = read(str(binary) + '.plan.json')
    shapes = {g['family']: [g['n'], g['k']] for g in plan['gemms']}
    work = dict(shapes=[shapes[k] for k in ('qkv', 'o', 'gate_up', 'down', 'lm_head')],
        batch=batch, layers=hf['num_hidden_layers'],
        head_dim=hf.get('head_dim', hf['hidden_size'] // hf['num_attention_heads']),
        queries=hf['num_attention_heads'] // hf['num_key_value_heads'], qk_norm=hf['model_type'] == 'qwen3')
    a.out = a.out.resolve()
    if a.out.exists() or not a.out.is_relative_to(ROOT / 'runs/sm90r2'):
        raise ValueError('retain validation observations in a new runs/sm90r2 directory')
    a.out.mkdir(parents=True)
    wp, cp = a.out / 'workload.json', a.out / 'coordinates.json'
    write(wp, work); write(cp, plan['native_dense'])
    prediction = json.loads(subprocess.check_output([str(a.probe.resolve()), str(a.target.resolve()),
        str(wp), str(cp)], text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='')))
    write(a.out / 'predictions.json', prediction)
    # Boundary instrumentation has no per-class uninstrumented timestamps.
    # Uniform correction is an inference, recorded separately from raw totals.
    correction = float(overhead[0]['overhead_ms']) * 1e6 / (5 * work['layers'] + 3)
    if int(overhead[0]['past']) != 575:
        raise ValueError('the exported prediction is evaluated at past 575')
    rows = []
    for stage in prediction['predictions'][0]['past575']['stages']:
        if stage['kind'] not in ('qkv', 'attention', 'o', 'gate_up', 'down', 'lm_head'):
            continue
        source = next(r for r in ledger if r['kind'] == stage['kind'])
        count = float(source['instances_per_step'])
        if count != stage['instances']:
            raise ValueError('predicted and observed class instance counts differ')
        raw = float(source['class_total_ns'])
        observed = raw - correction * count
        if observed <= 0:
            raise ValueError('instrumentation correction exceeds the class signal')
        error = (stage['class_ns'] / observed - 1) * 100
        rows.append(dict(model=model, batch=batch, tag=a.tag, kind=stage['kind'],
            instances=count, raw_class_ns=raw, corrected_class_ns=observed,
            prediction_class_ns=stage['class_ns'], error_pct=error, class_pass=abs(error) <= 15,
            correction_method='inferred uniform boundary overhead'))
    with (a.out / 'classes.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    source_arg = 'ranking' if a.ranking else 'control'
    write(a.out / 'receipt.json', dict(model=model, batch=batch, tag=a.tag,
        inputs={k: dict(path=str(getattr(a, k)), sha256=sha(getattr(a, k)))
                for k in ('target', 'probe', source_arg, 'ledger', 'overhead')},
        binary_sha256=sha(binary), trace_sha256=sha(instrumented),
        class_pass=all(r['class_pass'] for r in rows), solver_gate_pass=False,
        trace_overhead_ms=float(overhead[0]['overhead_ms']),
        evidence='verified stage observations with inferred instrumentation correction; class check only'))
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
