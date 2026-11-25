#!/usr/bin/env python3
"""Read R11 pairs and verify timing, numeric, and frozen input evidence."""
import argparse
import math
from pathlib import Path

from a100_baselines import ROOT, sha, inspect_inputs
from a100_results import guarded_measurement, write_csv
from a800_baselines import INVENTORY, hardware_class
from a800_r11_gate import GATE, INVARIANTS, read, verify
from a800_results import native_provenance


TARGET = ROOT/'configs/targets/sm_80_a800_sxm4_80gb_registry.json'


def timing(runs, job, model, batch, sample, artifact, inputs):
    round_id = sample['round']
    history = runs/'native'/sample['gpu_uuid']/model/f'B{batch}'
    [path] = [path for path in history.glob(f'*/tune-decode-{round_id}.json')
              if read(path)['decode_sha256'] == artifact['decode_sha256']]
    receipt = read(path)
    evidence = Path(receipt['evidence'])
    if (receipt['pass'] is not True or receipt['job'] != str(job)
            or receipt['device_uuid'] != sample['gpu_uuid']
            or any(receipt[key] != digest for key, digest in artifact.items())
            or receipt['tokens_sha256'] != sample['tokens_sha256']
            or sha(evidence/'reference_tokens.json') != receipt['tokens_sha256']
            or hardware_class(receipt['device']) != hardware_class(inputs['device'])):
        raise ValueError('timing receipt differs from its guard or own token evidence')
    raw = guarded_measurement(dict(name=f'{model} B{batch} {sample["arm"]}',
        model=model, batch=batch, round=round_id), evidence/f'B{batch}/measurements.json',
        evidence.parent.parent, dict(models=inputs['models'], device=receipt['device']))
    if (sample['measurement_sha256'] != raw['receipt']
            or sample['checkpoint_sha256'] != raw['checkpoint_sha256']
            or not math.isclose(sample['tpot_seconds'], raw['tpot_ms']/1000, rel_tol=1e-7)):
        raise ValueError('timing summary differs from the guarded process')
    return raw, native_provenance(raw, receipt)


def collect(runs, job, models):
    inputs = dict(device=read(TARGET)['device_provenance'], models=models)
    folder = runs/'optimization/decode-pairs'/str(job)
    binding = read(folder/'planned.json')
    if (binding['protocol_version'], binding['numeric_gate'], binding['phase']) != ('a800-single-v2', GATE, 'decode'):
        raise ValueError('the allocation does not use R11 numeric admission and paired timing')
    model, batch = binding['model'], binding['batch']
    samples = {sample['arm']: sample for sample in read(folder/'rounds.json')}
    checks = read(folder/'numeric-gate.json')
    comparisons = read(folder/'result.json')
    control = samples['control']
    control_id = dict(decode_sha256=binding['variant_sha256']['control'],
                      prefill_sha256=binding['shared_sha256'], target_sha256=binding['target_sha256'])
    control_raw, _ = timing(runs, job, model, batch, control, control_id, inputs)

    def verify_checks(label, identity):
        proof = checks[label]
        paths = proof['proofs']
        if proof['passed'] != len(paths):
            raise ValueError('numeric proof count changed')
        observations = []
        for entry in paths:
            if sha(entry['path']) != entry['sha256']:
                raise ValueError('numeric proof changed')
            observations.append(verify(entry['path'], job=job, model=model, batch=batch, identity=identity))
        if len({row['pid'] for row in observations}) != len(observations):
            raise ValueError('numeric proof must use fresh processes')
        return observations

    def bound_routes(proof):
        row = proof['receipt']
        return row['router_indices_sha256'], row['routing_trajectory_sha256'], row['routing_export_sha256']

    paired_routes = None
    if 'control' in checks:
        [proof] = verify_checks('control', control_id)
        paired_routes = bound_routes(proof)
    for label in binding['order'][1:]:
        artifact = dict(decode_sha256=binding['variant_sha256'][label],
                        prefill_sha256=binding['shared_sha256'], target_sha256=binding['targets_sha256'][label])
        count = binding['correctness_processes']
        proofs = verify_checks(label, artifact)
        correct = len(proofs) == count
        sample = samples.get(label)
        routes_match = None
        if label in binding.get('routing_arms', []):
            routes_match = bool(proofs) and paired_routes is not None and all(
                bound_routes(row) == paired_routes for row in proofs)
        passed = correct and routes_match is not False and label in comparisons['arms']
        output = dict(model=model, batch=batch, arm=label, job=str(job), pass_=passed,
                      hf_pass=correct, executor_agree=correct, router_indices_match=routes_match,
                      processes=count, passed_processes=len(proofs), pass_rate=f'{len(proofs)}/{count}',
                      gpu_uuid=control['gpu_uuid'],
                      tpot_ms='', control_tpot_ms=control_raw['tpot_ms'], improvement_pct='',
                      status='rejected', decode_sha256=artifact['decode_sha256'], evidence=sha(folder/'result.json'))
        output['pass'] = output.pop('pass_')
        if not passed:
            yield output, None, control_id
            continue
        if (sample is None or any(sample[key] != control[key] for key in INVARIANTS)
                or any(any(proof[key] != sample[key] for key in INVARIANTS)
                       or proof['receipt']['tokens_sha256'] != sample['tokens_sha256'] for proof in proofs)):
            raise ValueError('paired GPU or input identity changed, or an arm changed its own tokens')
        raw, provenance = timing(runs, job, model, batch, sample, artifact, inputs)
        change = 1-raw['tpot_ms']/control_raw['tpot_ms']
        status = 'faster' if change > .005 else 'slower' if change < -.005 else 'tie'
        output.update(status=status, tpot_ms=raw['tpot_ms'], improvement_pct=100*change)
        yield output, (raw, artifact, provenance, sample), control_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', action='append', required=True, type=int)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    inputs = inspect_inputs(INVENTORY)
    rows = [report for job in args.job for report, _, _ in collect(args.runs, job, inputs)]
    write_csv(args.out, rows)


if __name__ == '__main__':
    main()
