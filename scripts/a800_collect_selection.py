#!/usr/bin/env python3
"""Join separately guarded A800 rank measurements for login-node finalization."""
import argparse
import json
import math
from pathlib import Path

from a100_baselines import sha
from a800_baselines import RUN, hardware_class
from a800_native import verify_request_tokens
from sm90r3_llm import selection_result
from tilemega.bench.scheduler import write_json


def read(path):
    return json.loads(path.read_text())


def collect(prepared, allocations):
    frozen = read(prepared)
    expected_hash = sha(prepared)
    ranks = {row['rank']: row for row in frozen['receipt']['candidates']
             if row['status'] == 'compiled'}
    if len(ranks) != len(allocations) or len({job for job, _ in allocations}) != len(ranks):
        raise ValueError('provide one distinct allocation for every compiled rank')
    arms, sources, receipts, linked = {}, {}, {}, {}
    workloads, devices = set(), set()
    for job, proof_path in allocations:
        if not job.isascii() or not job.isdecimal():
            raise ValueError('an allocation ID must be a numeric Slurm job ID')
        directory = RUN/'search'/job
        plan = read(directory/'planned.json')
        guard = read(directory/'guard_result.json')
        device = read(directory/'job/physical-device-identity.json')
        proof = read(proof_path)
        if (proof['schema'] != 1 or proof['status'] != 'ready_to_finalize'
                or proof['prepared_sha256'] != expected_hash or proof['scope'] != frozen['scope']
                or plan['prepared_shortlist_sha256'] != expected_hash
                or plan['target_sha256'] != frozen['scope']['target_sha256']
                or guard['code'] != 0 or guard['acceptance'] != 'strict'
                or not guard['ownership_verified'] or plan['device']['uuid'] != device['nvml_uuid']):
            raise ValueError(f'allocation {job} does not bind the guarded prepared shortlist')
        rebind = frozen.get('selector_rebind')
        if rebind and plan['select_sha256'] != rebind['selector_sha256']:
            raise ValueError(f'allocation {job} used another selector')
        rank = plan['measure_rank']
        label = f'c{rank}'
        stage = proof['receipt']
        selected = stage['candidates'][rank]
        measured = stage['measurement']
        occupancy = selected['occupancy']
        expected = ranks[rank]
        if (label in arms or stage['winner'] != rank or measured['winner_rank'] != rank
                or stage['ranked'] != frozen['receipt']['ranked']
                or any(selected[key] != expected[key] for key in
                       ('coordinates_sha256', 'binary_sha256', 'identity_sha256', 'manifest_sha256'))
                or occupancy['binary_sha256'] != expected['binary_sha256']
                or occupancy['identity_sha256'] != expected['identity_sha256']
                or occupancy['effective_residency'] < occupancy['requested_residency']
                or measured['winner_binary_sha256'] != expected['binary_sha256']
                or measured['gpu_uuid'] != device['nvml_uuid'] or not measured['split_rank_measurement']
                or set(measured['arms']) != {label} or measured['arms'][label]['processes'] != 1):
            raise ValueError(f'allocation {job} does not certify rank {rank}')
        sample = proof_path.parent/'measure'/f'00-{label}'/'measurements'/f'B{plan["batch"]}'/'measurements.json'
        verify_request_tokens(sample, plan['batch'], 3)
        value = read(sample)
        if (sha(sample) != measured['process_receipts'][label]
                or value['configuration']['decode_sha256'] != expected['binary_sha256']
                or not math.isclose(value['tpot_seconds']*1000, measured['arms'][label]['tpot_ms'])
                or value['checkpoint']['checkpoint_sha256'] != measured['checkpoint_sha256']):
            raise ValueError(f'allocation {job} has a different request receipt')
        workloads.add((plan['model'], plan['batch'], plan['prefill_sha256'], plan['select_sha256'],
                       value['model_config_sha256'], value['prompt_ids_sha256'], measured['checkpoint_sha256']))
        devices.add(hardware_class(plan['device']))
        arms[label] = measured['arms'][label]
        sources[label] = dict(job=job, gpu_uuid=device['nvml_uuid'], proof_sha256=sha(proof_path),
                              request_sha256=sha(sample), guard_sha256=sha(directory/'guard_result.json'))
        receipts[label] = stage
        linked[rank] = selected
    if len(workloads) != 1 or len(devices) != 1:
        raise ValueError('distributed P-SEL requires the same request inputs and hardware class')
    predicted = {f'c{rank}': row['predicted_ns'] for rank, row in ranks.items()}
    result = selection_result(arms, predicted)
    winner = result['winner']
    receipt = receipts[winner]
    for rank, selected in linked.items():
        receipt['candidates'][rank] = selected
    receipt['measurement'].update(result, arms=arms, predicted_ns=predicted, launch_order=[],
                                  split_rank_measurement=False, independent_allocations=sources,
                                  process_receipts={label: row['request_sha256'] for label, row in sources.items()})
    return dict(schema=1, status='ready_to_finalize', prepared_sha256=expected_hash,
                scope=frozen['scope'], receipt=receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--arm', nargs=2, action='append', metavar=('JOB', 'PROOF'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('retain the previous selection receipt')
    write_json(args.out, collect(args.prepared, [(job, Path(proof)) for job, proof in args.arm]))
