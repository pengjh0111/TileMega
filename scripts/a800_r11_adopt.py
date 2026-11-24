#!/usr/bin/env python3
"""Adopt a faster R11 arm after reconstructing its paired guarded measurement."""
import argparse
from pathlib import Path

from a100_baselines import inspect_inputs
from a800_baselines import INVENTORY
from a800_r10_adopt import timed_row
from a800_r11_collect_round import collect
from a800_r11_matrix import FROZEN_INPUTS, incumbent, write_results


def adopt(rows, submitted, inputs):
    selected = {}
    comparisons = {}
    for runs, job, requires_fresh in submitted:
        arms = list(collect(runs, job, inputs))
        if not arms:
            raise ValueError(f'no bound comparison in allocation {job}')
        cell = arms[0][0]['model'], arms[0][0]['batch']
        old = rows[cell]
        if cell in selected:
            # Additional phases must use the newest accepted incumbent.
            old = selected[cell][0]
        best = None
        validated = []
        for fact, observation, control in arms:
            if any(old[key] != digest for key, digest in control.items()):
                raise ValueError(f'{cell} was not compared against its current incumbent')
            if observation is None or requires_fresh and fact['passed_processes'] != 5:
                continue
            raw, artifact, provenance, sample = observation
            if any(old[key] != sample[key] for key in FROZEN_INPUTS):
                raise ValueError(f'{cell} changed the frozen checkpoint, prompt, or model configuration')
            validated.append(fact)
            if fact['status'] != 'faster' or best is not None and raw['tpot_ms'] >= best[0]['tpot_ms']:
                continue
            checks = fact['passed_processes']
            message = f'{checks}/{fact["processes"]} C1 and same-binary C2 checks, guarded paired timing'
            if fact['router_indices_match']:
                message += ', ordered routing matched'
            row = timed_row(old, artifact, raw, provenance, sample['gpu_uuid'], checks, message)
            best = row, dict(fact, prior_decode_sha256=old['decode_sha256'],
                             timing_sha256=raw['receipt'], correctness=message)
        if best:
            selected[cell] = best
        comparisons[cell] = (min(validated, key=lambda fact:fact['tpot_ms']) if validated else
                             dict(arms[0][0], status='no_eligible_trial', tpot_ms=''))
    return ({key: selected.get(key, (row, None))[0] for key, row in rows.items()},
            {key: fact for key, (_, fact) in selected.items()}, comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--incumbent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pair', action='append', required=True, nargs=3, metavar=('RUNS', 'JOB', 'FRESH'))
    args = parser.parse_args()
    rows = incumbent(args.incumbent)
    if any(not job.isdecimal() or fresh not in ('1', '5') for _, job, fresh in args.pair):
        parser.error('provide numerical Slurm job IDs and a required process count of 1 or 5')
    matrix, provenance, comparisons = adopt(
        rows, [(Path(run), int(job), fresh == '5') for run, job, fresh in args.pair], inspect_inputs(INVENTORY))
    print(write_results(args.out, rows, matrix, provenance, comparisons))


if __name__ == '__main__':
    main()
