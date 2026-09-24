#!/usr/bin/env python3
"""Audit Oracle fibers of the raw-log-selected GPU winner in each real arm.

The in-flight runners may predate winner-aware auditing. This independent
completion step selects from ten-process raw logs again, preserves their
original audit, and writes winner_oracle_audit.log plus command provenance.
Missing/failed arms do not prevent the remaining audits from running.
"""
import argparse
import hashlib
import json
import pathlib

from measure import ROOT, run
from report_tables import samples, table

HERE = pathlib.Path(__file__).resolve().parent


def audit(directory, seq, past, executable):
    choices = []
    for candidate in directory.glob('selected.cu.top*.cu.measurement'):
        _, _, times = samples(candidate)
        if times: choices.append((times[2], candidate))
    if not choices:
        print('WINNER_ORACLE pending_or_failed '+str(directory), flush=True)
        return False
    _, winner = min(choices, key=lambda x: x[0])
    entry = next(row for row in table(directory/'selected.cu.top3.tsv')
                 if pathlib.Path(row['source']).name == winner.name.removesuffix('.measurement'))
    cg = directory/pathlib.Path(entry['cg']).name
    command = [str(executable), str(cg), str(ROOT/'docs/experiments/COSTMODEL/event_fit/target.json'), str(seq), str(past)]
    log = directory/'winner_oracle_audit.log'
    metadata = log.with_suffix('.command.json')
    cg_digest=hashlib.sha256(cg.read_bytes()).hexdigest()
    executable_digest=hashlib.sha256(executable.read_bytes()).hexdigest()
    # Completed immutable candidates may be revisited after disconnection.
    if metadata.exists() and log.exists():
        previous = json.loads(metadata.read_text())
        if previous.get('command') == command and previous.get('exit_code') == 0 and previous.get('cg_sha256') == cg_digest and previous.get('executable_sha256') == executable_digest and 'ORACLE_SET_EQUAL' in log.read_text():
            print('WINNER_ORACLE existing '+str(cg), flush=True)
            return True
    status = run(command, log)
    provenance=json.loads(metadata.read_text());provenance['cg_sha256']=cg_digest
    metadata.write_text(json.dumps(provenance,indent=2)+'\n')
    print(f'WINNER_ORACLE exit={status} cg={cg}', flush=True)
    return status == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=pathlib.Path, default=HERE)
    parser.add_argument('--executable', type=pathlib.Path, default=ROOT/'build-portable/tools/tilemega-skeleton-audit')
    parser.add_argument('--arm-directory', type=pathlib.Path)
    parser.add_argument('--seq', type=int)
    parser.add_argument('--past', type=int, default=3)
    args = parser.parse_args()
    if args.arm_directory:
        if args.seq is None: parser.error('--arm-directory requires --seq')
        return int(not audit(args.arm_directory.resolve(), args.seq, args.past, args.executable))
    failures = []
    for model in ('llama', 'qwen3'):
        for seq in (1, 4, 16, 64):
            for k in ('4', '8', '16', 'W'):
                directory = args.evidence/'matrix'/f'{model}_s{seq}'/f'skeleton-k{k}'
                try:
                    if not audit(directory, seq, args.past, args.executable): failures.append(str(directory))
                except Exception as exc:
                    failures.append(str(directory))
                    print('WINNER_ORACLE error '+str(directory)+' '+repr(exc), flush=True)
    print(f'WINNER_ORACLE completed={32-len(failures)}/32', flush=True)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
