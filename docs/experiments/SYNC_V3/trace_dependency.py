#!/usr/bin/env python3
"""Expose a retained next-slot dependency on the current publication at k=2."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RAW = HERE / 'c2_dependency'


def witnesses(dump):
    slots = list(csv.DictReader((dump / 'slots.tsv').open(), delimiter='\t'))
    waits = {int(r['wait_index']): r for r in csv.DictReader((dump / 'waits.tsv').open(), delimiter='\t')}
    events = {int(r['event_index']): r for r in csv.DictReader((dump / 'events.tsv').open(), delimiter='\t')}
    by_slot = {int(r['slot']): r for r in slots}
    rows = []
    for slot, consumer in by_slot.items():
        producer = by_slot.get(slot-1)
        if not producer or producer['worker'] != consumer['worker']:
            continue
        begin, count = int(consumer['wait_begin_idx']), int(consumer['wait_count'])
        for i in range(begin, begin+count):
            wait = waits[i]
            event = events[int(wait['event_index'])]
            if (wait['producer'] == producer['stage'] and
                int(wait['group']) == int(producer['logical_task']) // 2 and int(event['fanin']) > 1):
                rows.append(dict(worker=consumer['worker'], producer_slot=slot-1, consumer_slot=slot,
                                 producer_stage=producer['stage'], logical_task=producer['logical_task'],
                                 event_index=wait['event_index'], group=wait['group'], fanin=event['fanin']))
    return rows


def main():
    global RAW
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('run', 'verify'))
    parser.add_argument('--raw', type=Path, default=RAW)
    args = parser.parse_args()
    RAW = args.raw.resolve()
    all_rows = []
    for model in ('gqa2', 'mha4'):
        for seq in (4, 128):
            key = f'{model}_s{seq}'
            dump = RAW / 'trace' / key
            if args.phase == 'run':
                dump.mkdir(parents=True, exist_ok=True)
                cmd = [str(RAW / 'bin' / f'{model}_p0_full'),
                       str(REPO / f'docs/experiments/SEQSCAN/raw/fixture/{model}_s{seq}_p3')]
                env = dict(os.environ, TILEMEGA_MODEL_NAME=model, TILEMEGA_TRACE_V2='1',
                           TILEMEGA_TRACE_V2_OUT=str(dump))
                result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
                output = result.stdout+result.stderr
                (dump / 'run.log').write_text(output)
                (dump / 'run.json').write_text(json.dumps(dict(command=cmd, exit_code=result.returncode))+'\n')
                if result.returncode or 'RESULT status=PASS' not in output:
                    raise ValueError(f'{dump}: correctness failed')
            rows = witnesses(dump)
            if not rows:
                raise ValueError(f'{dump}: no retained adjacent-slot grouped dependency')
            for r in rows:
                r.update(model=model, seq=seq, dump=str(dump.relative_to(REPO)))
            all_rows += rows
            print(f'DEPENDENCY {key}: {len(rows)} adjacent-slot kappa=2 witnesses', flush=True)
    if args.phase == 'run':
        with (RAW / 'witnesses.tsv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0]), delimiter='\t', lineterminator='\n')
            writer.writeheader()
            writer.writerows(all_rows)


if __name__ == '__main__':
    main()
