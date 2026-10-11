"""Print queue states without touching the scheduler or the GPU."""
import argparse
from collections import Counter
import json
import time
from pathlib import Path


def read_status(out, queue_dir=None, now=None, queue_files=None):
    out = Path(out)
    now = time.time() if now is None else now
    path = out / 'state.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    paths = [Path(path) for path in queue_files] if queue_files is not None else (
        sorted(Path(queue_dir).glob('queue_*.json')) if queue_dir is not None else [])
    selected = set()
    for path in paths:
        for step in json.loads(path.read_text()):
            selected.add(step['name'])
            state.setdefault(step['name'], dict(status='queued', attempts=0))
    if queue_files is not None:
        state = {name: row for name, row in state.items() if name in selected}
    path = out / 'control.json'
    control = json.loads(path.read_text()) if path.exists() else {}
    rows = []
    for name, record in sorted(state.items()):
        cooldown = max(0, record.get('retry_at', 0) - now)
        note = 'cooldown {:.0f}s'.format(cooldown) if cooldown else ''
        if record['status'] == 'pending' and record.get('exit_code') == 75:
            note = 'GPU guard retry' + ('; ' + note if note else '')
        if record.get('pid') and record['status'] == 'running':
            note = 'pid={}'.format(record['pid'])
        rows.append(dict(name=name, status=record['status'],
                         attempts=record.get('attempts', 0), note=note))
    return dict(counts=dict(Counter(row['status'] for row in rows)),
                timing_paused=bool(control.get('timing_paused')),
                pause_reason=control.get('reason'), steps=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--queue-dir', type=Path)
    parser.add_argument('--queue-file', type=Path, action='append',
                        help='show only steps from these queue files; may be repeated')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    value = read_status(args.out, args.queue_dir, queue_files=args.queue_file)
    if args.json:
        print(json.dumps(value, indent=2))
        return
    print('  '.join('{}={}'.format(k, v) for k, v in sorted(value['counts'].items())) or 'No steps yet')
    if value['timing_paused']:
        print('TIMING PAUSED: ' + (value['pause_reason'] or 'see control.json'))
    print('{:<32} {:<12} {:>8}  {}'.format('NAME', 'STATUS', 'ATTEMPTS', 'NOTE'))
    for row in value['steps']:
        print('{:<32} {:<12} {:>8}  {}'.format(row['name'], row['status'], row['attempts'], row['note']))


if __name__ == '__main__':
    main()
