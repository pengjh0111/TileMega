"""Estimate reserved child-job time, excluding time waiting for GPU locks."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/mx7'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    jobs, hours, seen = [], {}, set()
    for path in args.root.rglob('guard_result.json'):
        path = path.resolve()
        pgid = path.parent / 'pgid'
        if path in seen or not pgid.exists():
            continue
        seen.add(path)
        data = json.loads(path.read_text())
        cards = data.get('gpu_indices')
        if cards is None:
            continue
        seconds = (path.stat().st_mtime_ns - pgid.stat().st_mtime_ns) / 1e9
        if seconds < 0:
            raise ValueError('guard timestamps reversed: ' + str(path))
        jobs.append(dict(directory=str(path.parent), seconds=seconds,
                         cards=list(map(int, cards)), code=data['code']))
        for card in cards:
            hours[str(card)] = hours.get(str(card), 0) + seconds / 3600
    result = dict(evidence='inferred', method=__doc__,
                  caveat='pgid-file to guard-result-file mtime; includes child initialization/cleanup; excludes lock waiting and preflight; not hardware busy time',
                  card_hours=hours, sum_card_hours=sum(hours.values()), jobs=jobs)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print('reserved card-hours', result['sum_card_hours'], hours)


if __name__ == '__main__':
    main()
