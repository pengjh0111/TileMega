"""Append source-bound communication constants and final stage traces."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append(path, rows, keys):
    if not rows:
        return 0
    old = list(csv.DictReader(path.open())) if path.exists() else []
    seen = {tuple(row[key] for key in keys) for row in old}
    fresh = [row for row in rows if tuple(str(row[key]) for key in keys) not in seen]
    if fresh:
        with path.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            if not old:
                writer.writeheader()
            writer.writerows(fresh)
    return len(fresh)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    constants = args.root / 'constants.json'
    if constants.exists():
        data = json.loads(constants.read_text())
        rows = []
        for kind, values in data.items():
            for value in values:
                key = value['key']
                assert value['processes'] in (3, 5)
                assert len({sample['PID'] for sample in value['rounds']}) == value['processes']
                for sample in value['rounds']:
                    record = sample['input']
                    assert sha(Path(record['path'])) == record['sha256']
                rows.append(dict(record_type=kind, evidence='verified', model='qwen3_1_7b',
                    tp=key['tp'], batch=key['b'], stage=key['stage'], tn=key['tn'],
                    median_ns=value['median_ns'], p90_ns=value['p90_ns'], spread=value['spread'],
                    noisy=value['noisy'], fresh_processes=value['processes'],
                    solver_cost_ns=value['p90_ns'] if value['noisy'] else value['median_ns'],
                    contract='ordered_bf16_partial_sum', iterations=1000,
                    source_digest=value['source_digest'], binary_sha256=value['binary'],
                    rounds_json=json.dumps(value['rounds']), input=str(constants),
                    input_sha256=sha(constants), timing_eligible=False))
        assert len(rows) == 90
        print('constants appended', append(Path('docs/results/MX7_tp_constants_c550.csv'),
              rows, ['tp', 'batch', 'stage', 'tn', 'input_sha256']))
    rows = []
    for path in sorted((args.root / 'final_trace').glob('B*/trace/trace.json')):
        data = json.loads(path.read_text())
        assert data['tp'] == 4 and data['decode_samples'] == 1023
        for stage in data['stages']:
            rows.append(dict(record_type='selected_stage_trace', evidence='verified', tp=4,
                batch=data['batch'], layer=stage['layer'], stage=stage['stage'],
                decode_samples=stage['samples'], rank0_median_ns=stage['rank0_median_ns'],
                rank_median_ns_json=json.dumps(stage['rank_median_ns']),
                max_rank_difference_ns=stage['max_rank_difference_ns'],
                median_rank_difference_ns=stage['median_rank_difference_ns'],
                input=str(path), input_sha256=sha(path), timing_eligible=False))
    print('trace stages appended', append(Path('docs/results/MX7_tp_trace_c550.csv'),
          rows, ['batch', 'layer', 'stage', 'input_sha256']))


if __name__ == '__main__':
    main()
