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
    key_of = lambda row: tuple('' if row[key] is None else str(row[key]) for key in keys)
    seen = {key_of(row) for row in old}
    fresh = [row for row in rows if key_of(row) not in seen]
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
                    contract='fp32_argmax_lowest_global_index' if key['stage']=='head' else 'ordered_bf16_partial_sum', iterations=1000,
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
        capture_path = Path(data['capture']['path']) if 'capture' in data else None
        collector = capture_path.parent / 'collector.py' if capture_path else path.parent / 'collector.py'
        assert sha(collector) == data['collector']['sha256']
        if capture_path:
            assert sha(capture_path) == data['capture']['sha256']
            capture = json.loads(capture_path.read_text())
            assert capture['raw_traces'] == data['raw_traces'] and capture['execution_identity'] == data['execution_identity']
            for key in ('capture_guard', 'analyzer', 'analyzer_dependency'):
                assert sha(Path(data[key]['path'])) == data[key]['sha256']
            assert json.loads(Path(data['capture_guard']['path']).read_text())['code'] == 0
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
    rows = []
    for path in sorted((args.root / 'trace_resources').glob('B*/result.json')):
        guard = path.parent / 'guard/guard_result.json'
        if not guard.exists():
            continue
        assert json.loads(guard.read_text())['code'] == 0
        value = json.loads(path.read_text())
        assert value['qualified'] and len(value['ranks']) == 4
        assert all(r['resident_ctas'] == 1 and r['attributes']['localSizeBytes'] ==
                   r['attributes']['sharedSizeBytes'] == 0 for r in value['ranks'])
        batch = int(path.parent.name[1:])
        binary = args.root / f'trace_plans/tp4/B{batch}/plan.so'
        report_path = Path(str(binary) + '.resources.json')
        source_path = Path(str(binary) + '.source.json')
        report = json.loads(report_path.read_text())
        identity = json.loads(Path(str(binary) + '.identity.json').read_text())
        source = json.loads(source_path.read_text())
        kernel = report['kernel']
        assert not report['violations']
        ref = lambda p: json.dumps(dict(path=str(p), sha256=sha(p)))
        rows.append(dict(record_type='trace_actual_residency', evidence='verified',
            model='qwen3_1_7b', batch=batch, phase='decode', tp=4, threads=512,
            MTreg=kernel['registers'], STreg=kernel['scalar_registers'],
            stack_bytes=kernel['stack_bytes'], static_smem_bytes=kernel['static_smem_bytes'],
            dynamic_smem_bytes=identity['shared_memory_bytes'], resident_ctas=1,
            query_status='all_ranks_pass:' + ref(path), code_bytes=report['code_bytes'],
            source_head=source['head'], artifact=str(binary), artifact_sha256=sha(binary),
            resources_json=ref(report_path), source_snapshot_json=ref(source_path)))
    print('trace residency appended', append(Path('docs/results/MX7_resources_c550.csv'),
          rows, ['record_type', 'artifact']))


if __name__ == '__main__':
    main()
