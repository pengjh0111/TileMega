"""Append requalified physical-group units and selected-pin delay checks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from tilemega.tp.process_cohorts import checked, validate_run


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root):
    rows = []
    units = root / 'hardware_g1235/UNITS_QUALIFIED.json'
    if units.exists():
        for cohort in json.loads(units.read_text())['cohorts']:
            folder = checked(cohort['summary']).parent
            validate_run(cohort['run']['manifest'])
            files = sorted(folder.glob('process*/result.json'))
            data = [json.loads(path.read_text()) for path in files]
            assert len(data) == len({x['PID'] for x in data}) == 5
            assert all(x['passed'] and x['returncode'] == 0 and x['gpu_indices'] == [1, 2, 3, 5] for x in data)
            assert len({(x['source_sha256'], x['binary_sha256']) for x in data}) == 1
            first = data[0]
            assert sha(Path(first['source'])) == first['source_sha256']
            assert sha(Path(first['binary'])) == first['binary_sha256']
            rows.append(dict(record_type='physical_group_units', evidence='verified',
                test='tp_' + cohort['kind'], cases=first['records'][-1]['cases'], status='pass',
                fresh_processes=5, scope=f"TP4 B{cohort['batch']}; ordered BF16 reference; requalified physical group",
                gpu_indices='1,2,3,5', source_head=first['source_head'], test_sha256=first['source_sha256'],
                raw_log=str(cohort['summary']['path']), raw_log_sha256=cohort['summary']['sha256']))
    proofs = set(root.glob('litmus_g1235*/B*/QUALIFIED.json'))
    for proof in sorted(proofs):
        data = json.loads(proof.read_text())
        reference = checked(data['reference'])
        checked(data['ordinary_binary'])
        checked(data['litmus_binary'])
        assert json.loads(checked(data['occupancy']).read_text())['qualified']
        checks = []
        for item in data['runs']:
            guard = validate_run(item['run']['manifest'])
            check = json.loads(checked(item['check']).read_text())
            assert check['PID'] == guard['attempts'][-1]['child_pid']
            assert check['pass'] and check['rank_tokens_identical'] and check['steps'] == 1024
            assert check['gpu_indices'] == [1, 2, 3, 5]
            assert sha(checked(item['check']).parent / 'tokens.json') == sha(reference)
            checks.append(check)
        assert len(checks) == len({x['PID'] for x in checks}) == 5
        rows.append(dict(record_type='selected_physical_group_litmus', evidence='verified',
            test='tp_injected_delay', cases=data['batch'] * 1024, status='pass', fresh_processes=5,
            scope=f"TP4 B{data['batch']}; selected pin; exact ordinary tokens; requalified physical group",
            gpu_indices='1,2,3,5', source_head=checks[0]['host_source_identity']['head'],
            test_sha256=checks[0]['host_source_identity']['source_files']['python/tilemega/serving/check_tp.py'],
            raw_log=str(proof), raw_log_sha256=sha(proof)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    path = Path('docs/results/MX7_tp_correctness_c550.csv')
    old = list(csv.DictReader(path.open()))
    seen = {row['raw_log'] for row in old}
    fresh = [row for row in collect(args.root) if row['raw_log'] not in seen]
    if fresh:
        with path.open('a', newline='') as stream:
            csv.DictWriter(stream, fieldnames=list(old[0]), lineterminator='\n').writerows(fresh)
    print('requalified hardware cohorts appended', len(fresh))
    resource_path = Path('docs/results/MX7_resources_c550.csv')
    old_resources = list(csv.DictReader(resource_path.open()))
    seen_resources = {(x['record_type'], x['artifact_sha256']) for x in old_resources}
    resources = []
    for marker in sorted(args.root.glob('litmus_g1235*/B*/occupancy_GUARD_PASSED.json')):
        run = json.loads(marker.read_text())
        validate_run(run['manifest'])
        query_path = Path(run['output'])
        query = json.loads(query_path.read_text())
        spec = json.loads(checked(query['input_manifest']).read_text())
        binary = checked(spec['binary'])
        actual = json.loads(checked(spec['resources']).read_text())
        identity = json.loads(checked(spec['context']).read_text())
        source = json.loads(checked(spec['source_snapshot']).read_text())
        assert query['qualified'] and len(query['ranks']) == 4
        for rank in query['ranks']:
            assert rank['resident_ctas'] == 1 and rank['binary_sha256'] == sha(binary)
            assert rank['attributes']['localSizeBytes'] == rank['attributes']['sharedSizeBytes'] == 0
        kernel = actual['kernel']
        assert 0 < kernel['registers'] <= 255 and kernel['stack_bytes'] == kernel['static_smem_bytes'] == 0
        kind = 'diagnostic_litmus_actual_residency'
        key = (kind, sha(binary))
        if key in seen_resources:
            continue
        seen_resources.add(key)
        resources.append(dict(record_type=kind, evidence='verified', model='qwen3_1_7b',
            batch=marker.parent.name[1:], phase='decode', tp=4, threads=512,
            MTreg=kernel['registers'], STreg=kernel['scalar_registers'], stack_bytes=0,
            static_smem_bytes=0, dynamic_smem_bytes=identity['shared_memory_bytes'],
            resident_ctas=1, query_status='all_ranks_pass:' + str(query_path),
            code_bytes=actual['code_bytes'], source_head=source['head'], artifact=str(binary),
            artifact_sha256=sha(binary), resources_json=json.dumps(spec['resources']),
            source_snapshot_json=json.dumps(spec['source_snapshot'])))
    if resources:
        with resource_path.open('a', newline='') as stream:
            csv.DictWriter(stream, fieldnames=list(old_resources[0]), lineterminator='\n').writerows(resources)
    print('diagnostic litmus residency appended', len(resources))


if __name__ == '__main__':
    main()
