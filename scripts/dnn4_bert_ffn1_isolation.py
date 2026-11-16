"""Validate the FFN1-only TMA192 successor on the current B8 program."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(folder, decision, final):
    from dnn4_bert_current import validate_ffn1_tma192_coordinates, validate_timing_gpu
    from dnn4_bert_model_cohort import validate_bert_checks
    from dnn4_bert_prefetch import canonical_source
    from tilemega.build.feature_contract import build_defines
    from tilemega.build.identity import verify

    integration = read(folder/'integration.json')
    keys = ('preparation', 'original_plan', 'components', 'default_proof',
            'defaults', 'model_cohort', 'ledger_receipt', 'trace_source_proof')
    required = {integration[k] for k in keys} | set(decision['unit_gates'])
    if (not required <= integration['receipts'].keys()
            or any(digest(p) != sha for p, sha in integration['receipts'].items())):
        raise ValueError('FFN1 integration has a missing or changed receipt')
    pair = decision['pair']
    if (decision['batch'] != 8 or not decision['accepted']
            or decision['minimum_gain'] != .03
            or pair['ratio'] >= 1-max(.03, pair['threshold'])
            or final['library'] != decision['library'] or final['gate'] != decision['gate']
            or final['pair'] != pair or final['unit_gates'] != decision['unit_gates']):
        raise ValueError('FFN1 lacks its predeclared end-to-end gain')
    prepared = read(integration['preparation'])
    plan = read(integration['original_plan'])
    reference = verify(Path(decision['reference']))
    identity = verify(Path(decision['library']))
    source = Path(decision['library']+'.cu').read_text()
    profile = dict(json.loads(plan['reference']['profile']), TILEMEGA_F3_REGISTER_BUDGET='3')
    if (prepared['status'] != 'prepared_CPU_only'
            or prepared['model']['library'] != decision['library']
            or prepared['model']['artifact_id'] != identity['artifact_id']
            or plan['minimum_gain'] != .03 or plan['profile'] != profile
            or prepared['profile'] != profile
            or plan['reference']['library'] != decision['reference']
            or plan['reference']['gate'] != decision['reference_gate']
            or identity['trace'] or identity['plan']['runtime_batch'] != 8
            or identity['plan']['buffers'] != reference['plan']['buffers']
            or any(r['spill'] for r in identity['resources'].values())
            or build_defines(source, []) != dict(
                build_defines(Path(decision['reference']+'.cu').read_text(), []),
                TILEMEGA_F3_REGISTER_BUDGET='3')):
        raise ValueError('FFN1 changes its prepared artifact or feature scope')
    validate_ffn1_tma192_coordinates(decision['library'], decision['reference'])
    defaults = read(integration['defaults'])
    default_proof = read(integration['default_proof'])
    if (not default_proof['passed'] or default_proof['proof'] != integration['defaults']
            or not defaults['passed'] or defaults['batch'] != 8
            or defaults['reference'] != decision['reference']
            or defaults['reference_sha256'] != digest(decision['reference'])
            or defaults['encoded_words'] != 33712
            or not defaults['selected_default_explicit_source_identical']
            or canonical_source(Path(defaults['new_library']+'.cu').read_text()) !=
               canonical_source(Path(decision['reference']+'.cu').read_text())):
        raise ValueError('FFN1 drops its unchanged current-main control')

    components = read(integration['components'])
    gates = decision['unit_gates']
    if (not components['passed'] or len(gates) != 5 or len(set(gates)) != 5
            or [r['name'] for r in components['rows']] !=
               ['body', 'order0', 'order1', 'order2', 'order3']
            or [r['reference_gate'] for r in components['rows']] != gates):
        raise ValueError('FFN1 lacks its five exact component reuse links')
    for row, words in zip(components['rows'], [26880, 6640, 9072, 8384, 10880]):
        flags = dict(profile)
        if row['name'] == 'body':
            # This flag selects FFN1 fixtures; production has no 96-column stats.
            flags['TILEMEGA_F3_FFN2_STATS96'] = '1'
        if (not row['passed'] or row['encoded_words'] != words
                or row['reference_words'] != words or row['profile'] != profile
                or row['fixture_defines'] != flags
                or digest(row['binary']) != row['binary_sha256']
                or digest(row['reference']) != row['reference_sha256']
                or '-I/root/TileMega-dnn3/include' not in row['command']
                or {f'-D{k}={v}' for k, v in flags.items()} !=
                   {v for v in row['command'] if v.startswith('-DTILEMEGA_F3_')}):
            raise ValueError('FFN1 reused a different component or register profile')
        gate = read(row['reference_gate'])
        processes = gate['processes']
        if (not gate['passed'] or gate['gpu'] != 2 or len(processes) != 17
                or len({p['pid'] for p in processes}) != 17
                or len({p['output_hash'] for p in processes}) != 1
                or Counter(p['name'].split('-')[0] for p in processes) !=
                   Counter(unit=5, memcheck=1, initcheck=1, racecheck=5, synccheck=5)
                or any(p['exit_code'] or read(p['guard'])['code']
                       or p['binary_sha256'] != row['reference_sha256']
                       or digest(p['log']) != p['sha256'] for p in processes)):
            raise ValueError('FFN1 historical component cohort is incomplete')

    model = read(integration['model_cohort'])
    checks = [read(p['check']) for p in model['processes']]
    if (not model['passed'] or model['passes'] != '5/5' or model['gpu'] != 1
            or model['execution']['artifact_id'] != identity['artifact_id']
            or model['gate'] != decision['gate'] or len(model['processes']) != 5
            or len({p['pid'] for p in model['processes']}) != 5
            or any(p['exit_code'] or read(p['guard'])['code'] for p in model['processes'])):
        raise ValueError('FFN1 lacks its strict five-process model cohort')
    validate_bert_checks(checks, identity['artifact_id'])
    old_check = read(decision['reference_gate'])
    for key in ('reference', 'synthetic_weights', 'batch', 'checker_sha256',
                'bridge_sha256', 'archive_sha256', 'source_checkpoint_sha256',
                'input_tensors_sha256'):
        if checks[0][key] != old_check[key]:
            raise ValueError('FFN1 changed its reference inputs, weights or checker')
    for process in model['processes']:
        records = [json.loads(v) for v in
                   (Path(process['guard']).parent/'guard.jsonl').read_text().splitlines()]
        if not records or any(r['gpu_index'] != 1 for r in records if 'gpu_index' in r):
            raise ValueError('FFN1 model used a different physical card')

    allocation = read(folder/'allocation.json')
    authorization = read(allocation['snapshot'])
    if (allocation['gpu'] != 1 or allocation['original_anchor_gpu'] != 3
            or allocation['reanchored'] or allocation['python_entry_root'] != '/root/python'
            or authorization['allowed_gpu_indices'] != [1]
            or authorization['python_entry_root'] != '/root/python'
            or digest(allocation['snapshot']) != allocation['snapshot_sha256']):
        raise ValueError('FFN1 changes its historical allocation or fixed target')
    pids = []
    reference_measurement = read(pair['measurements'][0]['measurement'])
    for measured, order in ((pair, 'ABABAB'), (final['measurement'], 'BBB')):
        validate_timing_gpu(measured, 1, order)
        for row in measured['measurements']:
            raw = read(row['measurement'])
            candidate = row['arm'] == 'B'
            expected = identity if candidate else reference
            gate = decision['gate'] if candidate else decision['reference_gate']
            if (raw['artifact_id'] != expected['artifact_id']
                    or Path(raw['library_path']).resolve() != Path(expected['binary_path']).resolve()
                    or Path(raw['numerical_gate_path']).resolve() != Path(gate).resolve()
                    or raw['batch'] != 8 or raw['precision'] != 'bf16'
                    or any(raw[k] != reference_measurement[k]
                           for k in ('random_weights', 'accuracy_eligible'))
                    or raw['source_checkpoint_sha256'] != old_check['source_checkpoint_sha256']
                    or raw['input_sha256'] != pair['input_sha256']
                    or row['median_ms'] != statistics.median(raw['samples_ms'])):
                raise ValueError('FFN1 substitutes its timing artifact or input')
            boundaries = [json.loads(v) for v in Path(raw['guard_log']).read_text().splitlines()]
            for label, clock in [('before-measurement', 'start_sm_clock_mhz'),
                                 ('after-measurement', 'end_sm_clock_mhz')]:
                boundary = next(v for v in boundaries if v.get('label') == label)
                if (boundary['time'] < authorization['current_gpu_restriction_at']
                        or float(boundary['telemetry']['clocks.sm']) != row[clock]):
                    raise ValueError('FFN1 lacks its actual authorized clock boundaries')
            pids.append(raw['pid'])
    a = [r['median_ms'] for r in pair['measurements'] if r['arm'] == 'A']
    b = [r['median_ms'] for r in pair['measurements'] if r['arm'] == 'B']
    noise = max(.005, 2*max((max(v)-min(v))/statistics.median(v) for v in (a, b)))
    values = [r['median_ms'] for r in final['measurement']['measurements']]
    events = sorted(v for r in final['measurement']['measurements']
                    for v in read(r['measurement'])['samples_ms'])
    if (len(set(pids)) != 9 or pair['candidate_ms'] != statistics.median(b)
            or pair['reference_ms'] != statistics.median(a)
            or pair['ratio'] != statistics.median(b)/statistics.median(a)
            or pair['threshold'] != noise
            or final['measurement']['candidate_ms'] != statistics.median(values)
            or final['measurement']['process_range_ms'] != max(values)-min(values)
            or final['measurement']['p90_ms'] != events[(9*len(events)+9)//10-1]):
        raise ValueError('FFN1 changes the independent measured statistics')
    trace = read(integration['trace_source_proof'])
    ledger = read(integration['ledger_receipt'])
    traced = verify(Path(trace['library']))
    stages = list(csv.DictReader(Path(final['ledger']).open()))
    if (not trace['passed'] or ledger['gpu'] != 1 or ledger['trace_level'] != 1
            or not traced['trace'] or traced['artifact_id'] != ledger['artifact_id']
            or ledger['reference'] != decision['library']
            or ledger['reference_sha256'] != digest(decision['library'])
            or ledger['ledger'] != final['ledger']
            or sorted(int(r['stage']) for r in stages) != list(range(63))
            or trace['reference'] != decision['library']
            or Path(trace['library']+'.cu').read_text().replace(
                '#define TILEMEGA_TRACE_STAGE 1', '#define TILEMEGA_TRACE_STAGE 0') != source):
        raise ValueError('FFN1 lacks its exact production-binary ledger')
    return 1
