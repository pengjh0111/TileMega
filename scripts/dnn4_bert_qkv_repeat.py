"""Validate the repeated-QKV TMA successor while preserving first QKV."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def effective(profile):
    result = {k: v for k, v in profile.items()
              if k not in ('TILEMEGA_F3_REPEAT', 'TILEMEGA_F3_REPEAT_LAYERS')}
    result.setdefault('TILEMEGA_F3_PREFETCH_DISTANCE', '2')
    return result


def validate(folder, decision, final):
    from dnn4_bert_current import validate_timing_gpu
    from dnn4_bert_model_cohort import validate_bert_checks, validate_bert_output_identity
    from tilemega.build.identity import verify

    integration = read(folder / 'integration.json')
    keys = ('preparation', 'original_plan', 'binding_contract', 'units', 'fixtures',
            'control', 'model_cohort', 'model_bit_proof', 'ledger_receipt',
            'trace_source_proof', 'routing_preference')
    required = {integration[k] for k in keys} | set(decision['unit_gates'])
    if (not required <= integration['receipts'].keys()
            or any(digest(p) != value for p, value in integration['receipts'].items())):
        raise ValueError('QKV Repeat integration has a missing or changed receipt')
    pair = decision['pair']
    gpu = pair['gpu']
    if (decision['batch'] != 8 or not decision['accepted'] or decision['minimum_gain'] != .005
            or pair['ratio'] >= 1 - max(.005, pair['threshold'])
            or final['library'] != decision['library'] or final['gate'] != decision['gate']
            or final['pair'] != pair or final['unit_gates'] != decision['unit_gates']):
        raise ValueError('QKV Repeat lacks its original independent gain gate')
    identity = verify(Path(decision['library']))
    previous = verify(Path(decision['reference']))
    prepared = read(integration['preparation'])
    plan = read(integration['original_plan'])
    profile = dict(json.loads(plan['reference']['profile']), TILEMEGA_F3_REPEAT_OUTPUT_MAP='1')
    expected = json.loads(json.dumps(previous['plan']['native_forward']))
    for index in range(6, 57, 5):
        expected['stages'][index].update(slots=4, store='tma')
    if (prepared['status'] != 'prepared_CPU_only' or plan['minimum_gain'] != .005
            or prepared['model']['library'] != decision['library']
            or prepared['model']['artifact_id'] != identity['artifact_id']
            or prepared['profile'] != profile or plan['profile'] != profile
            or plan['reference']['library'] != decision['reference']
            or plan['reference']['gate'] != decision['reference_gate']
            or identity['trace'] or any(r['spill'] for r in identity['resources'].values())
            or identity['plan']['runtime_batch'] != 8
            or identity['plan']['native_forward'] != expected
            or identity['plan']['buffers'] != previous['plan']['buffers']):
        raise ValueError('QKV Repeat changes its fixed preparation, first QKV or other groups')
    source = Path(decision['library'] + '.cu').read_text()
    flags = dict(re.findall(r'^#define (TILEMEGA_F3_\w+) (\d+)\s*$', source, re.M))
    if effective(flags) != effective(profile) or 'Repeat<12,Loop_' not in source or 'Repeat declined:' in source:
        raise ValueError('QKV Repeat lacks its exact feature profile or compact loop')
    bindings = read(integration['binding_contract'])
    if (not bindings['passed'] or bindings['checks'] != 1380
            or len(bindings['optional_output_maps']) != 12
            or bindings['optional_output_maps'][0] is not None
            or any(v is None for v in bindings['optional_output_maps'][1:])):
        raise ValueError('QKV Repeat lacks all reversed optional-map lookups')
    units = read(integration['units'])
    fixtures = read(integration['fixtures'])
    if (units['status'] != 'prepared_CPU_only' or not units['ordering_all_reusable']
            or [r['name'] for r in units['rows']] != ['body0', 'body1', 'order0', 'order1', 'order2', 'order3', 'map']
            or not fixtures['passed'] or fixtures['old_b16_words'] != 1056
            or [r['name'] for r in fixtures['rows']] != ['control', 'body', 'map']):
        raise ValueError('QKV Repeat lacks its committed body/map and old-parser preservation proofs')
    for row in fixtures['rows']:
        executed = read(row['executed_build'])
        committed = read(row['committed_build'])
        expected_profile = json.loads(plan['reference']['profile']) if row['name'] == 'control' else profile
        if (not row['passed'] or row['encoded_words'] <= 0
                or digest(row['executed_build']) != row['executed_build_sha256']
                or digest(row['committed_build']) != row['committed_build_sha256']
                or digest(executed['binary']) != executed['binary_sha256']
                or digest(committed['binary']) != committed['binary_sha256']
                or effective(executed['profile']) != effective(expected_profile)
                or effective(committed['profile']) != effective(expected_profile)):
            raise ValueError('QKV Repeat committed fixture differs from its executed stream')
    gates = decision['unit_gates']
    if len(gates) != 6 or len(set(gates)) != 6:
        raise ValueError('QKV Repeat requires body, actual map-table and all four ordering gates')
    expected_protocol = Counter(unit=5, memcheck=1, initcheck=1, racecheck=5, synccheck=5)
    build_paths = [units['rows'][1]['build'], units['rows'][6]['build'],
                   *[r['build'] for r in units['rows'][2:6]]]
    for index, (path, build_path) in enumerate(zip(gates, build_paths)):
        gate, build = read(path), read(build_path)
        processes = gate['processes']
        expected_sha = (build['binary_sha256'] if index < 2 else units['rows'][index]['executed_sha256'])
        if (not gate['passed'] or len(processes) != 17 or len({p['pid'] for p in processes}) != 17
                or Counter(p['name'].split('-')[0] for p in processes) != expected_protocol
                or len({p['output_hash'] for p in processes}) != 1
                or build['spill_bytes'] != 0 or effective(build['profile']) != effective(profile)
                or digest(build['binary']) != build['binary_sha256']
                or any(p['exit_code'] or read(p['guard'])['code']
                       or p['binary_sha256'] != expected_sha or digest(p['log']) != p['sha256']
                       or p.get('gpu_lock', f"/tmp/tilemega_gpu{gate['gpu']}.lock") !=
                          f"/tmp/tilemega_gpu{gate['gpu']}.lock"
                       for p in processes)):
            raise ValueError('QKV Repeat component protocol or actual binary differs')
        if index >= 2:
            proof = units['rows'][index]
            if (not proof['passed'] or proof['encoded_words'] <= 0
                    or proof['reference_gate'] != path or digest(proof['executed']) != expected_sha):
                raise ValueError('QKV Repeat ordering reuse loses its actual executed identity')
    control = read(integration['control'])
    expected_sha = read(units['rows'][0]['build'])['binary_sha256']
    if (not control['passed'] or len(control['processes']) != 5
            or len({p['pid'] for p in control['processes']}) != 5
            or any(p['exit_code'] or read(p['guard'])['code']
                   or p['binary_sha256'] != expected_sha for p in control['processes'])):
        raise ValueError('QKV Repeat lacks five fresh control processes')
    model = read(integration['model_cohort'])
    if (not model['passed'] or model['passes'] != '5/5' or len(model['processes']) != 5
            or len({p['pid'] for p in model['processes']}) != 5 or model['gate'] != decision['gate']
            or model['execution']['artifact_id'] != identity['artifact_id']
            or any(p['exit_code'] or read(p['guard'])['code'] for p in model['processes'])):
        raise ValueError('QKV Repeat lacks its strict deterministic model cohort')
    checks = [read(p['check']) for p in model['processes']]
    validate_bert_checks(checks, identity['artifact_id'])
    validate_bert_output_identity(checks, read(decision['reference_gate']), identity['artifact_id'])
    bit = read(integration['model_bit_proof'])
    if (not bit['compared'] or not bit['bitwise_equal'] or bit['reference'] != decision['reference_gate']
            or bit['candidate'] != integration['model_cohort']):
        raise ValueError('QKV Repeat changed a named output')
    allocation = read(folder / 'allocation.json')
    preference = read(integration['routing_preference'])
    if (gpu not in (0, 1) or allocation['gpu'] != gpu or allocation['reanchored']
            or allocation['original_anchor_gpu'] != 3 or allocation['python_entry_root'] != '/root/python'
            or preference['preferred_gpu_indices'] != [0, 1]):
        raise ValueError('QKV Repeat changed its preferred-card or fixed-target allocation')
    pids = []
    for measured, order in ((pair, 'ABABAB'), (final['measurement'], 'BBB')):
        validate_timing_gpu(measured, gpu, order)
        for row in measured['measurements']:
            value = read(row['measurement'])
            candidate = row['arm'] == 'B'
            artifact = identity if candidate else previous
            gate = decision['gate'] if candidate else decision['reference_gate']
            if (value['artifact_id'] != artifact['artifact_id'] or value['library_path'] != artifact['binary_path']
                    or value['numerical_gate_path'] != gate or value['batch'] != 8 or value['precision'] != 'bf16'
                    or value['input_sha256'] != pair['input_sha256']
                    or value['source_checkpoint_sha256'] != checks[0]['source_checkpoint_sha256']
                    or statistics.median(value['samples_ms']) != row['median_ms']):
                raise ValueError('QKV Repeat timing substitutes an artifact, input or gate')
            boundaries = [json.loads(line) for line in Path(value['guard_log']).read_text().splitlines()]
            for label, clock in (('before-measurement', 'start_sm_clock_mhz'), ('after-measurement', 'end_sm_clock_mhz')):
                boundary = next(v for v in boundaries if v.get('label') == label)
                if boundary['time'] < preference['effective_at'] or float(boundary['telemetry']['clocks.sm']) != row[clock]:
                    raise ValueError('QKV Repeat lacks its actual preferred-card clock boundary')
            pids.append(value['pid'])
    a = [r['median_ms'] for r in pair['measurements'] if r['arm'] == 'A']
    b = [r['median_ms'] for r in pair['measurements'] if r['arm'] == 'B']
    threshold = max(.005, 2 * max((max(v) - min(v)) / statistics.median(v) for v in (a, b)))
    values = [r['median_ms'] for r in final['measurement']['measurements']]
    events = sorted(v for r in final['measurement']['measurements'] for v in read(r['measurement'])['samples_ms'])
    if (len(set(pids)) != 9 or pair['candidate_ms'] != statistics.median(b)
            or pair['reference_ms'] != statistics.median(a) or pair['ratio'] != statistics.median(b) / statistics.median(a)
            or pair['threshold'] != threshold or final['measurement']['candidate_ms'] != statistics.median(values)
            or final['measurement']['process_range_ms'] != max(values) - min(values)
            or final['measurement']['p90_ms'] != events[(9 * len(events) + 9) // 10 - 1]):
        raise ValueError('QKV Repeat changed its independent timing statistics')
    ledger, trace = read(integration['ledger_receipt']), read(integration['trace_source_proof'])
    traced = verify(Path(trace['library']))
    stages = list(csv.DictReader(Path(final['ledger']).open()))
    if (not trace['passed'] or ledger['trace_level'] != 1 or ledger['gpu'] != gpu
            or ledger['reference'] != decision['library'] or ledger['reference_sha256'] != digest(decision['library'])
            or ledger['artifact_id'] != traced['artifact_id'] or not traced['trace']
            or trace['reference'] != decision['library'] or trace['reference_sha256'] != digest(decision['library'])
            or ledger['ledger'] != final['ledger'] or sorted(int(r['stage']) for r in stages) != list(range(63))
            or Path(trace['library'] + '.cu').read_text().replace(
                '#define TILEMEGA_TRACE_STAGE 1', '#define TILEMEGA_TRACE_STAGE 0') != source):
        raise ValueError('QKV Repeat lacks its exact 63-stage nontrace-reference ledger')
    return gpu
