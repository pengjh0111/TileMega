"""Validate the Rows128 attention successor on the current B8 program."""
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
    from dnn4_bert_current import validate_timing_gpu
    from dnn4_bert_model_cohort import validate_bert_checks, validate_bert_output_identity
    from dnn4_bert_prefetch import canonical_source
    from tilemega.build.identity import verify

    integration = read(folder / 'integration.json')
    keys = ('preparation', 'original_plan', 'source_differential', 'defaults',
            'units', 'committed_fixture', 'control', 'model_cohort',
            'model_bit_proof', 'ledger_receipt', 'trace_source_proof', 'authorization',
            'order_identity')
    required = {integration[k] for k in keys} | set(decision['unit_gates'])
    if (not required <= integration['receipts'].keys()
            or any(digest(p) != value for p, value in integration['receipts'].items())):
        raise ValueError('Rows128 attention has a missing or changed receipt')
    pair = decision['pair']
    gpu = pair['gpu']
    if (decision['batch'] != 8 or not decision['accepted']
            or decision['minimum_gain'] != .005
            or pair['ratio'] >= 1 - max(.005, pair['threshold'])
            or final['library'] != decision['library'] or final['gate'] != decision['gate']
            or final['pair'] != pair or final['unit_gates'] != decision['unit_gates']):
        raise ValueError('Rows128 attention lacks its predeclared end-to-end gain')
    plan = read(integration['original_plan'])
    prepared = read(integration['preparation'])
    differential = read(integration['source_differential'])
    identity = verify(Path(decision['library']))
    previous = verify(Path(decision['reference']))
    profile = dict(json.loads(plan['reference']['profile']),
                   TILEMEGA_F3_WGMMA_QK='1', TILEMEGA_F3_WGMMA_PV='1')
    source = Path(decision['library'] + '.cu').read_text()
    off = source
    for feature in ('TILEMEGA_F3_WGMMA_QK', 'TILEMEGA_F3_WGMMA_PV'):
        define = '#define ' + feature + ' 1\n'
        if off.count(define) != 1:
            raise ValueError('Rows128 attention lacks its two fixed feature definitions')
        off = off.replace(define, '')
    if (plan['minimum_gain'] != .005 or plan['profile'] != profile
            or prepared['status'] != 'prepared_CPU_only' or prepared['profile'] != profile
            or prepared['model']['library'] != decision['library']
            or prepared['model']['artifact_id'] != identity['artifact_id']
            or plan['reference']['library'] != decision['reference']
            or plan['reference']['gate'] != decision['reference_gate']
            or not differential['passed'] or differential['unchanged_rows'] != 128
            or not all(differential[k] for k in
                       ('coordinates_equal', 'buffers_equal', 'all_stage_metadata_equal'))
            or differential['only_changed_defines'] !=
               ['TILEMEGA_F3_WGMMA_QK=1', 'TILEMEGA_F3_WGMMA_PV=1']
            or canonical_source(off) != canonical_source(Path(decision['reference'] + '.cu').read_text())
            or source.count('Sm90F3Attention<8,128>') != 12
            or identity['trace'] or identity['plan']['runtime_batch'] != 8
            or any(r['spill'] for r in identity['resources'].values())
            or any(identity['plan'][k] != previous['plan'][k] for k in
                   ('native_forward', 'buffers', 'stages'))):
        raise ValueError('Rows128 attention changes its prepared artifact or feature scope')
    defaults = read(integration['defaults'])
    proof = read(defaults['reused_exact_word_proof'])
    if (not defaults['passed'] or not defaults['source_identical']
            or defaults['reference'] != decision['reference']
            or defaults['encoded_words'] != 33616
            or digest(defaults['reused_exact_word_proof']) != defaults['proof_sha256']
            or not proof['passed'] or proof['encoded_words'] != 33616
            or proof['reference_sha256'] != digest(decision['reference'])):
        raise ValueError('Rows128 attention drops the preserved main-codegen control')
    units = read(integration['units'])
    fixture = read(integration['committed_fixture'])
    gates = decision['unit_gates']
    if (units['status'] != 'prepared_CPU_only' or not units['ordering_all_reusable']
            or len(gates) != 5 or len(set(gates)) != 5
            or [r['name'] for r in units['rows']] !=
               ['body0', 'body1', 'order0', 'order1', 'order2', 'order3']
            or not fixture['passed'] or [r['flag'] for r in fixture['rows']] != [0, 1]):
        raise ValueError('Rows128 attention lacks complete component and committed-fixture coverage')
    for index, row in enumerate(fixture['rows']):
        build = read(row['prototype_build'])
        expected = profile if index else json.loads(plan['reference']['profile'])
        if (not row['passed'] or row['profile'] != expected or build['profile'] != expected
                or row['encoded_words'] <= 0
                or row['prototype_build'] != units['rows'][index]['build']
                or digest(row['prototype_build']) != row['prototype_build_sha256']
                or digest(row['binary']) != row['binary_sha256']
                or row['prototype'] != build['binary']
                or digest(row['prototype']) != row['prototype_sha256']
                or '-DDNN4_ATTENTION_CURRENT_B8_TEST' not in row['command']):
            raise ValueError('Rows128 attention committed fixture does not reproduce its executed arm')
    expected_protocol = Counter(unit=5, memcheck=1, initcheck=1, racecheck=5, synccheck=5)
    order_identity = read(integration['order_identity'])
    if not order_identity['passed'] or len(order_identity['rows']) != 4:
        raise ValueError('Rows128 attention lacks the complete historical ordering identity chain')
    for index, path in enumerate(gates):
        gate = read(path)
        processes = gate['processes']
        row = units['rows'][index + 1]
        build = read(row['build'])
        expected_sha = (build['binary_sha256'] if index == 0 else
                        order_identity['rows'][index - 1]['executed_sha256'])
        if (not gate['passed'] or len(processes) != 17
                or len({p['pid'] for p in processes}) != 17
                or Counter(p['name'].split('-')[0] for p in processes) != expected_protocol
                or len({p['output_hash'] for p in processes}) != 1
                or build['profile'] != profile or build['spill_bytes'] != 0
                or digest(build['binary']) != build['binary_sha256']
                or any(p['exit_code'] or read(p['guard'])['code']
                       or p['binary_sha256'] != expected_sha
                       or digest(p['log']) != p['sha256']
                       or p.get('gpu_lock', f"/tmp/tilemega_gpu{gate['gpu']}.lock") !=
                          f"/tmp/tilemega_gpu{gate['gpu']}.lock" for p in processes)):
            raise ValueError('Rows128 attention has an incomplete or substituted component cohort')
        if index and (not row['whole_words_equal'] or not row['passed']
                      or row['encoded_words'] <= 0 or row['reference_gate'] != path
                      or digest(row['reference']) != row['reference_sha256']):
            raise ValueError('Rows128 attention ordering reuse lacks its instruction identity')
        if index:
            chain = order_identity['rows'][index - 1]
            if (not chain['passed'] or chain['name'] != row['name']
                    or chain['candidate'] != build['binary'] or chain['gate'] != path
                    or chain['intermediate'] != row['reference']
                    or digest(chain['executed']) != expected_sha
                    or chain['encoded_words'] != row['encoded_words']
                    or len(set(chain['word_digests'])) != 1
                    or len(chain['word_digests']) != 3):
                raise ValueError('Rows128 attention ordering cohort does not match all three instruction streams')
    control = read(integration['control'])
    control_sha = read(units['rows'][0]['build'])['binary_sha256']
    if (not control['passed'] or len(control['processes']) != 5
            or len({p['pid'] for p in control['processes']}) != 5
            or any(p['exit_code'] or read(p['guard'])['code']
                   or p['binary_sha256'] != control_sha for p in control['processes'])):
        raise ValueError('Rows128 attention lacks its five fresh per-head control processes')
    model = read(integration['model_cohort'])
    if (not model['passed'] or model['passes'] != '5/5' or len(model['processes']) != 5
            or len({p['pid'] for p in model['processes']}) != 5
            or model['gate'] != decision['gate']
            or model['execution']['artifact_id'] != identity['artifact_id']
            or any(p['exit_code'] or read(p['guard'])['code'] for p in model['processes'])):
        raise ValueError('Rows128 attention lacks its complete strict model cohort')
    checks = [read(p['check']) for p in model['processes']]
    validate_bert_checks(checks, identity['artifact_id'])
    validate_bert_output_identity(checks, read(decision['reference_gate']), identity['artifact_id'])
    bit = read(integration['model_bit_proof'])
    if (not bit['compared'] or not bit['bitwise_equal']
            or bit['candidate'] != integration['model_cohort']
            or bit['reference'] != decision['reference_gate']):
        raise ValueError('Rows128 attention changed a named model output')
    allocation = read(folder / 'allocation.json')
    authorization = read(integration['authorization'])
    if (gpu not in (0, 1, 2, 3) or allocation['gpu'] != gpu
            or allocation['original_anchor_gpu'] != 3 or allocation['reanchored']
            or allocation['python_entry_root'] != '/root/python'
            or authorization['allowed_gpu_indices'] != [0, 1, 2, 3]
            or authorization['latest_request'] != '可以使用任意卡'):
        raise ValueError('Rows128 attention changed its allocation or fixed target')
    pids = []
    for measured, order in ((pair, 'ABABAB'), (final['measurement'], 'BBB')):
        validate_timing_gpu(measured, gpu, order)
        for row in measured['measurements']:
            raw = read(row['measurement'])
            expected = identity if row['arm'] == 'B' else previous
            gate = decision['gate'] if row['arm'] == 'B' else decision['reference_gate']
            if (raw['artifact_id'] != expected['artifact_id']
                    or raw['library_path'] != expected['binary_path']
                    or raw['numerical_gate_path'] != gate
                    or raw['input_sha256'] != pair['input_sha256']
                    or raw['source_checkpoint_sha256'] != checks[0]['source_checkpoint_sha256']
                    or raw['batch'] != 8 or raw['precision'] != 'bf16'
                    or statistics.median(raw['samples_ms']) != row['median_ms']):
                raise ValueError('Rows128 attention substituted a timing artifact, input or gate')
            boundaries = [json.loads(line) for line in Path(raw['guard_log']).read_text().splitlines()]
            for label, clock in (('before-measurement', 'start_sm_clock_mhz'),
                                 ('after-measurement', 'end_sm_clock_mhz')):
                boundary = next(v for v in boundaries if v.get('label') == label)
                if float(boundary['telemetry']['clocks.sm']) != row[clock]:
                    raise ValueError('Rows128 attention changed a measured SM clock')
            pids.append(raw['pid'])
    a = [r['median_ms'] for r in pair['measurements'] if r['arm'] == 'A']
    b = [r['median_ms'] for r in pair['measurements'] if r['arm'] == 'B']
    threshold = max(.005, 2 * max((max(v) - min(v)) / statistics.median(v) for v in (a, b)))
    values = [r['median_ms'] for r in final['measurement']['measurements']]
    events = sorted(v for r in final['measurement']['measurements']
                    for v in read(r['measurement'])['samples_ms'])
    if (len(set(pids)) != 9 or pair['candidate_ms'] != statistics.median(b)
            or pair['reference_ms'] != statistics.median(a)
            or pair['ratio'] != statistics.median(b) / statistics.median(a)
            or pair['threshold'] != threshold
            or final['measurement']['candidate_ms'] != statistics.median(values)
            or final['measurement']['process_range_ms'] != max(values) - min(values)
            or final['measurement']['p90_ms'] != events[(9 * len(events) + 9) // 10 - 1]):
        raise ValueError('Rows128 attention changed its independent timing statistics')
    ledger = read(integration['ledger_receipt'])
    trace = read(integration['trace_source_proof'])
    traced = verify(Path(trace['library']))
    stages = list(csv.DictReader(Path(final['ledger']).open()))
    if (not trace['passed'] or trace['reference'] != decision['library']
            or trace['reference_sha256'] != digest(decision['library'])
            or ledger['gpu'] != gpu or ledger['trace_level'] != 1
            or ledger['artifact_id'] != traced['artifact_id'] or not traced['trace']
            or ledger['reference'] != decision['library']
            or ledger['reference_sha256'] != digest(decision['library'])
            or ledger['ledger'] != final['ledger']
            or sorted(int(r['stage']) for r in stages) != list(range(63))
            or Path(trace['library'] + '.cu').read_text().replace(
                '#define TILEMEGA_TRACE_STAGE 1', '#define TILEMEGA_TRACE_STAGE 0') != source):
        raise ValueError('Rows128 attention lacks its exact 63-stage production-binary ledger')
    return gpu
