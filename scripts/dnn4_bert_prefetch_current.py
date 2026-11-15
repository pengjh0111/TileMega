"""Validate PF1 on the accepted repeated-QKV B8 context."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def words(path):
    sass = subprocess.check_output(['cuobjdump', '-sass', str(path)], text=True)
    return re.findall(r'/\*\s*(0x[0-9a-f]+)\s*\*/', sass)


def validate(folder, decision, final):
    from dnn4_bert_current import validate_timing_gpu
    from dnn4_bert_model_cohort import validate_bert_checks, validate_bert_output_identity
    from dnn4_bert_qkv_repeat import effective, validate as validate_parent
    from tilemega.build.identity import verify

    integration = read(folder / 'integration.json')
    keys = ('preparation', 'original_plan', 'source_differential', 'body_map_identity',
            'model_cohort', 'model_bit_proof', 'ledger_receipt', 'trace_source_proof',
            'routing_preference', 'driver_sources', 'parent_experiment', 'parent_final', 'parent_integration')
    required = {integration[k] for k in keys} | set(integration['order_builds']) | set(decision['unit_gates'])
    if (not required <= integration['receipts'].keys()
            or any(digest(p) != value for p, value in integration['receipts'].items())):
        raise ValueError('PF1 integration has a missing or changed receipt')
    frozen = read(integration['driver_sources'])
    if len(frozen) < 6 or any(digest(p) != value for p, value in frozen.items()):
        raise ValueError('PF1 changed a frozen driver source')
    parent_folder = folder.parent / 'qkv_repeat_tma_current_b8_admitted_v1'
    if (integration['parent_experiment'] != str(parent_folder / 'experiment.json')
            or integration['parent_final'] != str(parent_folder / 'final_result.json')
            or integration['parent_integration'] != str(parent_folder / 'integration.json')):
        raise ValueError('PF1 substitutes its admitted repeated-QKV predecessor')
    parent, parent_final = read(integration['parent_experiment']), read(integration['parent_final'])
    validate_parent(parent_folder, parent, parent_final)
    pair = decision['pair']
    pair_gpu, gpu = pair['gpu'], final['measurement']['gpu']
    if (decision['batch'] != 8 or not decision['accepted'] or decision['minimum_gain'] != .01
            or pair['ratio'] >= 1 - max(.01, pair['threshold'])
            or decision['reference'] != parent['library'] or decision['reference_gate'] != parent['gate']
            or final['library'] != decision['library'] or final['gate'] != decision['gate']
            or final['pair'] != pair or final['unit_gates'] != decision['unit_gates']):
        raise ValueError('PF1 lacks its original independent one-percent gain gate')
    identity, previous = verify(Path(decision['library'])), verify(Path(decision['reference']))
    prepared, plan = read(integration['preparation']), read(integration['original_plan'])
    profile = dict(effective(json.loads(plan['reference']['profile'])), TILEMEGA_F3_PREFETCH_DISTANCE='1')
    if (prepared['status'] != 'prepared_CPU_only' or plan['minimum_gain'] != .01
            or plan['reference']['library'] != decision['reference']
            or plan['reference']['gate'] != decision['reference_gate']
            or effective(prepared['profile']) != profile or effective(plan['profile']) != profile
            or prepared['model']['library'] != decision['library']
            or prepared['model']['artifact_id'] != identity['artifact_id']
            or identity['trace'] or identity['plan']['runtime_batch'] != 8
            or any(r['spill'] for r in identity['resources'].values())
            or identity['plan']['native_forward'] != previous['plan']['native_forward']
            or identity['plan']['buffers'] != previous['plan']['buffers']):
        raise ValueError('PF1 changes the fixed profile, coordinates or preparation')
    source, old_source = (Path(p + '.cu').read_text() for p in (decision['library'], decision['reference']))
    differential = read(integration['source_differential'])
    if (source.count('#define TILEMEGA_F3_PREFETCH_DISTANCE 1\n') != 1
            or source.replace('#define TILEMEGA_F3_PREFETCH_DISTANCE 1\n',
                              '#define TILEMEGA_F3_PREFETCH_DISTANCE 2\n') != old_source
            or not differential['passed'] or differential['only'] != 'prefetchdistance2to1'
            or not differential['coordinates_identical']):
        raise ValueError('PF1 changes more than the existing prefetch distance')
    proof = read(integration['body_map_identity'])
    gates = decision['unit_gates']
    if (not proof['passed'] or [r['name'] for r in proof['rows']] != ['body1', 'map']
            or [r['encoded_words'] for r in proof['rows']] != [21664, 1216]
            or len(gates) != 6 or len(set(gates)) != 6 or gates[:2] != parent['unit_gates'][:2]
            or len(integration['order_builds']) != 4):
        raise ValueError('PF1 loses its inherited body/map or four fresh ordering cases')
    for row, path in zip(proof['rows'], gates[:2]):
        candidate_words, previous_words = words(row['binary']), words(row['reference'])
        if (not row['passed'] or digest(row['binary']) != row['binary_sha256']
                or digest(row['reference']) != row['reference_sha256']
                or not candidate_words or candidate_words != previous_words
                or len(candidate_words) != row['encoded_words']
                or any(p['binary_sha256'] != row['reference_sha256'] for p in read(path)['processes'])):
            raise ValueError('PF1 body/map reuse lacks complete executed instruction identity')
    protocol = Counter(unit=5, memcheck=1, initcheck=1, racecheck=5, synccheck=5)
    for path, build_path in zip(gates[2:], integration['order_builds']):
        gate, build = read(path), read(build_path)
        processes = gate['processes']
        if (not gate['passed'] or len(processes) != 17 or len({p['pid'] for p in processes}) != 17
                or Counter(p['name'].split('-')[0] for p in processes) != protocol
                or len({p['output_hash'] for p in processes}) != 1
                or build['spill_bytes'] != 0 or effective(build['profile']) != profile
                or '-DTILEMEGA_F3_PREFETCH_DISTANCE=1' not in build['command']
                or digest(build['binary']) != build['binary_sha256']
                or any(digest(p) != value for p, value in build['inputs'].items())
                or any(p['exit_code'] or read(p['guard'])['code']
                       or p['binary_sha256'] != build['binary_sha256'] or digest(p['log']) != p['sha256']
                       or p.get('gpu_lock', f"/tmp/tilemega_gpu{gate['gpu']}.lock") !=
                          f"/tmp/tilemega_gpu{gate['gpu']}.lock" for p in processes)):
            raise ValueError('PF1 lacks a complete fresh actual-profile ordering cohort')
    model = read(integration['model_cohort'])
    if (not model['passed'] or model['passes'] != '5/5' or len(model['processes']) != 5
            or len({p['pid'] for p in model['processes']}) != 5 or model['gate'] != decision['gate']
            or model['execution']['artifact_id'] != identity['artifact_id']
            or any(p['exit_code'] or read(p['guard'])['code'] for p in model['processes'])):
        raise ValueError('PF1 lacks its strict deterministic model cohort')
    checks = [read(p['check']) for p in model['processes']]
    validate_bert_checks(checks, identity['artifact_id'])
    validate_bert_output_identity(checks, read(decision['reference_gate']), identity['artifact_id'])
    bit = read(integration['model_bit_proof'])
    if (not bit['passed'] or bit['reference'] != decision['reference_gate'] or bit['candidate'] != decision['gate']):
        raise ValueError('PF1 changed a named output')
    allocation, preference = read(folder / 'allocation.json'), read(integration['routing_preference'])
    if (gpu not in (0, 1) or pair_gpu not in (0, 1)
            or allocation['gpu'] != gpu or allocation['pair_gpu'] != pair_gpu
            or allocation['smoke_gpu'] != decision['smoke']['gpu'] or allocation['reanchored']
            or allocation['original_anchor_gpu'] != 3 or allocation['python_entry_root'] != '/root/python'
            or preference['preferred_gpu_indices'] != [0, 1]):
        raise ValueError('PF1 substitutes its preferred-card or fixed-target allocation')
    pids = []
    for measured, order in ((decision['smoke'], 'B'), (pair, 'ABABAB'), (final['measurement'], 'BBB')):
        measured_gpu = measured['gpu']
        expected_gpu = pair_gpu if order == 'ABABAB' else gpu if order == 'BBB' else measured_gpu
        if measured_gpu not in (0, 1) or measured_gpu != expected_gpu:
            raise ValueError('PF1 timing left the preferred cards or its recorded cohort card')
        validate_timing_gpu(measured, measured_gpu, order)
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
                raise ValueError('PF1 timing substitutes an artifact, input or gate')
            boundaries = [json.loads(line) for line in Path(value['guard_log']).read_text().splitlines()]
            for label, clock in (('before-measurement', 'start_sm_clock_mhz'), ('after-measurement', 'end_sm_clock_mhz')):
                boundary = next(v for v in boundaries if v.get('label') == label)
                if boundary['time'] < preference['effective_at'] or float(boundary['telemetry']['clocks.sm']) != row[clock]:
                    raise ValueError('PF1 lacks its actual preferred-card clock boundary')
            pids.append(value['pid'])
    a = [r['median_ms'] for r in pair['measurements'] if r['arm'] == 'A']
    b = [r['median_ms'] for r in pair['measurements'] if r['arm'] == 'B']
    threshold = max(.005, 2 * max((max(v) - min(v)) / statistics.median(v) for v in (a, b)))
    values = [r['median_ms'] for r in final['measurement']['measurements']]
    events = sorted(v for r in final['measurement']['measurements'] for v in read(r['measurement'])['samples_ms'])
    if (len(set(pids)) != 10 or pair['candidate_ms'] != statistics.median(b)
            or pair['reference_ms'] != statistics.median(a) or pair['ratio'] != statistics.median(b) / statistics.median(a)
            or pair['threshold'] != threshold or final['measurement']['candidate_ms'] != statistics.median(values)
            or final['measurement']['process_range_ms'] != max(values) - min(values)
            or final['measurement']['p90_ms'] != events[(9 * len(events) + 9) // 10 - 1]):
        raise ValueError('PF1 changed its independent timing statistics')
    ledger, trace = read(integration['ledger_receipt']), read(integration['trace_source_proof'])
    traced = verify(Path(trace['library']))
    stages = list(csv.DictReader(Path(final['ledger']).open()))
    if (not trace['passed'] or ledger['trace_level'] != 1 or ledger['gpu'] != gpu
            or ledger['reference'] != decision['library'] or ledger['reference_sha256'] != digest(decision['library'])
            or ledger['artifact_id'] != traced['artifact_id'] or not traced['trace']
            or any(r['spill'] for r in traced['resources'].values())
            or trace['reference'] != decision['library'] or trace['reference_sha256'] != digest(decision['library'])
            or ledger['ledger'] != final['ledger'] or sorted(int(r['stage']) for r in stages) != list(range(63))
            or Path(trace['library'] + '.cu').read_text().replace(
                '#define TILEMEGA_TRACE_STAGE 1', '#define TILEMEGA_TRACE_STAGE 0') != source):
        raise ValueError('PF1 lacks its exact 63-stage nontrace-reference ledger')
    return gpu
