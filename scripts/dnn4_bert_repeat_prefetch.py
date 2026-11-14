#!/usr/bin/env python3
"""Validate the fixed B16 Repeat/output-map/prefetch admission evidence."""
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


def validate(folder, decision, final):
    from dnn4_bert_current import validate_timing_gpu
    from dnn4_bert_model_cohort import validate_bert_checks
    from tilemega.build.identity import verify
    integration = read(folder/'integration.json')
    for key, value in integration['receipts'].items():
        if digest(key) != value:
            raise ValueError('Repeat prefetch integration receipt changed: '+key)
    pair = decision['pair']
    if (decision['batch'] != 16 or not decision['accepted']
            or decision['minimum_gain'] != .03
            or pair['ratio'] >= 1-max(.03, pair['threshold'])
            or final['library'] != decision['library'] or final['gate'] != decision['gate']
            or final['pair'] != pair or final['unit_gates'] != decision['unit_gates']):
        raise ValueError('Repeat prefetch lacks its original strict gain gate')
    preparation = read(integration['preparation'])
    original_plan = read(integration['original_plan'])
    source = Path(decision['library']+'.cu').read_text()
    previous = Path(preparation['predecessor']+'.cu').read_text()
    pattern = r'^#define TILEMEGA_F3_PREFETCH_DISTANCE [12]\n'
    if (preparation['status'] != 'prepared_CPU_only'
            or preparation['library'] != decision['library']
            or original_plan['minimum_gain'] != .03
            or original_plan['reference']['library'] != decision['reference']
            or re.findall(pattern, source, re.M) != ['#define TILEMEGA_F3_PREFETCH_DISTANCE 1\n']
            or re.sub(pattern, '', source, flags=re.M) != re.sub(pattern, '', previous, flags=re.M)
            or not read(preparation['source_differential'])['passed']):
        raise ValueError('Repeat prefetch changed more than the prescribed hint distance')
    identity = verify(Path(decision['library']))
    if identity['trace'] or any(v['spill'] for v in identity['resources'].values()):
        raise ValueError('Repeat prefetch is not a zero-spill production program')
    expected = read(decision['reference']+'.plan.json')['native_forward']
    for stage in [1, *range(4, 60, 5)]:
        expected['stages'][stage].update(tile_m=128, tile_n=192, tk=64,
                                          split_k=1, slots=4, store='tma', mc=0)
    if identity['plan']['native_forward'] != expected:
        raise ValueError('Repeat prefetch changed coordinates outside the selected groups')
    bindings = read(integration['binding_contract'])
    main = read(integration['main_emission'])
    defaults = read(integration['defaults'])
    if (not bindings['passed'] or bindings['checks'] != 1380
            or bindings['optional_output_maps'] != [1]+[None]*11
            or not main['passed'] or [v['flag'] for v in main['rows']] != [0, 1]
            or any(not v['passed'] or not v['generated_source_identical']
                   or v['encoded_words'] <= 0 for v in main['rows'])
            or main['rows'][1]['prototype'] != preparation['predecessor']
            or not defaults['passed'] or [v['batch'] for v in defaults['rows']] != [1, 2, 4, 8, 16]
            or any(not v['passed'] or not v['default_explicit0_source_identical']
                   or v['encoded_words'] <= 0 for v in defaults['rows'])):
        raise ValueError('Repeat prefetch lacks its binding or main/default emission proof')
    body = read(integration['body_reuse'])
    fixture = read(integration['map_fixture'])
    if (not body['passed'] or body['encoded_words'] != 22656
            or digest(body['new_binary']) != body['new_sha256']
            or digest(body['reference_binary']) != body['reference_sha256']
            or not fixture['passed'] or fixture['encoded_words'] != 1056):
        raise ValueError('Repeat prefetch component reuse lacks exact instruction evidence')
    gates = decision['unit_gates']
    if len(gates) != 6 or len(set(gates)) != 6:
        raise ValueError('Repeat prefetch requires body, map and four ordering gates')
    for index, path in enumerate(gates):
        gate = read(path)
        processes = gate['processes']
        if (not gate['passed'] or len({p['pid'] for p in processes}) != 17
                or Counter(p['name'].split('-')[0] for p in processes) !=
                Counter(unit=5, memcheck=1, initcheck=1, racecheck=5, synccheck=5)
                or len({p['output_hash'] for p in processes}) != 1
                or any(p['exit_code'] or read(p['guard'])['code']
                       or digest(p['log']) != p['sha256'] for p in processes)):
            raise ValueError('Repeat prefetch has an incomplete component cohort')
        expected_binary = (body['reference_sha256'] if index == 0 else
                           fixture['prototype_sha256'] if index == 1 else None)
        if expected_binary and any(p['binary_sha256'] != expected_binary for p in processes):
            raise ValueError('Repeat prefetch reuses a different body or map fixture')
        if index >= 2:
            build = read(integration['order_builds'][index-2])
            if (build['profile'].get('TILEMEGA_F3_PREFETCH_DISTANCE') != '1'
                    or build['profile'].get('TILEMEGA_F3_REGISTER_BUDGET') != '3'
                    or '-DTILEMEGA_F3_PREFETCH_DISTANCE=1' not in build['command']
                    or any(p['binary_sha256'] != build['binary_sha256'] for p in processes)
                    or digest(build['binary']) != build['binary_sha256']
                    or gate['output_hash'] != read(integration['old_order_gates'][index-2])['output_hash']):
                raise ValueError('Repeat prefetch ordering gate has a different binary or output')
    model = read(integration['model_cohort'])
    bit = read(integration['model_bit_proof'])
    outputs = lambda p: [v['output_sha256'] for c in read(p)['cases'] for v in c['outputs']]
    if (not model['passed'] or model['passes'] != '5/5'
            or len({p['pid'] for p in model['processes']}) != 5
            or model['execution']['artifact_id'] != identity['artifact_id']
            or model['gate'] != decision['gate'] or not bit['passed']
            or bit['candidate'] != decision['gate']
            or bit['hashes'] != outputs(decision['gate'])
            or bit['hashes'] != outputs(bit['reference'])):
        raise ValueError('Repeat prefetch lacks its strict model and output byte proof')
    if any(p['exit_code'] or read(p['guard'])['code'] for p in model['processes']):
        raise ValueError('Repeat prefetch model process or guard failed')
    checked = validate_bert_checks([read(p['check']) for p in model['processes']], identity['artifact_id'])
    if model['cases'] != checked['cases']:
        raise ValueError('Repeat prefetch changes the strict model aggregate')
    selection = read(integration['selection'])
    specs = read(integration['candidate_spec'])
    if (len(specs) != 4 or specs[0]['library'] != decision['library']
            or specs[1]['library'] != decision['reference']
            or len({json.dumps(read(v['library']+'.plan.json')['native_forward'], sort_keys=True)
                    for v in specs}) != 4
            or selection['winner_library'] != decision['library']):
        raise ValueError('Repeat prefetch did not win the four distinct-program selection')
    by_arm = {arm: [v['median_ms'] for v in pair['measurements'] if v['arm'] == arm]
              for arm in 'AB'}
    medians = {arm: statistics.median(values) for arm, values in by_arm.items()}
    noise = max(.005, 2*max((max(v)-min(v))/statistics.median(v) for v in by_arm.values()))
    if (pair['reference_ms'] != medians['A'] or pair['candidate_ms'] != medians['B']
            or pair['ratio'] != medians['B']/medians['A'] or pair['threshold'] != noise):
        raise ValueError('Repeat prefetch changes its measured gain or noise')
    for arm, spec in zip('ABCD', specs):
        rows = [r for r in selection['measurements'] if r['arm'] == arm]
        for row in rows:
            measured = read(row['measurement'])
            if (Path(measured['numerical_gate_path']).resolve() != Path(spec['gate']).resolve()
                    or measured['artifact_id'] != read(spec['library']+'.identity.json')['artifact_id']):
                raise ValueError('Repeat prefetch selection substituted a different program')
    ranking = sorted(selection['ranking'], key=lambda r: r['median_ms'])
    for row in ranking:
        values = [v['median_ms'] for v in selection['measurements'] if v['arm'] == row['arm']]
        if row['median_ms'] != statistics.median(values):
            raise ValueError('Repeat prefetch selection ranking changes its samples')
    if (ranking[0]['library'] != decision['library'] and not selection['tied']
            or [s['tune_priority'] for s in specs] != [0, 1, 2, 3]):
        raise ValueError('Repeat prefetch selection lacks its predeclared priority')
    allocation = read(folder/'allocation.json')
    authorization = read(allocation['snapshot'])
    gpu = allocation['gpu']
    if (gpu not in [0, 2] or authorization['allowed_gpu_indices'] != [0, 2]
            or authorization['same_card_pair_required'] is not False
            or allocation['original_anchor_gpu'] != 3 or allocation['reanchored']
            or allocation['python_entry_root'] != '/root/python'
            or digest(allocation['snapshot']) != allocation['snapshot_sha256']):
        raise ValueError('Repeat prefetch changes its authorized allocation or target')
    for measurement, order in [(pair, 'ABABAB'), (final['measurement'], 'BBB'),
                               (selection, 'ABCDDCBA')]:
        validate_timing_gpu(measurement, gpu, order)
        for row in measurement['measurements']:
            measured = read(row['measurement'])
            boundaries = [json.loads(s) for s in Path(measured['guard_log']).read_text().splitlines()]
            if any(v['time'] < authorization['gpu_fallback_at'] for v in boundaries):
                raise ValueError('Repeat prefetch timing predates its device authorization')
    values = [r['median_ms'] for r in final['measurement']['measurements']]
    events = sorted(v for r in final['measurement']['measurements']
                    for v in read(r['measurement'])['samples_ms'])
    if (final['measurement']['candidate_ms'] != statistics.median(values)
            or final['measurement']['process_range_ms'] != max(values)-min(values)
            or final['measurement']['p90_ms'] != events[(9*len(events)+9)//10-1]):
        raise ValueError('Repeat prefetch final aggregate changes its measured samples')
    ledger = read(integration['ledger_receipt'])
    trace_proof = read(integration['trace_source_proof'])
    if (ledger['gpu'] != gpu or ledger['trace_level'] != 1
            or ledger['reference'] != decision['library']
            or ledger['reference_sha256'] != digest(decision['library'])
            or ledger['ledger'] != final['ledger']
            or len(list(csv.DictReader(Path(final['ledger']).open()))) != 63
            or not trace_proof['passed'] or trace_proof['reference'] != decision['library']
            or trace_proof['reference_sha256'] != digest(decision['library'])):
        raise ValueError('Repeat prefetch lacks its exact production-binary ledger')
    return gpu
