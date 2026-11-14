"""Validate a fixed prefetch-distance follow-up on an admitted BERT program."""
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


def canonical_source(text):
    lines = text.splitlines(keepends=True)
    end = 0
    while end < len(lines) and re.fullmatch(
            r'#define TILEMEGA_F3_[A-Z0-9_]+ [0-9]+\n', lines[end]):
        end += 1
    if len({line.split()[1] for line in lines[:end]}) != end:
        raise ValueError('prefetch source repeats a prefix definition')
    return ''.join(sorted(lines[:end])+lines[end:])


def validate(folder, decision, final):
    from dnn4_bert_current import validate_timing_gpu
    from dnn4_bert_model_cohort import validate_bert_output_identity
    from tilemega.build.identity import verify

    def guard_on_gpu(path):
        samples = [json.loads(line) for line in
                   (Path(path).parent/'guard.jsonl').read_text().splitlines()]
        cards = [sample['gpu_index'] for sample in samples if 'gpu_index' in sample]
        if not cards or any(card != 1 for card in cards):
            raise ValueError('prefetch functional process used a different card')

    integration = read(folder/'integration.json')
    paths = ('preparation', 'original_plan', 'source_differential', 'defaults',
             'body_proof', 'split_preparation', 'split_plan', 'model_cohort',
             'model_bit_proof', 'ledger_receipt', 'trace_source_proof')
    required = {integration[key] for key in paths} | set(integration['order_builds'])
    required |= set(decision['unit_gates'])
    if integration.get('body_origin'):
        required.add(integration['body_origin'])
    if not required <= integration['receipts'].keys() or any(
            digest(path) != value for path, value in integration['receipts'].items()):
        raise ValueError('prefetch integration receipt changed or is missing')

    batch = decision['batch']
    pair = decision['pair']
    if (batch not in (4, 8) or not decision['accepted']
            or decision['minimum_gain'] != .01
            or pair['ratio'] >= 1-max(.01, pair['threshold'])
            or final['library'] != decision['library'] or final['gate'] != decision['gate']
            or final['pair'] != pair or final['unit_gates'] != decision['unit_gates']):
        raise ValueError('prefetch follow-up lacks its predeclared gain gate')
    prepared = read(integration['preparation'])
    plan = read(integration['original_plan'])
    differential = read(integration['source_differential'])
    if (prepared['status'] != 'prepared_CPU_only'
            or prepared['model']['library'] != decision['library']
            or plan['minimum_gain'] != .01
            or plan['reference']['library'] != decision['reference']
            or plan['reference']['gate'] != decision['reference_gate']
            or not differential['passed'] or not differential['coordinates_identical']):
        raise ValueError('prefetch follow-up changed its fixed preparation')
    identity = verify(Path(decision['library']))
    previous = verify(Path(decision['reference']))
    source = Path(decision['library']+'.cu').read_text()
    reference_source = Path(decision['reference']+'.cu').read_text()
    pattern = r'^#define TILEMEGA_F3_PREFETCH_DISTANCE [12]\n'
    if (re.findall(pattern, source, re.M) != ['#define TILEMEGA_F3_PREFETCH_DISTANCE 1\n']
            or re.findall(pattern, reference_source, re.M) != ['#define TILEMEGA_F3_PREFETCH_DISTANCE 2\n']
            or re.sub(pattern, '', canonical_source(source), flags=re.M) !=
               re.sub(pattern, '', canonical_source(reference_source), flags=re.M)
            or identity['trace'] or identity['plan']['runtime_batch'] != batch
            or identity['plan']['native_forward'] != previous['plan']['native_forward']
            or identity['plan']['buffers'] != previous['plan']['buffers']
            or any(row['spill'] for row in identity['resources'].values())):
        raise ValueError('prefetch changes more than the existing hint distance')
    defaults = read(integration['defaults'])
    if (not defaults['passed'] or [r['batch'] for r in defaults['rows']] != [1, 2, 4, 8, 16]
            or any(not r['passed'] or not r['selected_default_explicit_source_identical']
                   or r['encoded_words'] <= 0
                   or digest(r['reference']) != r['reference_sha256']
                   or canonical_source(Path(r['new_library']+'.cu').read_text()) !=
                      canonical_source(Path(r['reference']+'.cu').read_text())
                   for r in defaults['rows'])):
        raise ValueError('prefetch lacks its five current-profile preservation proofs')

    body = read(integration['body_proof'])
    if (not body['passed'] or body['encoded_words'] != (53488 if batch == 4 else 53280)
            or digest(body['binary']) != body['binary_sha256']
            or digest(body['reference']) != body['reference_sha256']
            or body['reused_gate'] != decision['unit_gates'][0]):
        raise ValueError('prefetch body reuse lacks its exact instruction proof')
    body_sha = body['reference_sha256']
    if batch == 4:
        origin = read(integration['body_origin'])
        if (not origin['passed'] or origin['binary'] != body['reference']
                or origin['binary_sha256'] != body_sha
                or digest(origin['reference']) != origin['reference_sha256']
                or origin['encoded_words'] != body['encoded_words']
                or origin['reused_gate'] != body['reused_gate']):
            raise ValueError('prefetch drops the original GELU fixture reuse link')
        body_sha = origin['reference_sha256']
    split = read(integration['split_preparation'])
    split_plan = read(integration['split_plan'])
    cases = ['StoreCase<128,128,false,0,0>();', 'StoreCase<128,128,true,0,0>();',
             'StoreCase<128,128,false,3,13>();', 'StoreCase<128,128,true,3,13>();']
    gates = decision['unit_gates']
    if (split['status'] != 'prepared_CPU_only' or len(gates) != 5 or len(set(gates)) != 5
            or split_plan['cases'] != cases or len(split['rows']) != 4
            or [r['build'] for r in split['rows']] != integration['order_builds']):
        raise ValueError('prefetch lacks all four prescribed visibility cases')
    original_prefix = Path(split_plan['source']).read_text().split('int main(int,char**) {', 1)[0]
    for index, path in enumerate(gates):
        gate = read(path)
        processes = gate['processes']
        if (not gate['passed'] or len(processes) != 17
                or len({p['pid'] for p in processes}) != 17
                or len({p['output_hash'] for p in processes}) != 1
                or Counter(p['name'].split('-')[0] for p in processes) !=
                   Counter(unit=5, memcheck=1, initcheck=1, racecheck=5, synccheck=5)
                or any(p['exit_code'] or read(p['guard'])['code']
                       or digest(p['log']) != p['sha256'] for p in processes)):
            raise ValueError('prefetch component cohort is incomplete')
        expected_sha = body_sha
        if index:
            build = read(integration['order_builds'][index-1])
            sources = [arg for arg in build['command'] if arg.endswith('.cu')]
            if len(sources) != 1:
                raise ValueError('prefetch visibility build has no unique source')
            text = Path(sources[0]).read_text()
            prefix, main = text.split('int main(int,char**) {', 1)
            if (gate['gpu'] != 1 or build['profile'] != prepared['profile']
                    or build['spill_bytes'] != 0
                    or any(f'-D{k}={v}' not in build['command']
                           for k, v in prepared['profile'].items())
                    or digest(build['binary']) != build['binary_sha256']
                    or prefix != original_prefix or main.count(cases[index-1]) != 1
                    or any(case in main for case in cases if case != cases[index-1])):
                raise ValueError('prefetch visibility build changed its profile or case')
            expected_sha = build['binary_sha256']
        if any(p['binary_sha256'] != expected_sha for p in processes):
            raise ValueError('prefetch component reused a different binary')
        if index:
            for guard in {p['guard'] for p in processes}:
                guard_on_gpu(guard)

    model = read(integration['model_cohort'])
    bit = read(integration['model_bit_proof'])
    if (not model['passed'] or model['passes'] != '5/5' or model['gpu'] != 1
            or model['execution']['artifact_id'] != identity['artifact_id']
            or model['gate'] != decision['gate'] or len(model['processes']) != 5
            or len({p['pid'] for p in model['processes']}) != 5
            or any(p['exit_code'] or read(p['guard'])['code'] for p in model['processes'])):
        raise ValueError('prefetch model cohort is incomplete')
    named = validate_bert_output_identity([read(p['check']) for p in model['processes']],
                                          read(decision['reference_gate']), identity['artifact_id'])
    for guard in {p['guard'] for p in model['processes']}:
        guard_on_gpu(guard)
    if (not bit['passed'] or bit['reference'] != decision['reference_gate']
            or bit['named_output_sha256'] != named['named_output_sha256']
            or bit['candidate'] != integration['model_cohort']):
        raise ValueError('prefetch lost a named output byte identity')
    allocation = read(folder/'allocation.json')
    authorization = read(allocation['snapshot'])
    if (allocation['gpu'] != 1 or allocation['original_anchor_gpu'] != 3
            or allocation['reanchored'] or allocation['python_entry_root'] != '/root/python'
            or authorization['allowed_gpu_indices'] != [1]
            or authorization['python_entry_root'] != '/root/python'
            or digest(allocation['snapshot']) != allocation['snapshot_sha256']):
        raise ValueError('prefetch changes its authorized card or fixed target')
    pids = []
    reference_measurement = read(pair['measurements'][0]['measurement'])
    for measured, order in ((pair, 'ABABAB'), (final['measurement'], 'BBB')):
        validate_timing_gpu(measured, 1, order)
        for row in measured['measurements']:
            raw = read(row['measurement'])
            candidate = row['arm'] == 'B'
            expected = identity if candidate else previous
            expected_gate = decision['gate'] if candidate else decision['reference_gate']
            if (raw['artifact_id'] != expected['artifact_id']
                    or Path(raw['library_path']).resolve() != Path(expected['binary_path']).resolve()
                    or Path(raw['numerical_gate_path']).resolve() != Path(expected_gate).resolve()
                    or raw['batch'] != batch or raw['precision'] != 'bf16'
                    or any(raw[key] != reference_measurement[key]
                           for key in ('random_weights', 'accuracy_eligible'))
                    or raw['source_checkpoint_sha256'] !=
                       read(decision['reference_gate'])['source_checkpoint_sha256']
                    or raw['input_sha256'] != pair['input_sha256']
                    or row['median_ms'] != statistics.median(raw['samples_ms'])):
                raise ValueError('prefetch timing substitutes an artifact, input or sample')
            boundaries = [json.loads(line) for line in Path(raw['guard_log']).read_text().splitlines()]
            for label, clock in [('before-measurement', 'start_sm_clock_mhz'),
                                 ('after-measurement', 'end_sm_clock_mhz')]:
                boundary = next(v for v in boundaries if v.get('label') == label)
                if (boundary['time'] < authorization['current_gpu_restriction_at']
                        or 'clocks.sm' not in boundary.get('telemetry', {})
                        or float(boundary['telemetry']['clocks.sm']) != row[clock]):
                    raise ValueError('prefetch timing lacks its actual authorized clock boundaries')
            pids.append(raw['pid'])
    a = [r['median_ms'] for r in pair['measurements'] if r['arm'] == 'A']
    b = [r['median_ms'] for r in pair['measurements'] if r['arm'] == 'B']
    threshold = max(.005, 2*max((max(v)-min(v))/statistics.median(v) for v in (a, b)))
    values = [r['median_ms'] for r in final['measurement']['measurements']]
    events = sorted(v for r in final['measurement']['measurements']
                    for v in read(r['measurement'])['samples_ms'])
    if (len(set(pids)) != 9 or pair['candidate_ms'] != statistics.median(b)
            or pair['reference_ms'] != statistics.median(a)
            or pair['ratio'] != statistics.median(b)/statistics.median(a)
            or pair['threshold'] != threshold
            or final['measurement']['candidate_ms'] != statistics.median(values)
            or final['measurement']['process_range_ms'] != max(values)-min(values)
            or final['measurement']['p90_ms'] != events[(9*len(events)+9)//10-1]):
        raise ValueError('prefetch aggregate changes the independent measured statistics')
    ledger = read(integration['ledger_receipt'])
    trace = read(integration['trace_source_proof'])
    traced = verify(Path(trace['library']))
    stages = list(csv.DictReader(Path(final['ledger']).open()))
    if (ledger['gpu'] != 1 or ledger['trace_level'] != 1 or not trace['passed']
            or not traced['trace'] or traced['artifact_id'] != ledger['artifact_id']
            or ledger['reference'] != decision['library']
            or ledger['reference_sha256'] != digest(decision['library'])
            or ledger['ledger'] != final['ledger']
            or sorted(int(r['stage']) for r in stages) != list(range(63))
            or trace['reference'] != decision['library']
            or trace['reference_sha256'] != digest(decision['library'])
            or Path(trace['library']+'.cu').read_text().replace(
                '#define TILEMEGA_TRACE_STAGE 1', '#define TILEMEGA_TRACE_STAGE 0') != source):
        raise ValueError('prefetch lacks its exact production-binary ledger')
    return 1
