#!/usr/bin/env python3
"""Export current admitted BERT artifacts without replacing frozen Phase Z."""
import csv
import json
from pathlib import Path
import statistics
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
from tilemega.build.identity import verify
from tilemega.build.feature_contract import build_defines


def read(path):
    return json.loads(Path(path).read_text())


def main():
    base = ROOT/'runs/dnn4/bert_priority'
    targets = {int(r['batch']): float(r['baseline_ms']) for r in csv.DictReader(
        (ROOT/'docs/results/DNN3_final.csv').open()) if r['model'] == 'bert'}
    followups = {
        1: ('repeat_single_models_v1/b1/final_gpu3_v1/result.json',
            'repeat_single_models_v1/b1/g22_v3/experiment.json',
            {'TILEMEGA_F3_REPEAT': '1', 'TILEMEGA_F3_REPEAT_LAYERS': '1'}, .03),
        2: ('repeat_arithmetic_models_v1/b2/g22_followup_final_v1/final_result.json',
            'repeat_arithmetic_models_v1/b2/experiment.json',
            {'TILEMEGA_F3_REPEAT': '1', 'TILEMEGA_F3_REPEAT_LAYERS': '1',
             'TILEMEGA_F3_REGISTER_BUDGET': '1',
             'TILEMEGA_F3_REPEAT_ARITHMETIC': '1'}, .03),
        4: ('scalar_phase72_arithmetic_b4_v1/final_result.json',
            'scalar_phase72_arithmetic_b4_v1/experiment.json',
            {'TILEMEGA_F3_SIDE_NORM': '1', 'TILEMEGA_F3_REPEAT': '1',
             'TILEMEGA_F3_REPEAT_LAYERS': '1',
             'TILEMEGA_F3_REPEAT_ARITHMETIC': '1',
             'TILEMEGA_F3_REGISTER_BUDGET': '2'}, 0),
        8: ('scalar_phase72_arithmetic_b8_v1/final_result.json',
            'scalar_phase72_arithmetic_b8_v1/experiment.json',
            {'TILEMEGA_F3_SIDE_NORM': '1', 'TILEMEGA_F3_REPEAT': '1',
             'TILEMEGA_F3_REPEAT_LAYERS': '1',
             'TILEMEGA_F3_REPEAT_ARITHMETIC': '1',
             'TILEMEGA_F3_REGISTER_BUDGET': '2'}, .03),
        16: ('side_norm_models_v2/b16/g22_v1/final_result.json',
             'side_norm_models_v2/b16/g22_v1/experiment.json',
             {'TILEMEGA_F3_SIDE_NORM': '1'}, 0)}
    rows = []
    for batch in (1, 2, 4, 8, 16):
        if batch in followups:
            final_name, pair_name, profile, minimum_gain = followups[batch]
            source = base/final_name
            final = read(source)
            library, gate = Path(final['library']), Path(final['gate'])
            measurement = final['measurement']
            decision = read(base/pair_name)
            pair = decision['pair']
            threshold = max(pair['threshold'], minimum_gain)
            assert decision['accepted'] and pair['ratio'] < 1-threshold
            assert pair['gpu'] == 3 and len(pair['measurements']) == 6
            assert all(m['guard']['code'] == 0 for m in pair['measurements'])
            pair_source = str(base/pair_name)
            scope = 'BERT follow-up;original GPU3 anchor retained;not re-anchored'
        else:
            source = ROOT/f'runs/dnn4/phase_z/cells_v1/bert_b{batch}/result.json'
            final = read(source)
            library, gate = Path(final['winner']['library']), Path(final['winner']['gate'])
            measurement = read(final.get('final_measurement', source.parent/'final/result.json'))
            profile, pair, pair_source = {}, None, ''
            scope = 'unchanged original anchored Phase Z'
        identity = verify(library)
        assert not identity['trace'] and identity['plan']['runtime_batch'] == batch
        source_text = Path(str(library)+'.cu').read_text()
        defines = build_defines(source_text, [])
        for key, value in profile.items():
            if key not in ('TILEMEGA_F3_REPEAT', 'TILEMEGA_F3_REPEAT_LAYERS'):
                assert defines.get(key) == value
        if profile.get('TILEMEGA_F3_REPEAT') == '1':
            assert re.search(r',Stage_0,Repeat<12,Loop_0,Loop_1,Loop_2,Loop_3,Loop_4>,Stage_61,Stage_62>', source_text)
        checked = read(gate.parent.parent/'result.json')
        assert checked['status'] == 'passed' and checked['passes'] == '5/5'
        assert all(o.get('output_sha256') for c in checked['cases'] for o in c['outputs'])
        measured = measurement['measurements']
        assert measurement['protocol'] == 'P-DNN' and measurement['gpu'] == 3
        assert len(measured) == 3 and all(m['arm'] == 'B' and m['guard']['code'] == 0 for m in measured)
        assert statistics.median(m['median_ms'] for m in measured) == measurement['candidate_ms']
        for m in measured:
            process = read(m['measurement'])
            assert Path(process['numerical_gate_path']).resolve() == gate.resolve()
        ms = measurement['candidate_ms']
        if batch == 1:
            ledger = read(base/'repeat_single_models_v1/b1/ledger_gpu3_v1/result.json')['ledger']
        elif batch == 16:
            ledger = read(base/'side_norm_models_v2/b16/ledger_gpu3_v1/result.json')['ledger']
        else:
            ledger = final['ledger']
        ledger = Path(ledger)
        if ledger.is_dir():
            ledger = ledger/'ledger.csv'
        trace = read(ledger.parent/'result.json')
        assert trace['trace_level'] == 1 and trace['gpu'] == 3
        assert trace['reference_sha256'] == identity['binary_sha256']
        rows.append(dict(model='bert', batch=batch, gpu=3, median_ms=ms,
            p90_ms=measurement['p90_ms'], process_range_ms=measurement['process_range_ms'],
            baseline_ms=targets[batch], ratio=ms/targets[batch],
            faster_than_baseline=ms < targets[batch],
            profile=json.dumps(profile, sort_keys=True, separators=(',', ':')),
            pair_ratio=pair['ratio'] if pair else '',
            pair_threshold=pair['threshold'] if pair else '',
            artifact_id=identity['artifact_id'], binary_sha256=identity['binary_sha256'],
            source_digest=identity['source']['source_digest'],
            clocks=json.dumps([{k: m[k] for k in ('start_sm_clock_mhz', 'end_sm_clock_mhz')}
                               for m in measured], separators=(',', ':')),
            correctness='strict FP32 P-CORR5/5;cross-process output bytes identical',
            evidence='verified', scope=scope, qualification=measurement['qualification'],
            library=str(library), gate=str(gate), pair_source=pair_source, source=str(source),
            ledger_source=str(ledger)))
    output = ROOT/'docs/results/DNN4_BERT_current.csv'
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(evidence='verified', rows=5, output=str(output),
                          medians_ms={r['batch']: r['median_ms'] for r in rows})))


if __name__ == '__main__':
    main()
