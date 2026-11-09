#!/usr/bin/env python3
"""Export current admitted BERT artifacts without replacing frozen Phase Z."""
import argparse
from collections import Counter
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


def select_successors(base, batch, specification):
    final_name, pair_name, profile, minimum_gain = specification
    library = Path(read(base/final_name)['library'])
    names = ([('matrix_integrated_b16_v1', {'TILEMEGA_F3_MATRIX_STORE': '1'}),
              ('matrix_existing_retune_b16_v1', {'TILEMEGA_F3_MATRIX_STORE': '1'})]
             if batch == 16 else [])
    if batch in (4, 8, 16):
        common = {'TILEMEGA_F3_MATRIX_STORE': '1'} if batch == 16 else {}
        names += [(f'qk_wgmma_b{batch}_v1', dict(common, TILEMEGA_F3_WGMMA_QK='1')),
                  (f'pv_wgmma_b{batch}_v1', dict(common, TILEMEGA_F3_WGMMA_QK='1',
                                               TILEMEGA_F3_WGMMA_PV='1'))]
    if batch == 16:
        names.append(('attention_existing_retune_b16_v1', {}))
        names.append(('ppstore_ffn1_b16_v1', {'TILEMEGA_F3_MATRIX_STORE': '1'}))
    for name, features in names:
        folder = base/name
        if not (folder/'experiment.json').exists():
            continue
        decision = read(folder/'experiment.json')
        if not decision['accepted'] or not (folder/'final_result.json').exists():
            continue
        final = read(folder/'final_result.json')
        if Path(decision['reference']).resolve() != library.resolve():
            raise ValueError('follow-up was not compared against the current incumbent: '+name)
        if (Path(final['library']).resolve() != Path(decision['library']).resolve() or
                Path(final['gate']).resolve() != Path(decision['gate']).resolve()):
            raise ValueError('follow-up final differs from its paired artifact: '+name)
        expected = Counter(unit=5, memcheck=1, initcheck=1, racecheck=5, synccheck=5)
        units = decision.get('unit_gates', [])
        candidate_profile = dict(profile, **features)
        count = ((2 if batch == 16 else 0) +
                 int(candidate_profile.get('TILEMEGA_F3_WGMMA_QK') == '1') +
                 int(candidate_profile.get('TILEMEGA_F3_WGMMA_PV') == '1') +
                 (2 if name == 'ppstore_ffn1_b16_v1' else 0))
        if (len(units) != count or len(set(map(str, units))) != count or
                set(map(str, final.get('unit_gates', []))) != set(map(str, units)) or
                final['pair'] != decision['pair']):
            raise ValueError('follow-up lacks its complete component receipts: '+name)
        for unit_path in units:
            unit = read(unit_path)
            processes = unit['processes']
            if (not unit['passed'] or len(processes) != 17 or
                    len({p['pid'] for p in processes}) != 17 or
                    Counter(p['name'].split('-')[0] for p in processes) != expected or
                    any(p['exit_code'] or read(p['guard'])['code'] for p in processes) or
                    len({p['binary_sha256'] for p in processes}) != 1 or
                    len({p['output_hash'] for p in processes}) != 1):
                raise ValueError('follow-up component protocol is incomplete: '+name)
        if name in ('matrix_existing_retune_b16_v1', 'attention_existing_retune_b16_v1'):
            qualification = read(folder/'trace_qualification.json')
            selection = read(folder/'tune_selection.json')
            if (not qualification['passed'] or abs(qualification['relative_difference']) > .05
                    or len(selection['eligible_variants']) < 4):
                raise ValueError('follow-up used unqualified in-situ windows')
        pair = decision['pair']
        required_gain = .03 if name == 'ppstore_ffn1_b16_v1' else 0
        if (pair['gpu'] != 3 or len(pair['measurements']) != 6 or
                any(m['guard']['code'] for m in pair['measurements']) or
                pair['ratio'] >= 1-max(required_gain, pair['threshold'])):
            raise ValueError('follow-up did not clear its complete same-card pair: '+name)
        if name == 'ppstore_ffn1_b16_v1':
            validate_pp_coordinates(final['library'], decision['reference'])
        # Completed raw prototypes never appear in this ordered admission list.
        final_name = name+'/final_result.json'
        pair_name = name+'/experiment.json'
        profile = candidate_profile
        minimum_gain = required_gain
        library = Path(final['library'])
    return final_name, pair_name, profile, minimum_gain


def validate_pp_coordinates(library, reference):
    current = read(str(reference)+'.plan.json')['native_forward']
    candidate = read(str(library)+'.plan.json')['native_forward']
    expected = json.loads(json.dumps(current))
    if len(expected['stages']) != 63:
        raise ValueError('PP follow-up requires the complete BERT program')
    for index in range(4, 60, 5):
        stage = expected['stages'][index]
        fixed = dict(tile_m=128, tile_n=128, tk=64, slots=5, split_k=1, store='tma', mc=0)
        if any(stage.get(k) != v for k, v in fixed.items()) or stage.get('sched', 'coop') != 'coop':
            raise ValueError('PP follow-up changed its fixed FFN1 control')
        stage.update(tile_m=256, sched='pp')
    if candidate != expected:
        raise ValueError('PP follow-up changed coordinates outside its fixed FFN1 combination')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'docs/results/DNN4_BERT_current.csv')
    parser.add_argument('--ledger-out', type=Path)
    args = parser.parse_args()
    base = ROOT/'runs/dnn4/bert_priority'
    targets = {int(r['batch']): float(r['baseline_ms']) for r in csv.DictReader(
        (ROOT/'docs/results/DNN3_final.csv').open()) if r['model'] == 'bert'}
    followups = {
        1: ('coalesced_cyclic_b1_integrated_v1/final_result.json',
            'coalesced_cyclic_b1_integrated_v1/experiment.json',
            {'TILEMEGA_F3_REPEAT': '1', 'TILEMEGA_F3_REPEAT_LAYERS': '1',
             'TILEMEGA_F3_CYCLIC_SLOTS': '1',
             'TILEMEGA_F3_COALESCED_PARTIALS': '1'}, 0),
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
        16: ('cyclic216_b16_integrated_v1/final_result.json',
             'cyclic216_b16_integrated_v1/experiment.json',
             {'TILEMEGA_F3_SIDE_NORM': '1', 'TILEMEGA_F3_REPEAT': '1',
              'TILEMEGA_F3_REPEAT_LAYERS': '1',
              'TILEMEGA_F3_REPEAT_ARITHMETIC': '1',
              'TILEMEGA_F3_REGISTER_BUDGET': '3',
              'TILEMEGA_F3_CYCLIC_SLOTS': '1'}, .03)}
    rows = []
    for batch in (1, 2, 4, 8, 16):
        if batch in followups:
            final_name, pair_name, profile, minimum_gain = select_successors(
                base, batch, followups[batch])
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
        assert not any(r['spill'] for r in identity['resources'].values())
        source_text = Path(str(library)+'.cu').read_text()
        defines = build_defines(source_text, [])
        for key, value in profile.items():
            if key not in ('TILEMEGA_F3_REPEAT', 'TILEMEGA_F3_REPEAT_LAYERS'):
                assert defines.get(key) == value
                if key in ('TILEMEGA_F3_MATRIX_STORE', 'TILEMEGA_F3_WGMMA_QK',
                           'TILEMEGA_F3_WGMMA_PV'):
                    assert ('std::getenv("'+key+'")') in (
                        ROOT/'lib/Codegen/ForwardF3Codegen.cpp').read_text()
        if profile.get('TILEMEGA_F3_REPEAT') == '1':
            assert re.search(r',Stage_0,Repeat<12,Loop_0,Loop_1,Loop_2,Loop_3,Loop_4>,Stage_61,Stage_62>', source_text)
        checked = read(gate.parent.parent/'result.json')
        assert checked['status'] == 'passed' and checked['passes'] == '5/5'
        assert len(checked['processes']) == 5
        assert all(p['exit_code'] == 0 for p in checked['processes'])
        assert all(o.get('output_sha256') for c in checked['cases'] for o in c['outputs'])
        measured = measurement['measurements']
        assert measurement['protocol'] == 'P-DNN' and measurement['gpu'] == 3
        assert len(measured) == 3 and all(m['arm'] == 'B' and m['guard']['code'] == 0 for m in measured)
        assert statistics.median(m['median_ms'] for m in measured) == measurement['candidate_ms']
        for m in measured:
            process = read(m['measurement'])
            assert Path(process['numerical_gate_path']).resolve() == gate.resolve()
        ms = measurement['candidate_ms']
        if 'ledger' in final:
            ledger = final['ledger']
        elif batch == 1:
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
    output = args.out
    if output.exists():
        previous = {int(r['batch']): r for r in csv.DictReader(output.open())}
        for row in rows:
            old = previous.get(row['batch'], {})
            if old.get('artifact_id') == row['artifact_id']:
                for key in ('library', 'gate', 'pair_source', 'source', 'ledger_source'):
                    if old.get(key) and Path(old[key]).resolve() == Path(row[key]).resolve():
                        row[key] = old[key]
    ledger_rows = []
    for row in rows:
        coordinates = read(row['library']+'.plan.json')['native_forward']['stages']
        with Path(row['ledger_source']).open() as stream:
            stages = list(csv.DictReader(stream))
        assert sorted(int(s['stage']) for s in stages) == list(range(len(coordinates)))
        for stage in stages:
            ledger_rows.append(dict(model='bert', batch=row['batch'], gpu=3,
                artifact_id=row['artifact_id'], binary_sha256=row['binary_sha256'],
                profile=row['profile'], **stage,
                coordinate=json.dumps(coordinates[int(stage['stage'])],
                                      sort_keys=True, separators=(',', ':')),
                evidence='verified exact accepted reference P-TRACE1;attribution only',
                source=row['ledger_source']))
    ledger_output = args.ledger_out or output.with_name(output.stem+'_ledger.csv')
    with ledger_output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(ledger_rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(ledger_rows)
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(evidence='verified', rows=5, output=str(output),
                          ledger=str(ledger_output), ledger_rows=len(ledger_rows),
                          medians_ms={r['batch']: r['median_ms'] for r in rows})))


if __name__ == '__main__':
    main()
