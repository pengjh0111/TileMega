#!/usr/bin/env python3
"""Export current admitted BERT artifacts without replacing frozen Phase Z."""
import argparse
import copy
from collections import Counter
import csv
import hashlib
import json
import math
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
        names.append(('ppnorm_ffn1_b16_v1', {'TILEMEGA_F3_MATRIX_STORE': '1',
                                            'TILEMEGA_F3_PP_SIDE_NORM': '1'}))
        names.append(('pp64_ffn1_b16_v1', {'TILEMEGA_F3_MATRIX_STORE': '1',
                                          'TILEMEGA_F3_PP_SIDE_NORM': '1'}))
    if batch in (8, 16):
        names.append((f'attention_store_b{batch}_v1',
                      {'TILEMEGA_F3_ATTENTION_VECTOR_STORE': '1'}))
    if batch == 16:
        names.append(('fixed_grid_b16_admitted_v1', {'TILEMEGA_F3_H800_GRID': '1'}))
    if batch in (4, 8, 16):
        names.append((f'attention_bundle_b{batch}_v1',
                      {'TILEMEGA_F3_ATTENTION_VECTOR_STORE': '1',
                       'TILEMEGA_F3_WGMMA_QK': '1', 'TILEMEGA_F3_WGMMA_PV': '1'}))
    if batch in (8, 16):
        names.append((f'ffn1_tma192_matrix_b{batch}_v1',
                      {'TILEMEGA_F3_MATRIX_STORE': '1',
                       'TILEMEGA_F3_REGISTER_BUDGET': '3'}))
    if batch == 8:
        names.append(('ffn1_tma128_matrix_b8_v1', {'TILEMEGA_F3_MATRIX_STORE': '1'}))
        names.append(('whole_program216_b8_admitted_v1',
                      {'TILEMEGA_F3_REGISTER_BUDGET': '3'}))
        names.append(('grid_recount_b8_admitted_v1',
                      {'TILEMEGA_F3_H800_GRID': '1',
                       'TILEMEGA_F3_RECOMPUTE_TASK': '1'}))
    if batch == 16:
        names.append(('ffn2_tma256x64_b16_v1',
                      {'TILEMEGA_F3_MATRIX_STORE': '1',
                       'TILEMEGA_F3_REGISTER_BUDGET': '3'}))
    if batch == 16:
        names.append(('ffn1_tma256_matrix_b16_v2', {}))
        names.append(('qkv_tma192_matrix_b16_v1', {}))
    if batch == 4:
        names.append(('cyclic_phase72_b4_v1', {'TILEMEGA_F3_CYCLIC_SLOTS': '1'}))
        names.append(('ffn1_tma128_phase72_b4_admitted_v1', {}))
        names.append(('boundary_cache216_b4_admitted_v1',
                      {'TILEMEGA_F3_REGISTER_BUDGET': '3'}))
        names.append(('grid_recount_b4_admitted_v1',
                      {'TILEMEGA_F3_H800_GRID': '1',
                       'TILEMEGA_F3_RECOMPUTE_TASK': '1'}))
    if batch == 2:
        names.append(('ffn1_reg64_side216_b2_admitted_v1',
                      {'TILEMEGA_F3_SIDE_NORM': '1',
                       'TILEMEGA_F3_REGISTER_BUDGET': '3'}))
        names.append(('ffn2_reg64_split4_b2_v1',
                      {'TILEMEGA_F3_SIDE_NORM': '1',
                       'TILEMEGA_F3_REGISTER_BUDGET': '3'}))
    if batch in (4, 8, 16):
        features = {'TILEMEGA_F3_FULL_TILES': '1'}
        if batch == 16:
            features['TILEMEGA_F3_REPEAT_SELECTOR'] = '1'
        names.append((f'full_tiles_b{batch}_admitted_v1', features))
    if batch == 8:
        names.append(('full_store_bundle_b8_admitted_v1',
                      {'TILEMEGA_F3_MATRIX_STORE': '1',
                       'TILEMEGA_F3_ATTENTION_VECTOR_STORE': '1'}))
    if batch == 2:
        names.append(('pipeline_reduction_b2_admitted_v1',
                      {'TILEMEGA_F3_FULL_TILES': '1',
                       'TILEMEGA_F3_H800_GRID': '1',
                       'TILEMEGA_F3_RECOMPUTE_TASK': '1',
                       'TILEMEGA_F3_COALESCED_PARTIALS': '1',
                       'TILEMEGA_F3_CYCLIC_SLOTS': '1'}))
    if batch in (4, 8, 16):
        names.append((f'gelu_outline_b{batch}_admitted_v1', {'TILEMEGA_F3_GELU_OUTLINE': '1'}))
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
                 (2 if name == 'ppstore_ffn1_b16_v1' else
                  4 if name == 'ppnorm_ffn1_b16_v1' else
                  6 if name == 'pp64_ffn1_b16_v1' else 0) +
                 int(candidate_profile.get('TILEMEGA_F3_ATTENTION_VECTOR_STORE') == '1') +
                 (2 if name == 'fixed_grid_b16_admitted_v1' else 0) +
                 (2 if name in ('grid_recount_b4_admitted_v1',
                                'grid_recount_b8_admitted_v1') else 0) +
                 int(name.startswith('attention_bundle_')) +
                 (2 if name == 'cyclic_phase72_b4_v1' else 0) +
                 int(name == 'ffn1_tma128_phase72_b4_admitted_v1') +
                 int(name == 'ffn1_tma256_matrix_b16_v2') +
                 int(name == 'qkv_tma192_matrix_b16_v1') +
                 (2 if name == 'boundary_cache216_b4_admitted_v1' else 0) +
                 (2 if name == 'whole_program216_b8_admitted_v1' else 0) +
                 (3 if name == 'ffn1_tma128_matrix_b8_v1' else 0) +
                 (3 if name == 'ffn1_reg64_side216_b2_admitted_v1' else 0) +
                 (4 if name == 'ffn2_reg64_split4_b2_v1' else 0) +
                 (2 if name == 'ffn2_tma256x64_b16_v1' else 0) +
                 (2 if batch == 8 and name.startswith('ffn1_tma192_matrix_') else 0) +
                 int(name.startswith('ffn1_tma192_matrix_') and all(
                     candidate_profile.get(k) == '1' for k in
                     ('TILEMEGA_F3_WGMMA_QK', 'TILEMEGA_F3_WGMMA_PV',
                      'TILEMEGA_F3_ATTENTION_VECTOR_STORE'))))
        if name == f'full_tiles_b{batch}_admitted_v1':
            count = 4 if batch == 16 else 2
        if name == 'full_store_bundle_b8_admitted_v1':
            count = 3
        if name in ('pipeline_reduction_b2_admitted_v1',
                    'gelu_outline_b4_admitted_v1', 'gelu_outline_b8_admitted_v1',
                    'gelu_outline_b16_admitted_v1'):
            count = 1
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
        required_gain = .03 if name in ('ppstore_ffn1_b16_v1', 'pp64_ffn1_b16_v1',
                                      'gelu_outline_b4_admitted_v1', 'gelu_outline_b8_admitted_v1',
                    'gelu_outline_b16_admitted_v1') else 0
        if name == 'ppnorm_ffn1_b16_v1':
            reference_stages = read(decision['reference']+'.plan.json')['native_forward']['stages']
            required_gain = 0 if reference_stages[4].get('sched') == 'pp' else .03
        gpu = validate_followup_allocation(folder, name, decision, final)
        if (pair['gpu'] != gpu or len(pair['measurements']) != 6 or
                any(m['guard']['code'] for m in pair['measurements']) or
                pair['ratio'] >= 1-max(required_gain, pair['threshold'])):
            raise ValueError('follow-up did not clear its complete same-card pair: '+name)
        if name in ('ppstore_ffn1_b16_v1', 'ppnorm_ffn1_b16_v1'):
            validate_pp_coordinates(final['library'], decision['reference'],
                                    allow_existing=name == 'ppnorm_ffn1_b16_v1')
        if name == 'pp64_ffn1_b16_v1':
            validate_pp_coordinates(final['library'], decision['reference'], wg_rows=64)
        if (name.startswith(('attention_store_', 'attention_bundle_')) or
                name in ('cyclic_phase72_b4_v1', 'fixed_grid_b16_admitted_v1',
                         'grid_recount_b4_admitted_v1', 'grid_recount_b8_admitted_v1')):
            if (read(final['library']+'.plan.json')['native_forward'] !=
                    read(decision['reference']+'.plan.json')['native_forward']):
                raise ValueError('follow-up changed the current coordinates')
        if name == 'fixed_grid_b16_admitted_v1':
            if final_name != 'attention_store_b16_v1/final_result.json':
                raise ValueError('fixed grid lacks its accepted predecessor')
            predecessor = read(base/pair_name)
            if units[:3] != predecessor['unit_gates']:
                raise ValueError('fixed grid dropped inherited component gates')
            for unit in units[3:]:
                validate_component_profile(unit, {
                    '-DTILEMEGA_F3_H800_GRID=1', '-DTILEMEGA_F3_REGISTER_BUDGET=3',
                    '-DTILEMEGA_F3_SIDE_NORM=1', '-DTILEMEGA_F3_MATRIX_STORE=1',
                    '-DTILEMEGA_F3_ATTENTION_VECTOR_STORE=1'})
            validate_fixed_grid_integration(folder, decision, units[3:])
            validate_timing_gpu(pair, 3, 'ABABAB')
            validate_timing_gpu(final['measurement'], 3, 'BBB')
        if name in ('grid_recount_b4_admitted_v1', 'grid_recount_b8_admitted_v1'):
            if final_name != f'scalar_phase72_arithmetic_b{batch}_v1/final_result.json':
                raise ValueError('producer recomputation lacks its accepted predecessor')
            for unit in units:
                validate_component_profile(unit, {
                    '-DTILEMEGA_F3_H800_GRID=1', '-DTILEMEGA_F3_RECOMPUTE_TASK=1',
                    '-DTILEMEGA_F3_REGISTER_BUDGET=2', '-DTILEMEGA_F3_SIDE_NORM=1',
                    '-DTILEMEGA_F3_REPEAT_ARITHMETIC=1'})
            validate_fixed_grid_integration(folder, decision, units)
            validate_recompute_defaults(folder)
            validate_timing_gpu(pair, 3, 'ABABAB')
            validate_timing_gpu(final['measurement'], 3, 'BBB')
        if name == 'ffn1_tma128_phase72_b4_admitted_v1':
            validate_ffn1_component_profile(units[0], 128)
            validate_ffn1_tma_coordinates(final['library'], decision['reference'], 128, 5)
        if name in ('boundary_cache216_b4_admitted_v1',
                    'whole_program216_b8_admitted_v1'):
            validate_component_profile(units[0], {
                '-DDNN4_ALIAS_REPEAT_TEST=1', '-DTILEMEGA_F3_REGISTER_BUDGET=3',
                '-DTILEMEGA_F3_REPEAT_ARITHMETIC=1'})
            validate_component_profile(units[1], {
                '-DTILEMEGA_F3_SIDE_NORM=1', '-DTILEMEGA_F3_REGISTER_BUDGET=3'})
            if name == 'boundary_cache216_b4_admitted_v1':
                validate_boundary_selection(folder, final['library'], decision['reference'])
            else:
                validate_whole_program_selection(folder, final['library'], decision['reference'])
                validate_timing_gpu(pair, gpu, 'ABABAB')
                validate_timing_gpu(final['measurement'], gpu, 'BBB')
        if name == 'ffn1_tma256_matrix_b16_v2':
            validate_ffn1_component_profile(units[0], 256)
            validate_ffn1_tma_coordinates(final['library'], decision['reference'], 256, 3)
        if name == 'ffn1_tma128_matrix_b8_v1':
            validate_component_profile(units[0], {
                '-DDNN4_FFN1_TMA128_TEST=1', '-DDNN4_FFN1_TMA128_B8_TEST=1',
                '-DTILEMEGA_F3_SIDE_NORM=1', '-DTILEMEGA_F3_REGISTER_BUDGET=2',
                '-DTILEMEGA_F3_MATRIX_STORE=1'})
            validate_ffn1_tma_coordinates(final['library'], decision['reference'], 128, 5)
        if name == 'qkv_tma192_matrix_b16_v1':
            validate_component_profile(units[0], {
                '-DDNN4_QKV_TMA192_TEST=1', '-DTILEMEGA_F3_SIDE_NORM=1',
                '-DTILEMEGA_F3_REGISTER_BUDGET=3',
                '-DTILEMEGA_F3_MATRIX_STORE=1', '-DTILEMEGA_F3_CYCLIC_SLOTS=1'})
            validate_qkv_tma192_coordinates(final['library'], decision['reference'])
        if name.startswith('ffn1_tma192_matrix_'):
            validate_ffn1_tma192_coordinates(final['library'], decision['reference'])
        if name == 'ffn2_tma256x64_b16_v1':
            validate_ffn2_tma256x64_coordinates(final['library'], decision['reference'])
        if name == 'ffn1_reg64_side216_b2_admitted_v1':
            validate_ffn1_reg64_coordinates(final['library'], decision['reference'])
        if name == 'ffn2_reg64_split4_b2_v1':
            if final_name != 'ffn1_reg64_side216_b2_admitted_v1/final_result.json':
                raise ValueError('FFN2 follow-up lacks its admitted FFN1 predecessor')
            validate_ffn2_reg64_coordinates(final['library'], decision['reference'])
        if name == f'full_tiles_b{batch}_admitted_v1':
            validate_full_tiles_integration(folder, decision, final, batch)
        if name == 'full_store_bundle_b8_admitted_v1':
            if final_name != 'full_tiles_b8_admitted_v1/final_result.json':
                raise ValueError('store bundle lacks its admitted full-tile predecessor')
            validate_store_bundle_integration(folder, decision, final)
        if name == 'pipeline_reduction_b2_admitted_v1':
            if final_name != 'ffn1_reg64_side216_b2_admitted_v1/final_result.json':
                raise ValueError('pipeline reduction lacks its admitted B2 predecessor')
            validate_pipeline_reduction_integration(folder, decision, final)
        if name == f'gelu_outline_b{batch}_admitted_v1':
            predecessor = ('full_store_bundle_b8_admitted_v1' if batch == 8 else
                           f'full_tiles_b{batch}_admitted_v1')
            if final_name != predecessor+'/final_result.json':
                raise ValueError('outlined GELU lacks its admitted predecessor')
            validate_gelu_outline_integration(folder, decision, final)
        # Completed raw prototypes never appear in this ordered admission list.
        final_name = name+'/final_result.json'
        pair_name = name+'/experiment.json'
        profile = candidate_profile
        minimum_gain = required_gain
        library = Path(final['library'])
    return final_name, pair_name, profile, minimum_gain


def validate_store_bundle_integration(folder, decision, final):
    validate_fixed_feature_integration(folder, decision, final,
        ['body', 'order', 'attention'], '2',
        ['TILEMEGA_F3_MATRIX_STORE', 'TILEMEGA_F3_ATTENTION_VECTOR_STORE',
         'TILEMEGA_F3_SIDE_NORM', 'TILEMEGA_F3_FULL_TILES',
         'TILEMEGA_F3_H800_GRID', 'TILEMEGA_F3_RECOMPUTE_TASK'])


def validate_pipeline_reduction_integration(folder, decision, final):
    validate_fixed_feature_integration(folder, decision, final, ['body'], '3',
        ['TILEMEGA_F3_SIDE_NORM', 'TILEMEGA_F3_FULL_TILES',
         'TILEMEGA_F3_H800_GRID', 'TILEMEGA_F3_RECOMPUTE_TASK',
         'TILEMEGA_F3_COALESCED_PARTIALS', 'TILEMEGA_F3_CYCLIC_SLOTS'])
    integration = read(folder/'integration.json')
    bit = read(integration['unit_bit_proof'])
    control = read(bit['control'])
    candidate = read(decision['unit_gates'][0])
    if (not bit['passed'] or not control['passed']
            or len(control['processes']) != 5
            or len({p['pid'] for p in control['processes']}) != 5
            or any(p['exit_code'] or read(p['guard'])['code']
                   for p in control['processes'])
            or bit['output_hash'] != candidate['output_hash']
            or candidate['output_hash'] != control['output_hash']
            or read(bit['candidate'])['processes'] != candidate['processes']):
        raise ValueError('pipeline reduction changed component output bytes')
    profile = dict(read(integration['codegen_proof'])['profile'],
                   TILEMEGA_F3_COALESCED_PARTIALS='0', TILEMEGA_F3_CYCLIC_SLOTS='0')
    validate_component_profile(bit['control'], {f'-D{k}={v}' for k, v in profile.items()})


def validate_gelu_outline_integration(folder, decision, final):
    batch = decision['batch']
    if batch not in (4, 8, 16):
        raise ValueError('outlined GELU batch is not admitted')
    features = ['TILEMEGA_F3_SIDE_NORM', 'TILEMEGA_F3_FULL_TILES',
                'TILEMEGA_F3_H800_GRID', 'TILEMEGA_F3_GELU_OUTLINE']
    if batch in (8, 16):
        features += ['TILEMEGA_F3_MATRIX_STORE', 'TILEMEGA_F3_ATTENTION_VECTOR_STORE']
    features += (['TILEMEGA_F3_CYCLIC_SLOTS', 'TILEMEGA_F3_REPEAT_SELECTOR']
                 if batch == 16 else ['TILEMEGA_F3_RECOMPUTE_TASK'])
    validate_fixed_feature_integration(folder, decision, final, ['body'],
                                       '3' if batch == 16 else '2', features)
    integration = read(folder/'integration.json')
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    bit = read(integration['unit_bit_proof'])
    control, candidate = read(bit['control']), read(decision['unit_gates'][0])
    if (not bit['passed'] or not control['passed']
            or len(control['processes']) != 5
            or len({p['pid'] for p in control['processes']}) != 5
            or any(p['exit_code'] or read(p['guard'])['code'] for p in control['processes'])
            or bit['output_hash'] != candidate['output_hash']
            or candidate['output_hash'] != control['output_hash']
            or read(bit['candidate'])['processes'] != candidate['processes']):
        raise ValueError('outlined GELU changed component output bytes')
    profile = dict(read(integration['codegen_proof'])['profile'], TILEMEGA_F3_GELU_OUTLINE='0')
    validate_component_profile(bit['control'], {f'-D{k}={v}' for k, v in profile.items()})
    # The first fixture accidentally suppressed contraction in its scalar oracle.
    # Retain its failure and the independent full-domain diagnosis explicitly.
    for name in ('oracle_repair', 'main_fixture', 'defaults', 'emission',
                 'model_reuse' if batch == 16 else 'model_cohort'):
        if digest(integration[name]) != integration[name+'_sha256']:
            raise ValueError('outlined GELU integration receipt changed: '+name)
    oracle = read(integration['oracle_repair'])
    diagnostic = read(oracle['diagnostic'])
    if (oracle['original_scalar_mismatches'] or oracle['precision_errors']
            or oracle['max_normalized_error'] > 1e-6
            or not diagnostic['passed'] or len(diagnostic['processes']) != 5
            or len({p['pid'] for p in diagnostic['processes']}) != 5):
        raise ValueError('outlined GELU lacks its full-domain oracle diagnosis')
    for process in diagnostic['processes']:
        if (process['exit_code'] or read(process['guard'])['code']
                or process['binary_sha256'] != oracle['diagnostic_binary_sha256']
                or digest(process['log']) != process['sha256']
                or 'outlined_vs_original=0 outlined_vs_uncontracted=525595 precision_errors=0'
                   not in Path(process['log']).read_text()):
            raise ValueError('outlined GELU diagnostic process is incomplete')
    fixture = read(integration['main_fixture'])
    if not fixture['passed'] or [r['flag'] for r in fixture['rows']] != [0, 1]:
        raise ValueError('outlined GELU lacks both committed fixture proofs')
    for row in fixture['rows']:
        build = read(row['prototype_build'])
        expected_profile = dict(profile, TILEMEGA_F3_GELU_OUTLINE=str(row['flag']))
        if (not row['passed'] or row['encoded_words'] <= 0
                or row['profile'] != expected_profile or build['profile'] != expected_profile
                or digest(row['binary']) != row['binary_sha256']
                or digest(build['binary']) != build['binary_sha256']):
            raise ValueError('outlined GELU committed fixture has changed')
    defaults, emission = read(integration['defaults']), read(integration['emission'])
    if (not defaults['passed'] or sorted(r['batch'] for r in defaults['rows']) != [1, 2, 4, 8, 16]
            or any(not r['passed'] or r['words'] <= 0 or r['gelu_outline'] != 0
                   for r in defaults['rows'])
            or not emission['passed'] or not emission['invalid_setting_rejected']
            or [r['flag'] for r in emission['rows']] != ([0, 1] if batch == 16 else [1])
            or any(not r['matched'] or r['encoded_words'] <= 0 for r in emission['rows'])
            or emission['rows'][-1]['prototype'] != decision['library']):
        raise ValueError('outlined GELU default or codegen checks are incomplete')
    if batch == 16:
        reuse = read(integration['model_reuse'])
        if reuse['proof'] != integration['model_bit_proof']:
            raise ValueError('outlined GELU reused a different model proof')
        model = read(reuse['source'])
    else:
        model = read(integration['model_cohort'])
    if (not model['passed'] or model['passes'] != '5/5'
            or len(model['processes']) != 5 or model['gate'] != decision['gate']
            or model['execution']['artifact_id'] != verify(decision['library'])['artifact_id']):
        raise ValueError('outlined GELU has a different model cohort')


def validate_fixed_feature_integration(folder, decision, final, suites,
                                       register_profile, required_features):
    integration = read(folder/'integration.json')
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    proof = read(integration['codegen_proof'])
    fixtures = read(integration['fixture_proof'])
    if (digest(integration['codegen_proof']) != integration['codegen_proof_sha256']
            or digest(integration['fixture_proof']) != integration['fixture_proof_sha256']
            or not proof['passed'] or not fixtures['passed']
            or proof['reference'] != decision['library']
            or proof['reference_sha256'] != digest(decision['library'])
            or proof['generated_sha256'] != digest(proof['generated'])
            or proof['encoded_words'] <= 0):
        raise ValueError('store bundle lacks its exact main codegen proof')
    profile = proof['profile']
    if (profile.get('TILEMEGA_F3_REGISTER_BUDGET') != register_profile or
            any(profile.get(k) != '1' for k in required_features)):
        raise ValueError('store bundle codegen has a different feature profile')
    if (read(final['library']+'.plan.json')['native_forward'] !=
            read(decision['reference']+'.plan.json')['native_forward']):
        raise ValueError('store bundle changed the accepted coordinates')
    rows = fixtures['rows']
    units = decision['unit_gates']
    if ([r['suite'] for r in rows] != suites
            or len(units) != len(suites) or not fixtures['source_files']):
        raise ValueError('store bundle lacks body, ordering and attention coverage')
    for row, unit in zip(rows, units):
        if (not row['passed'] or row['profile'] != profile
                or row['build_sha256'] != digest(row['build'])
                or row['binary_sha256'] != digest(row['binary'])):
            raise ValueError('store bundle fixture identity changed')
        required = {f'-D{k}={v}' for k, v in profile.items()}
        validate_component_profile(unit, required)
        gate = read(unit)
        processes = gate['processes']
        if (not gate['passed'] or len(processes) != 17
                or len({p['pid'] for p in processes}) != 17
                or Counter(p['name'].split('-')[0] for p in processes) !=
                Counter(unit=5, memcheck=1, initcheck=1, racecheck=5, synccheck=5)
                or len({p['output_hash'] for p in processes}) != 1
                or any(p['exit_code'] or read(p['guard'])['code'] for p in processes)):
            raise ValueError('store bundle component cohort is incomplete')
        build = read(row['build'])
        if (build['binary'] != row['binary'] or
                any(build['inputs'].get(r['snapshot']) != r['sha256']
                    or digest(r['snapshot']) != r['sha256']
                    for r in fixtures['source_files'])):
            raise ValueError('store bundle fixture source differs from its main proof')
        for process in processes:
            gpu = process.get('gpu', gate['gpu'])
            if (process['binary_sha256'] != row['binary_sha256'] or
                    process.get('gpu_lock', f'/tmp/tilemega_gpu{gpu}.lock') !=
                    f'/tmp/tilemega_gpu{gpu}.lock'):
                raise ValueError('store bundle component used a different binary or lock')
    model = read(integration['model_bit_proof'])
    outputs = lambda p: [o['output_sha256'] for c in read(p)['cases'] for o in c['outputs']]
    if (not model['passed'] or model['candidate'] != decision['gate']
            or model['reference'] != decision['reference_gate']
            or not model['hashes'] or model['hashes'] != outputs(decision['gate'])
            or model['hashes'] != outputs(decision['reference_gate'])):
        raise ValueError('store bundle changed the strict model output bytes')
    ledger = read(integration['ledger_receipt'])
    if (ledger['trace_level'] != 1 or ledger['gpu'] != 3
            or ledger['reference_sha256'] != digest(decision['library'])
            or ledger['ledger'] != final['ledger']
            or len(list(csv.DictReader(Path(final['ledger']).open()))) != 63):
        raise ValueError('store bundle lacks the exact measured binary ledger')
    validate_timing_gpu(decision['pair'], 3, 'ABABAB')
    validate_timing_gpu(final['measurement'], 3, 'BBB')


def validate_full_tiles_integration(folder, decision, final, batch):
    integration = read(folder/'integration.json')
    proof = read(integration['main_proof'])
    identity = read(proof['source_identity'])
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    # Freeze the integration receipt: later source changes must not invalidate
    # an already measured, independently identified binary.
    if (digest(integration['main_proof']) != integration['main_proof_sha256']
            or digest(proof['source_identity']) != integration['source_identity_sha256']
            or not proof['passed'] or not identity['passed'] or not identity['files']):
        raise ValueError('full tiles integration differs from its checked sources')
    emitted = read(proof['codegen'])
    if (emitted['status'] != 'passed' or emitted['profiles'] != 20
            or emitted['query_maps'] != 12 or emitted['invalid_settings_rejected'] != 11):
        raise ValueError('full tiles profile emission checks are incomplete')
    on = read(proof['on_proofs'])
    rows = [r for r in on['rows'] if r['batch'] == batch]
    if (not on['passed'] or len(rows) != 1 or not rows[0]['passed']
            or rows[0]['words'] <= 0
            or Path(rows[0]['reference']).resolve() != Path(decision['library']).resolve()
            or verify(rows[0]['model']['library'])['artifact_id'] != rows[0]['model']['artifact_id']):
        raise ValueError('full tiles main codegen does not identify the measured kernel')
    defaults = read(proof['prototype_defaults'])
    b16 = read(proof['prototype_b16_off'])
    b16_off = read(Path(proof['prototype_b16_off']).parent/'default_proof.json')
    off = read(proof['main_b8_off'])
    if (not defaults['passed'] or not off['passed'] or off['words'] <= 0
            or b16['status'] != 'prepared' or not b16_off['passed']
            or b16_off['words'] <= 0 or b16_off['reference'] != b16['reference']['library']
            or sorted((r['batch'], r['full_tiles'], r['repeat_selector'])
                      for r in defaults['rows']) != [(1, 0, 0), (2, 0, 0), (4, 0, 0), (8, 1, 0)]
            or any(not r['passed'] or r['words'] <= 0 for r in defaults['rows'])
            or digest(off['reference']) != off['reference_sha256']):
        raise ValueError('full tiles default preservation proofs are incomplete')
    if (read(final['library']+'.plan.json')['native_forward'] !=
            read(decision['reference']+'.plan.json')['native_forward']):
        raise ValueError('full tiles changed the accepted coordinates')
    required = {'-DTILEMEGA_F3_FULL_TILES=1', '-DTILEMEGA_F3_SIDE_NORM=1',
                '-DTILEMEGA_F3_REGISTER_BUDGET='+('3' if batch == 16 else '2')}
    if batch == 16:
        required.add('-DTILEMEGA_F3_REPEAT_SELECTOR=1')
    if batch == 4:
        required.update(('-DTILEMEGA_F3_H800_GRID=1',
                         '-DTILEMEGA_F3_RECOMPUTE_TASK=1'))
    units = decision['unit_gates']
    for unit in units:
        validate_component_profile(unit, required)
        gate = read(unit)
        for process in gate['processes']:
            if ('gpu_lock' in process and
                    process['gpu_lock'] != f'/tmp/tilemega_gpu{process["gpu"]}.lock'):
                raise ValueError('full tiles component used a different device lock')
    suites = ['body', 'order'] + (['repeat_even', 'repeat_odd'] if batch == 16 else [])
    records = integration['unit_bit_proofs']
    if list(records) != suites or len(units) != len(suites):
        raise ValueError('full tiles lacks complete body, ordering and Repeat coverage')
    for suite, unit in zip(suites, units):
        bit = read(records[suite]);candidate = read(unit);control = read(bit['control'])
        alias = read(bit['candidate'])
        if (not bit['passed'] or not control['passed'] or len(control['processes']) != 5
                or candidate['output_hash'] != control['output_hash']
                or alias['processes'] != candidate['processes']):
            raise ValueError('full tiles component output differs from its disabled control')
    fixtures = read(proof['fixtures'])
    # B4 reuses the same batch-independent full/partial fixtures as B8.
    # Require their complete emitted feature profiles to match before reuse.
    fixture_batch = 8 if batch == 4 else batch
    if batch == 4:
        inherited = [r for r in on['rows'] if r['batch'] == fixture_batch]
        if (len(inherited) != 1 or not inherited[0]['passed']
                or inherited[0]['profile'] != rows[0]['profile']):
            raise ValueError('full tiles inherited fixtures have a different profile')
    selected = [r for r in fixtures['rows'] if r['batch'] == fixture_batch]
    if not fixtures['passed'] or [r['suite'] for r in selected] != ['body', 'order']:
        raise ValueError('full tiles committed fixtures lack both instruction proofs')
    for row, unit in zip(selected, units[:2]):
        if (not row['passed'] or row['words'] <= 0
                or digest(row['binary']) != row['binary_sha256']
                or digest(row['reference']) != row['reference_sha256']
                or any(p['binary_sha256'] != row['reference_sha256'] for p in read(unit)['processes'])):
            raise ValueError('full tiles fixture differs from its checked CUDA binary')
    model = read(integration['model_bit_proof'])
    outputs = lambda p: [o['output_sha256'] for c in read(p)['cases'] for o in c['outputs']]
    if (not model['passed'] or model['candidate'] != decision['gate']
            or model['reference'] != decision['reference_gate']
            or not model['hashes'] or model['hashes'] != outputs(decision['gate'])
            or model['hashes'] != outputs(decision['reference_gate'])):
        raise ValueError('full tiles changed the strict model output bytes')
    ledger = read(integration['ledger_receipt'])
    if (ledger['trace_level'] != 1 or ledger['gpu'] != 3
            or ledger['reference_sha256'] != digest(decision['library'])
            or ledger['ledger'] != final['ledger']
            or len(list(csv.DictReader(Path(final['ledger']).open()))) != 63):
        raise ValueError('full tiles lacks the exact measured binary ledger')
    validate_timing_gpu(decision['pair'], 3, 'ABABAB')
    validate_timing_gpu(final['measurement'], 3, 'BBB')


def validate_fixed_grid_integration(folder, decision, units):
    integration = read(folder/'integration.json')
    main = read(integration['main_proof'])
    if not main['passed'] or [p['flag'] for p in main['proofs']] != [0, 1]:
        raise ValueError('fixed grid main instruction proof is incomplete')
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for proof, reference in zip(main['proofs'], [decision['reference'], decision['library']]):
        if (not proof['passed'] or proof['words'] <= 0
                or Path(proof['reference']).resolve() != Path(reference).resolve()
                or proof['reference_sha256'] != digest(reference)
                or proof['library_sha256'] != digest(proof['library'])):
            raise ValueError('fixed grid instruction proof names a stale artifact')
    fixture = read(integration['unit_proof'])
    if not fixture['passed'] or [r['suite'] for r in fixture['rows']] != ['body', 'attention']:
        raise ValueError('fixed grid repository fixture proof is incomplete')
    for row, unit in zip(fixture['rows'], units):
        gate = read(unit)
        if (not row['passed'] or row['words'] <= 0
                or row['reference_sha256'] != digest(row['reference_binary'])
                or row['main_sha256'] != digest(row['main_binary'])
                or any(p['binary_sha256'] != row['reference_sha256'] for p in gate['processes'])):
            raise ValueError('fixed grid fixture proof differs from its CUDA gate')
    host = read(integration['host_guard'])
    if (not host['passed'] or len(host['passes']) != 5
            or any(p['returncode'] for p in host['passes'])
            or host['gpu_executed'] or host['binary_sha256'] != digest(host['binary'])):
        raise ValueError('fixed grid host rejection gate is incomplete')
    if integration['runtime_header_sha256'] != digest(
            ROOT/'include/tilemega/Codegen/tasks/Sm90F3Runtime.cuh'):
        raise ValueError('fixed grid runtime differs from the checked host guard')
    model = read(integration['model_bit_proof'])
    if (not model['passed'] or model['candidate_gate'] != decision['gate']
            or model['reference_gate'] != decision['reference_gate']):
        raise ValueError('fixed grid model byte proof names a stale gate')
    outputs = lambda p: [o['output_sha256'] for c in read(p)['cases'] for o in c['outputs']]
    if (not model['hashes'] or model['hashes'] != outputs(decision['gate'])
            or outputs(decision['gate']) != outputs(decision['reference_gate'])):
        raise ValueError('fixed grid changed model output bytes')


def validate_recompute_defaults(folder):
    integration = read(folder/'integration.json')
    proofs = [read(p) for p in integration.get('default_proofs', [])]
    if [(p['batch'], p['flag']) for p in proofs] != [
            (1, 0), (2, 0), (4, 0), (8, 0), (16, 0), (8, 1)]:
        raise ValueError('producer recomputation lacks all default and enabled proofs')
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for proof in proofs:
        if (not proof['passed'] or proof['words'] <= 0
                or proof['reference_sha256'] != digest(proof['reference'])
                or proof['library_sha256'] != digest(proof['library'])):
            raise ValueError('producer recomputation default proof names a stale artifact')


def validate_intermittent(path, count):
    receipt = read(Path(path)/'intermittent.json')
    if (receipt['gpu'] != 0 or receipt['jobs'] != count
            or receipt['minimum_gap_s'] < 30 or len(receipt['source_states']) != count):
        raise ValueError('GPU0 continuation lacks individual guarded reservations')
    previous_end = None
    for source in receipt['source_states']:
        states = read(source)
        if len(states) != 1:
            raise ValueError('GPU0 continuation contains a grouped reservation')
        state = next(iter(states.values()))
        guard = read(Path(state['out'])/'guard_result.json')
        queue = read(Path(source).parent.parent/'queue/queue_single.json')
        if (state['status'] != 'done' or state['exit_code'] or guard['code']
                or len(queue) != 1 or queue[0]['env']['CUDA_VISIBLE_DEVICES'] != '0'
                or queue[0]['env']['TILEMEGA_DEVICE_INDEX'] != '0'
                or any(not Path(arg).is_relative_to('/root/python')
                       for arg in queue[0]['command'] if arg.endswith('.py'))):
            raise ValueError('GPU0 continuation has an invalid per-process receipt')
        if previous_end is not None and state['started']-previous_end < 30:
            raise ValueError('GPU0 continuation did not release the card between jobs')
        previous_end = state['ended']


def validate_timing_gpu(result, gpu, order):
    rows = result['measurements']
    if result['gpu'] != gpu or [r['arm'] for r in rows] != list(order):
        raise ValueError('GPU0 continuation mixed timing cards or protocol order')
    pids = []
    for row in rows:
        measured = read(row['measurement'])
        pids.append(measured['pid'])
        if (measured['gpu_index'] != gpu or row['guard']['code']
                or measured['measurement']['warmup'] != 50
                or measured['measurement']['iterations'] != 200
                or len(measured['samples_ms']) != 200
                or row['median_ms'] != measured['median_ms']):
            raise ValueError('GPU0 continuation lacks a complete same-card timing process')
        samples = [json.loads(line) for line in Path(measured['guard_log']).read_text().splitlines()]
        for label in ('before-measurement', 'after-measurement'):
            boundary = [s for s in samples if s.get('label') == label]
            if len(boundary) != 1 or boundary[0]['gpu_index'] != gpu:
                raise ValueError('GPU0 continuation lacks same-card clock boundaries')
    if len(set(pids)) != len(pids):
        raise ValueError('GPU0 continuation reused a timing process')


def validate_followup_allocation(folder, name, decision, final):
    if decision['pair']['gpu'] == 3:
        return 3
    if name != 'boundary_cache216_b4_admitted_v1' or decision['pair']['gpu'] != 0:
        raise ValueError('follow-up did not clear its complete same-card pair: '+name)
    allocation = read(folder/'allocation.json')
    authorization = read(allocation['authorization'])
    if (allocation['gpu'] != 0 or allocation['original_anchor_gpu'] != 3
            or allocation['reanchored'] or authorization['allowed_gpu_indices'] != [0]
            or authorization['minimum_idle_gap_s'] < 30
            or authorization['python_entry_root'] != '/root/python'):
        raise ValueError('GPU0 continuation lacks its explicit allocation provenance')
    protocols = [('pair', decision['pair'], 'ABABAB'),
                 ('selection', read(folder/'selection.json'), 'ABCDDCBA'),
                 ('final', final['measurement'], 'BBB')]
    for name, result, order in protocols:
        validate_timing_gpu(result, 0, order)
        validate_intermittent(allocation['protocol_sources'][name], len(order))
    for path in decision['unit_gates']:
        if read(path)['gpu'] != 0:
            raise ValueError('GPU0 continuation mixed component cards')
        validate_intermittent(Path(path).parent/'scheduler', 17)
    return 0


def validate_boundary_selection(folder, library, reference):
    qualification = read(folder/'trace_qualification.json')
    selection = read(folder/'tune_selection.json')
    prepared = read(folder/'tune_inputs.json')
    eligible = selection['eligible_variants']
    if (not qualification['passed'] or abs(qualification['relative_difference']) > .05
            or len(set(eligible)) < 4 or len(prepared['variants']) != 6
            or any(rank not in range(6) for rank in eligible)):
        raise ValueError('boundary selection lacks qualified four-variant coverage')
    assembled = copy.deepcopy(read(str(reference)+'.plan.json')['native_forward'])
    if len(selection['groups']) != len(prepared['groups']):
        raise ValueError('boundary selection group coverage differs')
    seen = set()
    for index, (chosen, group) in enumerate(zip(selection['groups'], prepared['groups'])):
        windows = chosen['windows_ns']
        if (chosen['group'] != index or chosen['stages'] != group['stages']
                or len(windows) != 6
                or any(windows[rank] is None or not math.isfinite(windows[rank])
                       or windows[rank] <= 0 for rank in eligible)
                or chosen['candidate'] != min(eligible, key=lambda r: (windows[r], r))):
            raise ValueError('boundary selection differs from measured group minima')
        for stage in group['stages']:
            if stage in seen or assembled['stages'][stage]['body'] != 'gemm':
                raise ValueError('boundary selection changed a non-GEMM or repeated group')
            seen.add(stage)
            assembled['stages'][stage] = prepared['variants'][chosen['candidate']]['stages'][stage]
    if assembled != selection['coordinates']:
        raise ValueError('boundary selection changed unselected coordinates')
    specifications = read(folder/'candidate_spec.json')
    result = read(folder/'selection.json')
    if (len(specifications) != 4 or len(result['measurements']) != 8
            or [m['arm'] for m in result['measurements']] != list('ABCDDCBA')
            or any(m['guard']['code'] for m in result['measurements'])
            or Path(result['winner_library']).resolve() != Path(library).resolve()):
        raise ValueError('boundary selection lacks its complete nontrace P-SEL')
    candidates = []
    for spec in specifications:
        expected = assembled if spec['name'] == 'tuned' else prepared['variants'][
            int(spec['name'].removeprefix('existing_variant_'))]
        actual = read(spec['library']+'.plan.json')['native_forward']
        if actual != expected:
            raise ValueError('boundary P-SEL candidate coordinates differ')
        receipt = read(spec['tune_window_source'])
        if (receipt['status'] != 'passed' or len(receipt['traces']) != 3
                or len(receipt['processes']) != 3
                or len({p['pid'] for p in receipt['processes']}) != 3
                or any(p['exit_code'] for p in receipt['processes'])):
            raise ValueError('boundary tie priority lacks three fresh trace processes')
        processes = [read(Path(p)/'samples.json') for p in receipt['traces']]
        if any(len(process) != 50 for process in processes):
            raise ValueError('boundary tie priority lacks fifty windows per process')
        total = statistics.median(statistics.median(
            sum(stage['stage_ns'] for stage in sample) for sample in process)
            for process in processes)
        if spec['tune_window_ns'] != total:
            raise ValueError('boundary tie priority differs from measured windows')
        candidates.append(actual)
    if len({json.dumps(c, sort_keys=True) for c in candidates}) != 4:
        raise ValueError('boundary P-SEL candidates were not deduplicated')
    order = sorted(range(4), key=lambda i: (specifications[i]['tune_window_ns'], i))
    if any(specifications[i]['tune_priority'] != rank for rank, i in enumerate(order)):
        raise ValueError('boundary P-SEL tie priority was not measured')
    proof = read(folder/'instruction_proof.json')
    if (not proof['passed'] or proof['sass_words'] <= 0
            or Path(proof['prototype_library']).resolve() != Path(library).resolve()):
        raise ValueError('boundary candidate lacks its main-source instruction proof')


def validate_whole_program_selection(folder, library, reference):
    """Partial trace coverage can select whole programs only by untied P-SEL."""
    specifications = read(folder/'candidate_spec.json')
    prepared = read(folder/'tune_inputs.json')
    result = read(folder/'selection.json')
    if (len(specifications) != 4 or specifications[0]['name'] != 'current'
            or len(prepared['variants']) != 6 or result['protocol'] != 'P-SEL'
            or result['batch'] != 8 or result['tied']
            or Path(specifications[0]['library']).resolve() != Path(reference).resolve()):
        raise ValueError('whole-program selection lacks an untied four-program comparison')
    validate_timing_gpu(result, 3, 'ABCDDCBA')
    current = read(str(reference)+'.plan.json')['native_forward']
    candidates, ranks, identities = [], [], []
    for index, spec in enumerate(specifications):
        actual = read(spec['library']+'.plan.json')['native_forward']
        if index == 0:
            expected = current
        else:
            match = re.fullmatch(r'existing_variant_([0-5])', spec['name'])
            if not match:
                raise ValueError('whole-program selection contains an assembled candidate')
            rank = int(match[1]); ranks.append(rank)
            expected = prepared['variants'][rank]
            proof = read(spec['instruction_proof'])
            if (not proof['passed'] or proof['sass_words'] <= 0
                    or Path(proof['prototype_library']).resolve() != Path(spec['library']).resolve()):
                raise ValueError('whole-program candidate lacks its main-source instruction proof')
        if actual != expected:
            raise ValueError('whole-program selection changed fixed candidate coordinates')
        identity = read(spec['library']+'.identity.json')
        identities.append(identity['artifact_id'])
        candidates.append((json.dumps(actual, sort_keys=True), json.dumps(
            build_defines(Path(spec['library']+'.cu').read_text(), []), sort_keys=True)))
        gate = Path(spec['gate'])
        checked = read(gate.parent.parent/'result.json')
        processes = checked['processes']
        if (checked['status'] != 'passed' or checked['passes'] != '5/5'
                or len(processes) != 5 or len({p['pid'] for p in processes}) != 5
                or any(p['exit_code'] for p in processes)
                or checked['execution']['artifact_id'] != identity['artifact_id']):
            raise ValueError('whole-program candidate lacks five fresh model processes')
    if len(set(ranks)) != 3 or len(set(candidates)) != 4:
        raise ValueError('whole-program candidates were not deduplicated')
    ranked = []
    for index, (arm, spec) in enumerate(zip('ABCD', specifications)):
        rows = [r for r in result['measurements'] if r['arm'] == arm]
        for row in rows:
            measured = read(row['measurement'])
            if (Path(measured['library_path']).resolve() != Path(spec['library']).resolve()
                    or Path(measured['numerical_gate_path']).resolve() != Path(spec['gate']).resolve()
                    or measured['artifact_id'] != identities[index]):
                raise ValueError('whole-program measurement differs from its checked artifact')
        values = [r['median_ms'] for r in rows]
        median = statistics.median(values)
        if not math.isfinite(median) or median <= 0:
            raise ValueError('whole-program selection has invalid timings')
        ranked.append(dict(arm=arm, median_ms=median, library=str(Path(spec['library']).resolve()),
            process_range_ratio=(max(values)-min(values))/median,
            tune_priority=spec.get('tune_priority', index)))
    ranked.sort(key=lambda r: r['median_ms'])
    threshold = max(.005, 2*max(r['process_range_ratio'] for r in ranked[:2]))
    winner = ranked[0]
    if (result['ranking'] != ranked or result['threshold'] != threshold
            or ranked[1]['median_ms']/winner['median_ms']-1 <= threshold
            or result['winner'] != winner['arm'] or winner['arm'] == 'A'
            or result['candidate_ms'] != winner['median_ms']
            or Path(result['winner_library']).resolve() != Path(winner['library']).resolve()
            or Path(library).resolve() != Path(winner['library']).resolve()):
        raise ValueError('whole-program winner did not clear its measured noise threshold')


def validate_pp_coordinates(library, reference, allow_existing=False, wg_rows=128):
    if wg_rows not in (64, 128):
        raise ValueError('PP follow-up requires a prescribed WG row width')
    current = read(str(reference)+'.plan.json')['native_forward']
    candidate = read(str(library)+'.plan.json')['native_forward']
    expected = json.loads(json.dumps(current))
    if len(expected['stages']) != 63:
        raise ValueError('PP follow-up requires the complete BERT program')
    existing = allow_existing and expected['stages'][4].get('sched') == 'pp'
    for index in range(4, 60, 5):
        stage = expected['stages'][index]
        fixed = dict(tile_m=256 if existing else 128, tile_n=128, tk=64,
                     slots=5, split_k=1, store='tma', mc=0)
        if (any(stage.get(k) != v for k, v in fixed.items()) or
                stage.get('sched', 'coop') != ('pp' if existing else 'coop')):
            raise ValueError('PP follow-up changed its fixed FFN1 control')
        stage.update(tile_m=2*wg_rows, slots=7 if wg_rows == 64 else 5, sched='pp')
    if candidate != expected:
        raise ValueError('PP follow-up changed coordinates outside its fixed FFN1 combination')


def validate_ffn1_component_profile(unit_path, columns):
    required = {f'-DDNN4_FFN1_TMA{columns}_TEST=1',
                '-DTILEMEGA_F3_SIDE_NORM=1',
                f'-DTILEMEGA_F3_REGISTER_BUDGET={2 if columns == 128 else 3}'}
    if columns == 256:
        required.update(('-DTILEMEGA_F3_MATRIX_STORE=1',
                         '-DTILEMEGA_F3_CYCLIC_SLOTS=1'))
    validate_component_profile(unit_path, required)


def validate_component_profile(unit_path, required):
    unit = read(unit_path)
    plan = read(Path(unit_path).parent/'plan.json')
    build = read(plan['build'])
    if (not required.issubset(set(build['command'])) or
            any(p['binary_sha256'] != build['binary_sha256'] for p in unit['processes'])):
        raise ValueError('F3 component lacks its exact compiled profile')


def validate_qkv_tma192_coordinates(library, reference):
    current = read(str(reference)+'.plan.json')['native_forward']
    candidate = read(str(library)+'.plan.json')['native_forward']
    expected = json.loads(json.dumps(current))
    if len(expected['stages']) != 63:
        raise ValueError('QKV follow-up requires the complete BERT program')
    for index in range(1, 60, 5):
        expected['stages'][index].update(tile_m=128, tile_n=192, tk=64,
            slots=4, split_k=1, store='tma', mc=0)
    if candidate != expected:
        raise ValueError('QKV follow-up changed coordinates outside its fixed group')


def validate_ffn1_tma_coordinates(library, reference, columns, slots):
    current = read(str(reference)+'.plan.json')['native_forward']
    candidate = read(str(library)+'.plan.json')['native_forward']
    expected = json.loads(json.dumps(current))
    if len(expected['stages']) != 63:
        raise ValueError('FFN1 follow-up requires the complete BERT program')
    for index in range(4, 60, 5):
        expected['stages'][index].update(tile_m=128, tile_n=columns, tk=64,
            slots=slots, split_k=1, store='tma', mc=0)
    if candidate != expected:
        raise ValueError('FFN1 TMA follow-up changed coordinates outside its fixed group')


def validate_ffn1_tma192_coordinates(library, reference):
    current = read(str(reference)+'.plan.json')['native_forward']
    candidate = read(str(library)+'.plan.json')['native_forward']
    expected = json.loads(json.dumps(current))
    if len(expected['stages']) != 63:
        raise ValueError('FFN1 follow-up requires the complete BERT program')
    for index in range(4, 60, 5):
        expected['stages'][index].update(tile_m=128, tile_n=192, tk=64, slots=4,
                                         split_k=1, store='tma', mc=0)
    if candidate != expected:
        raise ValueError('FFN1 follow-up changed coordinates outside its fixed group')


def validate_ffn2_tma256x64_coordinates(library, reference):
    current = read(str(reference)+'.plan.json')['native_forward']
    candidate = read(str(library)+'.plan.json')['native_forward']
    expected = json.loads(json.dumps(current))
    if len(expected['stages']) != 63:
        raise ValueError('FFN2 follow-up requires the complete BERT program')
    for index in range(5, 61, 5):
        expected['stages'][index].update(tile_m=256, tile_n=64, tk=64, slots=4,
                                         split_k=1, store='tma', mc=0)
    if candidate != expected:
        raise ValueError('FFN2 follow-up changed coordinates outside its fixed group')


def validate_ffn1_reg64_coordinates(library, reference):
    current = read(str(reference)+'.plan.json')['native_forward']
    candidate = read(str(library)+'.plan.json')['native_forward']
    expected = json.loads(json.dumps(current))
    if len(expected['stages']) != 63:
        raise ValueError('FFN1 follow-up requires the complete BERT program')
    for index in range(4, 60, 5):
        expected['stages'][index].update(tile_m=128, tile_n=64, tk=64, slots=8,
                                         split_k=1, store='reg', mc=0)
    if candidate != expected:
        raise ValueError('FFN1 follow-up changed coordinates outside its fixed group')


def validate_ffn2_reg64_coordinates(library, reference):
    current = read(str(reference)+'.plan.json')['native_forward']
    candidate = read(str(library)+'.plan.json')['native_forward']
    expected = json.loads(json.dumps(current))
    if len(expected['stages']) != 63:
        raise ValueError('FFN2 follow-up requires the complete BERT program')
    for index in range(5, 61, 5):
        expected['stages'][index].update(tile_m=128, tile_n=64, tk=64, slots=8,
                                         split_k=4, store='reg', mc=0)
    if candidate != expected:
        raise ValueError('FFN2 follow-up changed coordinates outside its fixed group')


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
            gpu = validate_followup_allocation(source.parent, source.parent.name, decision, final)
            assert pair['gpu'] == gpu and len(pair['measurements']) == 6
            assert all(m['guard']['code'] == 0 for m in pair['measurements'])
            pair_source = str(base/pair_name)
            scope = ('BERT follow-up;original GPU3 anchor retained;not re-anchored' if gpu == 3
                     else 'BERT follow-up on user-authorized GPU0;same-card pair;fixed GPU3 target retained;not re-anchored')
        else:
            source = ROOT/f'runs/dnn4/phase_z/cells_v1/bert_b{batch}/result.json'
            final = read(source)
            library, gate = Path(final['winner']['library']), Path(final['winner']['gate'])
            measurement = read(final.get('final_measurement', source.parent/'final/result.json'))
            profile, pair, pair_source = {}, None, ''
            gpu = 3
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
                           'TILEMEGA_F3_WGMMA_PV', 'TILEMEGA_F3_PP_SIDE_NORM',
                           'TILEMEGA_F3_ATTENTION_VECTOR_STORE', 'TILEMEGA_F3_H800_GRID',
                           'TILEMEGA_F3_RECOMPUTE_TASK', 'TILEMEGA_F3_GELU_OUTLINE'):
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
        assert measurement['protocol'] == 'P-DNN' and measurement['gpu'] == gpu
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
        assert trace['trace_level'] == 1 and trace['gpu'] == gpu
        assert trace['reference_sha256'] == identity['binary_sha256']
        rows.append(dict(model='bert', batch=batch, gpu=gpu, median_ms=ms,
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
            ledger_rows.append(dict(model='bert', batch=row['batch'], gpu=row['gpu'],
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
