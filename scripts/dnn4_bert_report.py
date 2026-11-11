#!/usr/bin/env python3
"""Publish guarded BERT optimization comparisons separately from final cells."""
import argparse
import csv
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
from tilemega.build.identity import verify


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'docs/results')
    args = parser.parse_args()
    base = ROOT/'runs/dnn4/bert_priority'
    probes = [('attention B1', base/'attention_models_v1/bert_b1/experiment.json'),
              ('attention B2', base/'attention_models_v1/bert_b2/experiment.json'),
              ('TK128 B16', base/'tk128_b16_v1/experiment.json'),
              ('residual TMA B16', base/'tma_residual_b16_v1/experiment.json'),
              ('in-situ B16', base/'in_situ_b16_gpu1_v1/experiment.json'),
              ('in-situ B1', base/'in_situ_b1_gpu1_v1/experiment.json'),
              ('register40/232 B1', base/'register_budget_v1/b1/experiment.json'),
              ('FFN1 128x256 B16', base/'large_ffn_b16_v1/experiment.json'),
              ('prefetch1 B16', base/'prefetch_b16_v1/comparison/experiment.json'),
              ('Repeat inline B1', base/'repeat_inline_models_v1/b1/comparison/experiment.json'),
              ('Repeat inline B16', base/'repeat_inline_models_v1/b16/comparison/experiment.json'),
              ('GELU four B16', base/'gelu_four_b16_v1/comparison/experiment.json'),
              ('Repeat single B1', base/'repeat_single_models_v1/b1/comparison/experiment.json'),
              ('Repeat single B1 G22', base/'repeat_single_models_v1/b1/g22_v3/experiment.json'),
              ('Repeat single B16', base/'repeat_single_models_v1/b16/comparison/experiment.json'),
              ('Repeat declined B2', base/'repeat_single_middle_models_v1/b2/comparison/experiment.json'),
              ('Repeat single B4', base/'repeat_single_middle_models_v1/b4/comparison/experiment.json'),
              ('Repeat inline16 B1', base/'repeat_single_inline16_b1_v1/experiment.json'),
              ('FFN1 TMA192 B16', base/'ffn1_tma192_b16_v1/comparison/experiment.json'),
              ('P1 normalization B16', base/'side_norm_models_v2/b16/experiment.json'),
              ('P1 normalization B16 G22', base/'side_norm_models_v2/b16/g22_v1/experiment.json'),
              ('scalar phase B1', base/'scalar_phase_b1_v1/experiment.json'),
              ('FFN2 arena B1', base/'ffn2_arena_b1_v1/experiment.json'),
              ('FFN2 split4 B1', base/'ffn2_split4_b1_v1/experiment.json'),
              ('attention Rows64 B16', base/'attention_rows64_b16_v1/experiment.json'),
              ('96-lane normalization B1', base/'side_norm96_models_v1/b1/experiment.json'),
              ('96-lane normalization B16', base/'side_norm96_models_v1/b16/experiment.json'),
              ('Repeat reg64 B2 GPU2', base/'register64_216_b2_v1/comparison/experiment.json'),
              ('Repeat reg64 B2 G22', base/'register64_216_b2_v1/g22_v1/experiment.json'),
              ('Repeat alias B8 G22', base/'alias_repeat_b8_v1/g22_v1/experiment.json'),
              ('prefetch table B1', base/'prefetch_table_models_v1/b1/experiment.json'),
              ('prefetch table B2', base/'prefetch_table_models_v1/b2/experiment.json'),
              ('P1 normalization B2', base/'side_norm_middle_v1/b2/experiment.json'),
              ('P1 normalization B4', base/'side_norm_middle_v1/b4/experiment.json'),
              ('P1 normalization B8', base/'side_norm_middle_v1/b8/experiment.json'),
              ('FFN2 TK128 B1', base/'ffn2_tk128_b1_v1/experiment.json'),
              ('GELU reciprocal B1', base/'gelu_rcp_models_v1/b1/experiment.json'),
              ('GELU reciprocal B16', base/'gelu_rcp_models_v1/b16/experiment.json'),
              ('attention unroll B1', base/'attention_unroll_models_v2/b1/experiment.json'),
              ('FFN1 TM256 B16', base/'ffn1_tm256_b16_v1/experiment.json'),
              ('Repeat P1 B8', base/'side_norm_alias_b8_v1/experiment.json'),
              ('binding arithmetic B1', base/'repeat_arithmetic_models_v1/b1/experiment.json'),
              ('binding arithmetic B2', base/'repeat_arithmetic_models_v1/b2/experiment.json'),
              ('Repeat P1 normalization B16', base/'repeat_side_norm_b16_v1/experiment.json'),
              ('statistics TMA B16', base/'stats_tma_b16_v1/experiment.json'),
              ('Repeat arithmetic P1 reg64 B4', base/'repeat_arithmetic_middle_v1/b4/experiment.json'),
              ('Repeat arithmetic P1 reg56 B4', base/'repeat_arithmetic_b4_reg56_v1/experiment.json'),
              ('P1 normalization arithmetic B2', base/'arithmetic_side_norm_b2_v1/experiment.json'),
              ('in-situ P1 B16', base/'side_norm_retune_b16_v1/experiment.json'),
              ('in-situ P1 B8', base/'side_norm_retune_b8_v1/experiment.json'),
              ('FFN1 TK128 B16', base/'ffn1_tk128_b16_v1/experiment.json'),
              ('FFN1 TK128 single-store B16', base/'single_store_tk128_b16_v1/experiment.json'),
              ('FFN1 TN256 single-store B16', base/'single_store_tn256_b16_v1/v2/experiment_gpu0.json'),
              ('consumer216 B8 diagnostic', base/'phase72_consumer216_b8_v1/experiment.json'),
              ('consumer216 in-situ B8', base/'phase72_consumer216_retune_b8_gpu0_v1/experiment.json'),
              ('consumer216 in-situ B16', base/'phase72_consumer216_retune_b16_gpu0_v1/experiment.json'),
              ('two-layer arithmetic B16', base/'two_layer_arithmetic_b16_v1/experiment.json'),
              ('attention load pairs B1', base/'attention_load_pair_b1_v1/experiment.json'),
              ('consumer216 tuned B16 G22', base/'consumer216_tuned_b16_g22_v1/experiment.json'),
              ('phase72 B1 G22', base/'phase72_b1_v1/experiment.json'),
              ('Repeat candidate selection B1 G22', base/'repeat_candidate_selection_b1_v2/experiment.json'),
              ('attention Rows64 B1', base/'attention_rows64_b1_v1/experiment.json'),
              ('coalesced partial B1', base/'coalesced_partial_b1_v1/experiment.json'),
              ('cyclic slots B1', base/'cyclic_slot_b1_v1/experiment.json'),
              ('cyclic slots tuned B16 prototype', base/'cyclic_slot_tuned_b16_v1/experiment.json'),
              ('cyclic216 B16 integrated', base/'cyclic216_b16_integrated_v1/experiment.json'),
              ('cyclic coalesced B1 prototype', base/'cyclic_coalesced_b1_v2/experiment.json'),
              ('coalesced cyclic B1 integrated', base/'coalesced_cyclic_b1_integrated_v1/experiment.json'),
              ('coalesced cyclic B2 integrated', base/'coalesced_cyclic_b2_integrated_v1/experiment.json'),
              ('cyclic admitted B8', base/'cyclic_admitted_b8_v1/experiment.json'),
              ('cyclic in-situ B16', base/'cyclic216_retune_b16_v1/experiment.json'),
              ('repaired in-situ B8', base/'repair216_retune_b8_v1/experiment.json'),
              ('current whole-program B4', base/'current_selection_b4_v1/experiment.json'),
              ('producer cyclic B1', base/'producer_cyclic_b1_v1/experiment.json'),
              ('producer cyclic B16', base/'producer_cyclic_b16_v1/experiment.json'),
              ('present bindings B2', base/'present_binding_b2_v1/experiment.json'),
              ('present bindings B16', base/'present_binding_b16_v1/experiment.json'),
              ('side216 B2', base/'side216_b2_v1/experiment.json'),
              ('matrix store B8 prototype', base/'stmatrix_b8_v1/experiment.json'),
              ('matrix store B16 prototype', base/'stmatrix_b16_v1/experiment.json'),
              ('matrix store B8 integrated', base/'matrix_integrated_b8_v1/experiment.json'),
              ('matrix store B16 integrated', base/'matrix_integrated_b16_v1/experiment.json'),
              ('matrix existing in-situ B16', base/'matrix_existing_retune_b16_v1/experiment.json'),
              ('warpgroup QK B4', base/'qk_wgmma_b4_v1/experiment.json'),
              ('warpgroup QK B8', base/'qk_wgmma_b8_v1/experiment.json'),
              ('warpgroup QK B16', base/'qk_wgmma_b16_v1/experiment.json'),
              ('warpgroup QK/PV B4', base/'pv_wgmma_b4_v1/experiment.json'),
              ('warpgroup QK/PV B8', base/'pv_wgmma_b8_v1/experiment.json'),
              ('warpgroup QK/PV B16', base/'pv_wgmma_b16_v1/experiment.json'),
              ('attention existing in-situ B16', base/'attention_existing_retune_b16_v1/experiment.json'),
              ('FFN1 PP matrix B16', base/'ppstore_ffn1_b16_v1/experiment.json'),
              ('FFN1 PP P1 matrix B16', base/'ppnorm_ffn1_b16_v1/experiment.json'),
              ('attention vector stores B8', base/'attention_store_b8_v1/experiment.json'),
              ('attention vector stores B16', base/'attention_store_b16_v1/experiment.json'),
              ('combined attention B4', base/'attention_bundle_b4_v1/experiment.json'),
              ('combined attention B8', base/'attention_bundle_b8_v1/experiment.json'),
              ('combined attention B16', base/'attention_bundle_b16_v1/experiment.json'),
              ('FFN1 TMA192 matrix B8', base/'ffn1_tma192_matrix_b8_v1/experiment.json'),
              ('FFN1 TMA192 matrix B16', base/'ffn1_tma192_matrix_b16_v1/experiment.json'),
              ('FFN2 TMA256x64 matrix B16', base/'ffn2_tma256x64_b16_v1/experiment.json'),
              ('cyclic phase72 B4', base/'cyclic_phase72_b4_v1/experiment.json'),
              ('FFN1 reg64 side216 B2', base/'ffn1_reg64_side216_b2_v1/experiment.json'),
              ('FFN2 reg64 split4 B2', base/'ffn2_reg64_split4_b2_v1/experiment.json'),
              ('FFN1 TMA128 phase72 B4', base/'ffn1_tma128_phase72_b4_v1/experiment.json'),
              ('FFN1 TMA256 matrix B16', base/'ffn1_tma256_matrix_b16_v2/experiment.json'),
              ('QKV TMA192 matrix B16', base/'qkv_tma192_matrix_b16_v1/experiment.json'),
              ('FFN1 TMA128 matrix B8', base/'ffn1_tma128_matrix_b8_v1/experiment.json'),
              ('FFN1 PP full-wait repair B16', base/'pphandoff_full_wait_b16_v1/experiment.json'),
              ('FFN2 TMA128 split2 B8', base/'ffn2_tma128_split2_b8_v1/experiment.json'),
              ('FFN2 TMA128 split2 coalesced B8', base/'ffn2_tma128_split2_b8_coalesced_v1/experiment.json'),
              ('FFN1 M64N128 B8', base/'ffn1_m64n128_b8_v1/experiment.json'),
              ('FFN1 PP nonblocking side B16', base/'pp64_arrive_side_b16_v2/experiment.json'),
              ('Current coalesced B2', base/'current_coalesced_b2_v1/experiment.json'),
              ('cached-boundary in-situ B4', base/'boundary_cache_retune_b4_v1/experiment.json'),
              ('cached-boundary216 in-situ B4', base/'boundary_cache216_b4_v1/experiment.json'),
              ('Current prefetch1 B16', base/'current_prefetch_b16_v1/experiment.json'),
              ('FFN2 register store B8', base/'ffn2_reg_b8_v1/experiment.json'),
              *[(f'phase72 B{batch} GPU2 diagnostic',
                 base/f'scalar_phase72_arithmetic_b{batch}_v1/gpu2_diagnostic_v1/experiment.json')
                for batch in (2, 4, 8, 16)],
              *[(f'phase72 B{batch} G22',
                 base/f'scalar_phase72_arithmetic_b{batch}_v1/experiment.json')
                for batch in (2, 4, 8, 16)]]
    rows = []
    choices = []
    for name, path in probes:
        if not path.exists():
            continue
        experiment = read(path)
        if name in ('attention Rows64 B1', 'coalesced partial B1', 'cyclic slots B1',
                    'cyclic slots tuned B16 prototype', 'cyclic216 B16 integrated',
                    'cyclic coalesced B1 prototype', 'cyclic admitted B8',
                    'coalesced cyclic B1 integrated', 'coalesced cyclic B2 integrated',
                    'producer cyclic B1', 'producer cyclic B16',
                    'present bindings B2', 'present bindings B16', 'side216 B2',
                    'matrix store B8 prototype', 'matrix store B16 prototype',
                    'matrix store B8 integrated', 'matrix store B16 integrated',
                    'matrix existing in-situ B16', 'warpgroup QK B4', 'warpgroup QK B8',
                    'warpgroup QK B16', 'warpgroup QK/PV B4', 'warpgroup QK/PV B8',
                    'warpgroup QK/PV B16', 'attention existing in-situ B16',
                    'FFN1 PP matrix B16', 'FFN1 PP P1 matrix B16',
                    'attention vector stores B8', 'attention vector stores B16',
                    'combined attention B4', 'combined attention B8', 'combined attention B16',
                    'FFN1 TMA192 matrix B8', 'FFN1 TMA192 matrix B16',
                    'FFN2 TMA256x64 matrix B16', 'cyclic phase72 B4',
                    'FFN1 reg64 side216 B2', 'FFN2 reg64 split4 B2',
                    'FFN1 TMA128 phase72 B4', 'FFN1 TMA256 matrix B16',
                    'QKV TMA192 matrix B16', 'FFN1 TMA128 matrix B8',
                    'FFN1 PP full-wait repair B16', 'FFN2 TMA128 split2 B8',
                    'FFN2 TMA128 split2 coalesced B8', 'FFN1 M64N128 B8',
                    'FFN1 PP nonblocking side B16', 'Current coalesced B2'):
            unit_paths = experiment.get('unit_gates', [experiment.get('unit_gate')])
            for unit_path in unit_paths:
                unit = read(Path(unit_path))
                if not unit['passed'] or len(unit['processes']) != 17 or any(
                        p['exit_code'] for p in unit['processes']):
                    raise ValueError('pipeline follow-up requires its complete component gates')
        if name in ('cyclic in-situ B16', 'repaired in-situ B8',
                    'matrix existing in-situ B16', 'attention existing in-situ B16'):
            qualification = read(path.parent/'trace_qualification.json')
            selection = read(path.parent/'tune_selection.json')
            if (not qualification['passed'] or abs(qualification['relative_difference']) > .05
                    or len(selection['eligible_variants']) < 4):
                raise ValueError('in-situ follow-up requires qualified traces and four legal variants')
        if name in ('cached-boundary in-situ B4', 'cached-boundary216 in-situ B4'):
            selection = read(path.parent/'tune_selection.json')
            qualification = read(Path(selection['qualification']))
            if (not qualification['passed'] or abs(qualification['relative_difference']) > .05
                    or len(selection['eligible_variants']) < 4):
                raise ValueError('cached-boundary selection requires qualified traces and four variants')
        if name == 'Current prefetch1 B16':
            preparation = read(Path(experiment['preparation']))
            if not preparation['source_differential']['passed']:
                raise ValueError('prefetch comparison requires its one-define source proof')
        if name.startswith('phase72'):
            for suite in ('scalar_phase72_arithmetic_alias_unit_v1', 'side_norm_reg72_unit_v1'):
                state = read(base/suite/'scheduler/state.json')['dnn4-focused-f3-gates']
                unit = read(Path(state['out'])/'cohort/result.json')
                if state['status'] != 'done' or not unit['passed'] or len(unit['processes']) != 17:
                    raise ValueError('phase72 requires scalar phase and P1 component gates')
        if name in ('FFN1 TK128 single-store B16', 'FFN1 TN256 single-store B16',
                    'consumer216 B8 diagnostic', 'two-layer arithmetic B16'):
            for unit_path in experiment['unit_gates']:
                unit = read(Path(unit_path))
                if not unit['passed'] or len(unit['processes']) != 17 or any(p['exit_code'] for p in unit['processes']):
                    raise ValueError('prototype requires its complete stated component gates')
        if name.startswith('consumer216 in-situ'):
            qualification = read(path.parent/'trace_qualification.json')
            selection = read(path.parent/'tune_selection.json')
            required = 3 if experiment['batch'] == 8 else 4
            if (not qualification['passed'] or abs(qualification['relative_difference']) > .05
                    or len(selection['eligible_variants']) < required):
                raise ValueError('consumer216 diagnostic lacks its stated trace coverage')
        if name == 'attention load pairs B1':
            unit = read(Path(experiment['unit_gate']))
            if not unit['passed'] or len(unit['processes']) != 17 or not experiment['same_output_bytes']:
                raise ValueError('attention scheduling requires component and unchanged-byte gates')
        if name == 'statistics TMA B16':
            state = read(base/'stats_tma_unit_v1/scheduler/state.json')['dnn4-focused-f3-gates']
            unit = read(Path(state['out'])/'cohort/result.json')
            if state['status'] != 'done' or state['exit_code'] or not unit['passed'] or len(unit['processes']) != 17:
                raise ValueError('statistics TMA requires an accepted complete unit/sanitizer cohort')
        if name.startswith('Repeat arithmetic P1') or name == 'P1 normalization arithmetic B2':
            suites = (('repeat_arithmetic_odd_unit_v1', 'side_norm_unit_v2') if 'reg56' in name else
                      ('repeat_arithmetic_alias_unit_v1', 'side_norm_reg64_unit_v1'))
            for suite in suites:
                unit = read(base/suite/'raw/dnn4-focused-f3-gates/attempt1/cohort/result.json')
                if not unit['passed'] or len(unit['processes']) != 17 or any(p['exit_code'] for p in unit['processes']):
                    raise ValueError('arithmetic/P1 combination requires both component gates')
        if name.startswith('in-situ P1'):
            qualification = read(path.parent/'trace_qualification.json')
            if not qualification['passed'] or abs(qualification['relative_difference']) > .05:
                raise ValueError('in-situ P1 requires qualified boundary traces')
        if name == 'FFN1 TK128 B16':
            unit = read(Path(experiment['unit_gate']))
            if not unit['passed'] or len(unit['processes']) != 17 or any(p['exit_code'] for p in unit['processes']):
                raise ValueError('wide-K FFN1 requires its exact focused unit and sanitizer gates')
        if name.startswith('binding arithmetic'):
            suite = ('repeat_arithmetic_odd_unit_v1' if experiment['batch'] == 1
                     else 'repeat_arithmetic_alias_unit_v1')
            unit = read(base/suite/'raw/dnn4-focused-f3-gates/attempt1/cohort/result.json')
            if not unit['passed'] or len(unit['processes']) != 17 or any(p['exit_code'] for p in unit['processes']):
                raise ValueError('arithmetic binding requires pattern reconstruction and unit/sanitizer gates')
        if name == 'Repeat P1 normalization B16':
            unit = read(base/'repeat_single_unit_v3/raw/dnn4-focused-f3-gates/attempt1/cohort/result.json')
            if not unit['passed'] or len(unit['processes']) != 17 or any(p['exit_code'] for p in unit['processes']):
                raise ValueError('B16 loop combination requires the separately qualified Repeat component gate')
        if name == 'FFN1 TM256 B16' or name == 'Repeat P1 B8':
            suites = (('ffn1_tm256_unit_v1',) if name.startswith('FFN1') else
                      ('side_norm_reg64_unit_v1', 'alias_repeat_reg64_unit_v1'))
            for suite in suites:
                unit = read(base/suite/'raw/dnn4-focused-f3-gates/attempt1/cohort/result.json')
                if not unit['passed'] or len(unit['processes']) != 17 or any(p['exit_code'] for p in unit['processes']):
                    raise ValueError('FFN1/Repeat combination requires its focused unit/sanitizer gates')
        if name.startswith('attention unroll'):
            unit = read(base/'attention_unroll_unit_v2/raw/dnn4-focused-f3-gates/attempt1/cohort/result.json')
            if not unit['passed'] or len(unit['processes']) != 17 or any(p['exit_code'] for p in unit['processes']):
                raise ValueError('attention unroll requires unit5 and sanitizer1/1/5/5')
        if name.startswith('GELU reciprocal'):
            precision = read(base/'gelu_rcp_accuracy_v1/gpu3_v2/raw/dnn4-gelu-rcp-accuracy-gpu3/attempt1/probe/result.json')
            unit = read(base/'gelu_rcp_unit_v1/raw/dnn4-focused-f3-gates/attempt1/cohort/result.json')
            if (precision['passes'] != '5/5' or precision['median']['bad'] or
                    precision['median']['max_relative'] > 1e-6 or not unit['passed'] or
                    len(unit['processes']) != 17 or any(p['exit_code'] for p in unit['processes'])):
                raise ValueError('GELU reciprocal requires precision5 plus unit5/sanitizer1/1/5/5')
        if name.startswith('prefetch table'):
            unit = read(base/'prefetch_table_unit_v1/raw/dnn4-focused-f3-gates/attempt1/cohort/result.json')
            if not unit['passed'] or len(unit['processes']) != 17 or any(p['exit_code'] for p in unit['processes']):
                raise ValueError('prefetch table requires unit5 and sanitizer1/1/5/5')
        if 'alias B8' in name:
            for suite in ('alias_repeat_unit_v1', 'alias_repeat_reg64_unit_v1'):
                unit = read(base/suite/'raw/dnn4-focused-f3-gates/attempt1/cohort/result.json')
                if not unit['passed'] or len(unit['processes']) != 17 or any(
                        p['exit_code'] for p in unit['processes']):
                    raise ValueError('virtual aliases require both register-profile unit/sanitizer gates')
        if 'reg64' in name:
            unit = read(base/'register64_216_unit_v1/raw/dnn4-focused-f3-gates/attempt1/cohort/result.json')
            if not unit['passed'] or len(unit['processes']) != 17 or any(
                    p['exit_code'] for p in unit['processes']):
                raise ValueError('64/216 requires unit5 and sanitizer1/1/5/5')
        if name == 'P1 normalization B2':
            unit = read(base/'side_norm_reg64_unit_v1/raw/dnn4-focused-f3-gates/attempt1/cohort/result.json')
            if not unit['passed'] or len(unit['processes']) != 17 or any(p['exit_code'] for p in unit['processes']):
                raise ValueError('B2 normalization requires the combined64/216 unit/sanitizer gate')
        if 'normalization' in name:
            suite = 'side_norm96_unit_v1' if name.startswith('96-lane') else 'side_norm_unit_v2'
            unit = read(base/suite/'raw/dnn4-focused-f3-gates/attempt1/cohort/result.json')
            if not unit['passed'] or len(unit['processes']) != 17 or any(
                    p['exit_code'] for p in unit['processes']):
                raise ValueError('P1 normalization requires unit5 and sanitizer1/1/5/5')
        identity = verify(Path(experiment['library']))
        source_text = Path(experiment['library']+'.cu').read_text()
        repeat = re.search(r',Stage_0,Repeat<(6|12),', source_text)
        gate_path = Path(experiment['gate']).parent.parent/'result.json'
        gate = read(gate_path)
        if gate['status'] != 'passed' or gate['passes'] != '5/5' or any(
                not o.get('output_sha256') for c in gate['cases'] for o in c['outputs']):
            raise ValueError('BERT comparison requires five fresh byte-hashed FP32 gates')
        pair_path = Path(experiment.get('pair_source', path.parent/'pair/result.json'))
        if not pair_path.exists() and (path.parent/'winner_pair/result.json').exists():
            pair_path = path.parent/'winner_pair/result.json'
        if not pair_path.exists():
            pair_path = path.parent/f'pair_gpu{experiment["gpu"]}/result.json'
        first = read(pair_path)
        comparisons = [(f'GPU{first["gpu"]} same-card diagnostic', pair_path)]
        if experiment.get('initial_pair_source'):
            comparisons.append(('initial complete pair;extreme variability;not admission',
                                Path(experiment['initial_pair_source'])))
        diagnostic = path.parent/'gpu0_diagnostic_v1/pair/result.json'
        if diagnostic.exists():
            comparisons.append(('GPU0 allocation deviation;diagnostic only;not admission', diagnostic))
        g22 = path.parent/'g22_pair/result.json'
        if g22.exists():
            comparisons.append(('G22 diagnostic versus rank0', g22))
        for scope, source in comparisons:
            pair = read(source)
            if pair['protocol'] != 'P-PAIR' or len(pair['measurements']) != 6:
                raise ValueError('BERT comparison requires a complete paired protocol')
            if any(m['guard']['code'] for m in pair['measurements']):
                raise ValueError('BERT timing guard did not pass')
            rows.append(dict(experiment=name, batch=experiment['batch'], gpu=pair['gpu'],
                reference_ms=pair['reference_ms'], candidate_ms=pair['candidate_ms'],
                ratio=pair['ratio'], gain_pct=100*(1-pair['ratio']),
                threshold_pct=100*pair['threshold'],
                accepted=pair['gpu'] != 0 and pair['ratio']<1-max(pair['threshold'],
                    experiment.get('minimum_gain', 0),
                    .03 if name.startswith('Repeat') or (name.startswith('phase72') and experiment['batch'] in (8, 16)) else 0), scope=scope,
                correctness='P-CORR5/5 with cross-process output byte hashes',
                artifact_id=identity['artifact_id'], binary_sha256=identity['binary_sha256'],
                source_digest=identity['source']['source_digest'],
                clocks=json.dumps([{k:m[k] for k in ('start_sm_clock_mhz','end_sm_clock_mhz')}
                                   for m in pair['measurements']], separators=(',', ':')),
                evidence='verified', qualification=pair['qualification'],
                gate=str(gate_path), source=str(source),
                repeat_layers=12//int(repeat[1]) if repeat else 0))
        selection = path.parent/'tune_selection.json'
        if selection.exists():
            d = read(selection)
            choices.append(name+': '+', '.join(f"group{g['group']}→candidate{g['candidate']}"
                                              for g in d['groups'])+'.')
    if not rows:
        raise ValueError('no completed BERT comparisons')
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/'DNN4_BERT_optimization.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# DNN4 BERT optimization progress', '',
             'Verified guarded comparisons use the same BF16 model inputs and the coordinates identified by each artifact. '
             'All candidates passed five fresh FP32-reference processes with identical cross-process output hashes. '
             'P-PAIR uses A/B/A/B/A/B, three fresh processes per side, 50 warmups and 200 events per process. '
             'Acceptance requires a gain larger than max(0.5%, twice the larger process range). '
             'GPU1 diagnostics guide optimization; G22 diagnostics compare on GPU3 against the explicitly identified control and do not '
             'replace selection against the eventual final winner or three-process P-DNN.', '',
             '| Experiment | GPU | Reference ms | Candidate ms | Gain % | Noise threshold % | Significant gain |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        lines.append(f'| {row["experiment"]} | {row["gpu"]} | {row["reference_ms"]:.6f} | '
                     f'{row["candidate_ms"]:.6f} | {row["gain_pct"]:+.3f} | '
                     f'{row["threshold_pct"]:.3f} | {row["accepted"]} |')
    lines += ['', 'Verified attention experiment: QK/PV columns were distributed across eight consumer warps '
              'without changing any output accumulation order. Per-head unit5/5 and sanitizer memcheck1/1, '
              'initcheck1/1, racecheck5/5, synccheck5/5 passed; whole-model outputs also matched the control '
              'byte for byte. Neither timing gain exceeded process variability, so the original header was restored. '
              'The initial driver added an unnecessary3% cutoff, which does not affect either rejection '
              'because both measured noise thresholds exceed3%. Frozen sources and the patch remain under runs/dnn4/bert_priority.', '',
              'Verified TK128 experiment: one deterministic max-legal-slot configuration on12 QKV stages; '
              'other stages and tile/split/store coordinates unchanged. No significant gain. '
              'Verified residual TMA experiment: only24 residual-reuse stages change store mode, with no '
              'additional shared allocation or changes to tiles/splits/TK/Slots.', '',
              'Verified in-situ experiments use the six already authorized candidate programs, three fresh '
              'processes per variant and20 warmups/50 trace windows. Group sums select a coordinate, then '
              'the assembled nontrace program faces a whole-model P-PAIR gate. These experiments add no parameter scan.', '',
              *choices, '',
              'Verified post-collection checks reject40/232: the89-case scan still spills28B on256x128 '
              'residual_ln_rowstats, and the full B16 program spills36B; it is never executed. B1 compiles '
              'without spill but has no significant gain. FFN1 128x256 uses one deterministic legal '
              'four-slot register-store variant; it regresses and is rejected. Distance1 has no significant gain.', '',
              'Verified Repeat inline tables use grid-constant parameter storage instead of dependent '
              'global index reads. Unit5/5 and sanitizer memcheck1/1,initcheck1/1,racecheck5/5,synccheck5/5 '
              'pass; both whole-model comparisons fail DT-CODE. The implementation patch is archived '
              'and the production headers are restored. Four-element GELU scheduling also has no '
              'end-to-end gain: five full precision probes report0 bad values and preserve scalar bits; '
              'the132-CTA resident epilogue ratio is1.0072, and its full B16 comparison is rejected.', '',
              'Verified single-layer Repeat12x5 is an explicit deviation from the prescribed two-layer '
              'Repeat6x10. Logical stage numbers preserve arena parity for odd loop bodies. Odd and even '
              'body unit cohorts pass5/5; each passes isolated memcheck1/1,initcheck1/1,racecheck5/5, '
              'synccheck5/5. B1 and B16 strict whole-model checks pass5/5 with identical cross-process '
              'output hashes. GPU2 B1 improves19.08%;GPU3 complete P-PAIR confirms19.09%. B16 and B4 '
              'have no significant gain. The B2 published rank1 coordinates decline Repeat and remain '
              'flat; a separate periodic B2 candidate spills4B and is never executed. The mode remains '
              'explicitly selected rather than enabled for unmeasured coordinates.', '',
              'Verified B1 boundary trace sum535456ns exceeds fresh nontrace smoke498320ns by7.45%, '
              'so these windows are excluded from P-TUNE. Four of the six original complete variants '
              'spill and are never executed. Nontrace P-SEL uses the two legal loop variants, the accepted '
              'loop control and an existing distinct model-rank candidate; it retains the loop control. '
              'GPU3 P-DNN has median0.498176ms,p900.500640ms,process range0.013232ms. This is still '
              'above the unchanged0.4858ms BF16 goal. Failed or killed partial paired cohorts are discarded '
              'in full; occupancy exit75 retries the complete cohort.', '',
              'Verified follow-up rejects16-bit inline parameter tables: odd-loop unit5/5 and '
              'sanitizer1/1/5/5 pass;full B1 FP32/byte checks5/5 pass, but the complete P-PAIR '
              'against the accepted loop is0.498656→0.507488ms,ratio1.017712,noise0.9369%. '
              'The prototype headers are restored. Earlier B8 canonical buffer aliases required '
              'retaining its exact flat program instead of failing. Four real '
              'CPU codegen cases cover B1 one/two-layer loops,B2 nonperiodic fallback and B8 alias '
              'fallback;all retain63 logical stages. No new B8 performance is claimed.', '',
              'Verified P1 row normalization is explicitly selected only for B16: the side warp '
              'reduces the existing statistics in their original order and publishes cached '
              '(mean,inv) through32 individual release arrivals before consumer side_full. '
              'This deviates from consumer-side normalization in section5.2. The first prototype '
              'failed racecheck with89 errors and is rejected regardless of timing. The repaired '
              'implementation passes unit5/5 and isolated memcheck1/1,initcheck1/1,racecheck5/5, '
              'synccheck5/5. B1 loop and B16 whole-model FP32/byte checks each pass5/5. '
              'B16 GPU2/GPU3 complete paired gains are3.03%/2.71%;GPU3 P-DNN median1.421744ms, '
              'p901.446144ms,process range0.001824ms. This still exceeds the unchanged1.0869ms '
              'BF16 goal. Combining P1 normalization with the periodic B2 loop spills8B stores/32B '
              'loads and is never executed;the standalone periodic B2 loop also remains rejected.', '',
              'Verified explicit scalar phase selection eliminates B1 local-memory instructions '
              'and passes odd-body unit5/5 and sanitizer1/1/5/5, but has no significant E2E gain; '
              'the prototype is archived and restored. B2 still spills28B and is never executed. '
              'Combining scalar phase with FFN2 Slots6 restores one-arena preissue and measures '
              '2.03% faster, below its6.36% noise threshold;it is not admitted. A separate FFN2 '
              'split4/Slots6 variant regresses5.82% and is also rejected. Fixed BF16 targets and '
              'the50/200 measurement protocol remain unchanged.', '',
              'Verified attention Rows64 B16 passes per-head unit5/5,sanitizer1/1/5/5 and '
              'whole-model FP32/byte checks5/5, but its0.60% gain over the accepted Rows128 '
              'P1-normalization control is below1.70% noise. Three-warp row normalization '
              'also passes unit5/5,sanitizer1/1/5/5 and B1/B16 model checks5/5;B1 regresses3.05%, '
              'and B16 improves0.43% below0.66% noise. Both mechanisms are rejected for '
              'performance admission;the three-warp implementation is archived and restored. '
              'Codegen exposes explicit SIDE_NORM and ATTENTION_ROWS settings while preserving '
              'the default source. A real three-profile CPU test checks all12 query-map boxes, '
              'all63 stage bindings and invalid setting rejection.', '',
              'Verified opt-in register profile64/216 keeps the per-SMSP sum496 below the '
              'entry pool504. It deviates from the prescribed56/224 fallback. The periodic B2 '
              'single-layer loop now compiles without spill;focused odd-body unit5/5 and '
              'memcheck1/1,initcheck1/1,racecheck5/5,synccheck5/5 pass. Whole-model FP32/byte '
              'checks5/5 pass. GPU2 gain7.49% is below7.99% noise and is not admitted;the '
              'required GPU3 comparison confirms5.14% gain with0.50% noise and passes DT-CODE. '
              'B2 SASS falls from1868800B to275584B. Three existing B1 variants become '
              'nonspilling;their strict model checks5/5 pass,but G22 four-candidate P-SEL '
              'retains the previously accepted loop. One remaining B1 variant spills12B and '
              'is never executed. No new coordinate scan is added;the profile stays explicit. '
              'A real four-profile CPU test also verifies invalid register-budget rejection.', '',
              'Verified B8 alias handling now gives each repeated operand role a virtual index '
              'when physical aliases vary across layers. Physical allocations and prefetch IDs '
              'stay unchanged. Six real CPU codegen cases certify63-stage coverage;B1 one/two-layer '
              'and accepted periodic B2 sources remain byte-identical. B8 one/two-layer contract '
              'tests each reject six remap corruptions and preserve the flat source. Virtual-ID '
              'units under56/224 and64/216 each pass5/5 and sanitizer1/1/5/5. The B8 model spills '
              '4B stores/28B loads under56/224 and is never executed;64/216 compiles without spill '
              'and passes strict model checks5/5. GPU2 was cancelled before measurement because '
              'a separate baseline controller held its lock;that controller was left running. '
              'GPU3 P-PAIR is1.051232→1.070064ms,ratio1.017914,noise2.9649%;DT-CODE fails, '
              'so B8 keeps the published flat winner. The newly valid Repeat path remains opt-in.', '',
              'Verified host-selected Repeat prefetch tables preserve prefetch distance and byte budgets, '
              'pass odd-body unit5/5,sanitizer1/1/5/5 and B1/B2 model checks5/5. '
              'Complete GPU3 P-PAIR gives B1 ratio1.039358,noise6.8132% and B2 ratio0.959454,noise5.6363%. '
              'Neither gain exceeds noise;the prototype header is archived and restored.', '',
              'Verified middle-batch P1 normalization preserves accepted coordinates and passes strict '
              'model checks5/5. B2 GPU3 ratio0.950017 is below5.61% noise,so its old loop stays selected. '
              'B4 GPU3 ratio0.963260 beats1.94% noise;P-DNN median0.760672ms,p900.764224ms,range0.006784ms. '
              'The64/216 side-normalization unit also passes5/5 and sanitizer1/1/5/5. '
              'B8 ratio0.970675 gives2.9325% gain below2.9386% noise and is rejected. '
              'B4 remains above its fixed BF16 goal and retains the original anchor without re-anchoring.', '',
              'Verified FFN2 TK128 retains split3 and the tile,using the largest legal six-slot ring. '
              'B1 model checks5/5 pass;GPU3 ratio1.033929,noise4.85% gives no gain,and accepted coordinates stay unchanged. '
              'Verified GELU reciprocal without Newton refinement passes the2^24-point grid and all finite BF16 '
              'values5/5 with max normalized error1.972e-7. The explicitly reserved GPU3 precision cohort is used; '
              'earlier scheduler-environment attempts are excluded. Unit5/5,sanitizer1/1/5/5 and B1/B16 model checks5/5 pass. '
              'The132-CTA three-task epilogue ratio is0.923058 over the current approximation on GPU3, '
              'but B1 E2E ratio1.031251 and B16 ratio0.991282 fail noise6.58%/1.60%. '
              'The prototype is archived and both production header and temporary gate option are restored.', '',
              'Verified attention QK contraction-loop unrolling preserves MMA order and passes per-head unit5/5 '
              'plus sanitizer1/1/5/5. The B1 program compiles without spill and passes model checks5/5, '
              'but its0.63% gain is below7.25% noise. The B16 program spills16B stores/244B loads and is '
              'never executed. Initial pragma-macro compilation failures are retained separately. '
              'The prototype header and temporary gate option are restored.', '',
              'Verified cooperative FFN1 TM256/TN128 uses two m64 fragments per WG,maximum legal four slots '
              'and register stores. Its exact NT12/spanning body passes unit5/5 and sanitizer1/1/5/5; '
              'the full B16 program is nonspilling and passes model checks5/5. GPU3 P-PAIR ratio1.145325 '
              'exceeds3.03% noise;reject and retain the accepted B16 coordinates. The B8 alias-safe loop '
              'combined with P1 normalization and64/216 also passes model checks5/5, but its0.73% gain '
              'fails2.44% noise and DT-CODE3%;retain the flat B8 winner.', '',
              'GPU3 foreign occupancy delayed additional same-card G22 probes; only our pending '
              'schedulers were cancelled, and fresh candidate/control pairs moved to authorized GPU2. '
              'The separately identified B1/B2/B16 follow-ups do not silently replace the earlier frozen20-cell collection. '
              'That BERT batch collection '
              'completed on GPU3: B1/B2/B4/B8/B16 =0.614016/0.695504/0.790144/1.043264/1.463168ms. '
              'None beats its frozen BF16 target. CNN collection is complete and paused; '
              'existing CNN results and all60 baseline arms are reused. GPU ownership remains '
              'qualified by the container PID namespace. See DNN4_BERT_optimization.csv for identities, '
              'clocks, correctness receipts and raw timing sources.']
    pending = []
    for path in (base/'repeat_arithmetic_models_v1/b1', base/'repeat_arithmetic_models_v1/b2',
                 base/'repeat_side_norm_b16_v1'):
        if (path/'native/result.json').exists() and not (path/'experiment.json').exists():
            identity = verify(path/'native/model.so')
            state = read(path/f'correctness/{identity["artifact_id"][:20]}/scheduler/state.json')
            reservation = state['dnn4-f3-model-gates']
            gate = read(Path(reservation['out'])/'cohort/result.json')
            if reservation['status'] != 'done' or gate['status'] != 'passed' or gate['passes'] != '5/5':
                raise ValueError('pending candidate lacks complete strict correctness')
            pending.append(path.name if path.name.startswith('repeat_') else 'binding arithmetic '+path.name)
    if pending:
        cpu = read(base/'host_build_v1/main_codegen_result.json')['cases']
        if (len(cpu) != 4 or any(c['status'] != 'passed' or c['logical_stages'] != 63 or
                                not c['invalid_arithmetic_rejected'] for c in cpu) or
                '100% tests passed, 0 tests failed out of 7' not in
                (base/'checkpoint_ctest_v7.log').read_text()):
            raise ValueError('pending implementation lacks host codegen/ctest receipts')
        lines += ['', 'Verified pending candidates: '+', '.join(pending)+'. '
                  'All compile without spill and pass five fresh strict FP32/byte checks. '
                  'Arithmetic remaps use only exactly verified constant, affine or period-two columns; '
                  'irregular columns keep the original load. Odd56/224 and alias64/216 component cohorts '
                  'each pass unit5/5 and sanitizer1/1/5/5, including absent and irregular index cases. '
                  'Four real codegen cases preserve63 logical stages and reject invalid settings; '
                  'the incremental host build and seven related ctest cases pass. '
                  'The arithmetic feature stays explicitly selected and default0 pending complete GPU3 pairs. '
                  'An additional GPU0 diagnostic is a recorded card-allocation deviation caused by '
                  'prolonged foreign occupancy; it cannot replace G22 admission or the current accepted artifacts.']
    arithmetic = base/'repeat_arithmetic_models_v1/b2/experiment.json'
    if arithmetic.exists():
        decision = read(arithmetic)
        pair = decision['pair']
        assert decision['accepted'] and pair['gpu'] == 3
        final = read(arithmetic.parent/'g22_followup_final_v1/final_result.json')
        assert not final['smoke_regression'] and final['measurement']['protocol'] == 'P-DNN'
        lines += ['', 'Verified arithmetic remaps use compile-time checked constant, affine or '
                  'period-two index columns; irregular columns retain lookup. Odd56/224 and '
                  'alias64/216 units each pass5/5 and sanitizer1/1/5/5;both models pass strict5/5 '
                  'without spill. Four real codegen cases and seven related ctests pass. '
                  f'B2 GPU3 P-PAIR {pair["reference_ms"]:.6f}→{pair["candidate_ms"]:.6f}ms '
                  f'beats{100*pair["threshold"]:.2f}% noise;P-DNN '
                  f'{final["measurement"]["candidate_ms"]:.6f}ms passes B2 admission. '
                  'B1 has no significant gain;B16 Repeat plus P1 normalization also fails '
                  'DT-CODE and keeps the flat control. GPU0 allocation-deviation diagnostics '
                  'are separately identified and cannot replace G22 admission. '
                  'The arithmetic feature remains default0 and is explicitly selected by admitted profiles.']
    diagnostic = base/'ncu_b16_v1/result.json'
    if diagnostic.exists():
        ncu = read(diagnostic)
        if ncu['error'] != 'ERR_NVGPUCTRPERM' or ncu['settings_changed']:
            raise ValueError('unexpected hardware-counter diagnostic state')
        lines += ['', 'Verified hardware-counter diagnostic: ncu2025.1.1 exits1 with ERR_NVGPUCTRPERM; '
                  'no counter samples are available and no device/driver settings were changed. '
                  'Fresh host CMake configuration also fails because the external MLIR installation '
                  'lacks obj.MLIRCAPIDebug/Debug.cpp.o. The existing configured incremental build succeeds.']
    statistics_tma = base/'stats_tma_b16_v1/experiment.json'
    if statistics_tma.exists():
        decision = read(statistics_tma)
        assert not decision['accepted']
        lines += ['', 'Verified statistics-TMA prototype maps even-M statistics as a BF16 byte view '
                  'and loads all NT parts with one unswizzled3D TMA;odd M and NT1 keep Bulk. '
                  'This explicitly deviates from per-part Bulk in section5.2. Focused units '
                  'pass5/5,memcheck1/1,initcheck1/1,racecheck5/5,synccheck5/5;the complete '
                  'second guarded cohort is used and the interrupted first is excluded. '
                  'B16 compiles without spill and passes model5/5. GPU3 P-PAIR '
                  f'ratio{decision["pair"]["ratio"]:.6f} exceeds '
                  f'{100*decision["pair"]["threshold"]:.2f}% noise. Reject;the prototype '
                  'patch is archived and all production files are restored.']
    for batch in (8, 16):
        folder = base/f'side_norm_retune_b{batch}_v1'
        if not (folder/'tune_selection.json').exists():
            continue
        selection = read(folder/'tune_selection.json')
        qualification = read(folder/'trace_qualification.json')
        eligible = selection.get('eligible_variants', list(range(6)))
        lines += ['', f'Verified P1 B{batch} in-situ follow-up: boundary overhead '
                  f'{100*qualification["relative_difference"]:.3f}% passes5%;'
                  f'{len(eligible)}/6 existing variants complete three fresh20/50 traces each. '
                  'Spilled variants, if any, are excluded before execution and remain recorded. '
                  'Group-window winners describe local selection;only the assembled nontrace '
                  'program can qualify through the complete GPU3 end-to-end pair.']
        if (folder/'experiment.json').exists():
            decision = read(folder/'experiment.json')
            lines += [f'Verified assembled B{batch} P-PAIR ratio{decision["pair"]["ratio"]:.6f};'
                      f'noise{100*decision["pair"]["threshold"]:.3f}%;'
                      f'accepted={decision["accepted"]}.']
    wide_k = base/'ffn1_tk128_b16_v1/experiment.json'
    if wide_k.exists():
        decision = read(wide_k)
        lines += ['', 'Verified FFN1 TK128 uses three slots and register stores on the12 FFN1 '
                  'stages;all other accepted B16 coordinates remain unchanged. The exact '
                  '2048x3072x768,NT12 unit passes5/5,memcheck1/1,initcheck1/1,racecheck5/5,'
                  'synccheck5/5;the whole model passes5/5 without spill. '
                  f'GPU3 P-PAIR ratio{decision["pair"]["ratio"]:.6f} regresses and is rejected. '
                  'The regression unit remains as coverage for wide-K spanning slots.']
    scalar = base/'scalar_arithmetic_b8_v1/native/model.so.ptxas.log'
    if scalar.exists():
        unit = read(base/'scalar_arithmetic_alias_unit_v1/raw/dnn4-focused-f3-gates/attempt1/cohort/result.json')
        spills = list(map(int, re.findall(r'(\d+) bytes spill (?:stores|loads)', scalar.read_text())))
        if not unit['passed'] or len(unit['processes']) != 17 or not any(spills):
            raise ValueError('scalar arithmetic rejection lacks its unit or compilation evidence')
        lines += ['', 'Verified scalar phase plus arithmetic alias component passes unit5/5 and '
                  'sanitizer1/1/5/5. The B8 Repeat/P1 whole-model build still spills '
                  +'/'.join(map(str, spills))+'B stores/loads under64/216;it is never executed. '
                  'The prototype patch is archived and its production header is restored.']
    single_store = base/'single_store_tk128_b16_v1/experiment.json'
    if single_store.exists():
        decision = read(single_store)
        assert not decision['accepted']
        lines += ['', 'Verified TK128 single-store prototype explicitly deviates from section6.2: '
                  'one8KiB staging box per WG replaces two and uses wait_group.read0 before '
                  'each reuse. This fits three TK128 slots while retaining TMA writes. '
                  'Exact FFN1 and cross-stage ordering cohorts each pass unit5/5 and '
                  'sanitizer1/1/5/5;the whole model passes5/5 without spill. '
                  'The first pair has351.287% noise and is inconclusive;all six processes '
                  'remain recorded. A complete second pair measures '
                  f'ratio{decision["pair"]["ratio"]:.6f} against '
                  f'{100*decision["pair"]["threshold"]:.3f}% noise. Reject and restore the '
                  'production files;the archived patch retains the prototype.']
    wide_store = base/'single_store_tn256_b16_v1/v2/experiment_gpu0.json'
    if wide_store.exists():
        decision = read(wide_store)
        assert not decision['accepted']
        lines += ['', 'Verified TN256 single-buffer follow-up retains TK64 and four slots on '
                  'the12 FFN1 stages. The first model fails correctness because its serialized '
                  'weight-packing recipes still use TN128;that artifact is never timed. '
                  'A separate corrected artifact updates all12 recipes,compiles without spill '
                  'and passes model5/5. The exact TN256 component passes unit5/5 and '
                  'sanitizer1/1/5/5;cross-stage ordering reuses the prior17/17 single-buffer '
                  'protocol gate. GPU2 never launches under foreign occupancy;GPU0 '
                  'allocation-deviation P-PAIR gives '
                  f'ratio{decision["pair"]["ratio"]:.6f},noise'
                  f'{100*decision["pair"]["threshold"]:.3f}%. '
                  'Reject and restore the prototype headers;no G22 admission.']
    attention_loads = base/'attention_load_pair_b1_v1/experiment.json'
    if attention_loads.exists():
        decision = read(attention_loads)
        lines += ['', 'Verified attention operand scheduling issues two independent shared '
                  'matrix loads before their dependent MMAs,without changing any output '
                  'accumulation order or Rows16 geometry. Per-head unit5/5 and '
                  'sanitizer1/1/5/5 pass;the B1 model passes5/5 and its output bytes '
                  'match the admitted control. SASS is259456B,maxR195,zero spill. '
                  'GPU0 diagnostic P-PAIR gives '
                  f'ratio{decision["pair"]["ratio"]:.6f},noise'
                  f'{100*decision["pair"]["threshold"]:.3f}%. '
                  'No significant end-to-end gain;archive and restore the isolated header.']
    for batch in (8, 16):
        experiment = base/f'phase72_consumer216_retune_b{batch}_gpu0_v1/experiment.json'
        if experiment.exists():
            decision = read(experiment)
            selection = read(experiment.parent/'tune_selection.json')
            qualification = read(experiment.parent/'trace_qualification.json')
            lines += ['', f'Verified72/216 B{batch} re-selection qualifies'
                      f'{100*qualification["relative_difference"]:.3f}% boundary overhead;'
                      f'{len(selection["eligible_variants"])}/6 existing variants complete'
                      '3/3 fresh traces. '
                      + ('B8 deliberately uses reduced three-member coverage;this is a '
                         'diagnostic deviation,not complete P-TUNE coverage. ' if batch == 8 else '')
                      +
                      'The assembled nontrace model passes5/5;GPU0 P-PAIR gives'
                      f'ratio{decision["pair"]["ratio"]:.6f} versus'
                      f'{100*decision["pair"]["threshold"]:.3f}% noise. '
                      'Admission and current artifacts remain unchanged.']
    two_layer = base/'two_layer_arithmetic_b16_v1/experiment.json'
    if two_layer.exists():
        decision = read(two_layer)
        lines += ['', 'Verified prescribed two-layer Repeat6x10 combined with arithmetic '
                  'bindings and P1 retains the B16 coordinates and56/224 roles. Alias '
                  'and P1 component gates each pass17/17;the model passes5/5 without '
                  'spill. GPU0 diagnostic P-PAIR gives '
                  f'ratio{decision["pair"]["ratio"]:.6f} versus'
                  f'{100*decision["pair"]["threshold"]:.3f}% noise and is rejected. '
                  'The B8 counterpart spills52/192B stores/loads and is never executed.']
    boundary = base/'trace3_local_boundary_b1_v2/trace_qualification.json'
    if boundary.exists():
        qualification = read(boundary)
        assert not qualification['passed']
        lines += ['', 'Verified B1 boundary-trace cache is rejected:thread0 keeps the four '
                  'markers in unused header padding and flushes at leave;three fresh '
                  'GPU3 processes compare output bytes against the admitted control. '
                  f'Window sum differs by{100*qualification["relative_difference"]:.3f}% '
                  'from nontrace,above5%,so these windows never select coordinates. '
                  'The first static-shared variant exceeds the full dynamic-memory '
                  'allocation and is rejected before launch. Both patches remain raw '
                  'and the production header is restored. B1 instead tests existing '
                  'whole-program candidates using nontrace P-SEL and P-PAIR.']
    cached = base/'boundary_cache_current_v1/result.json'
    if cached.exists():
        for trial in read(cached)['rows']:
            preparation = read(Path(trial['default0_proof']))
            proof = next(row for row in preparation['rows']
                         if row['batch'] == trial['batch'] and row['trace'] == 0)
            cohort = read(Path(trial['trace_receipt']))
            if (not proof['default0_proof']['passed'] or cohort['status'] != 'passed'
                    or len(cohort['processes']) != 3):
                raise ValueError('cached instrumentation requires production equivalence and three traces')
            disposition = ('qualifies for existing-candidate in-situ selection'
                           if trial['passed'] else 'fails5%;these windows never select coordinates')
            lines += ['', f'Verified current B{trial["batch"]} reuses the exact prior '
                      'header-padding boundary cache;nontrace instruction words equal the '
                      'accepted binary. Three fresh guarded GPU3 trace processes compare '
                      'output bytes with that binary;this is diagnostic coverage,not '
                      'five-process P-CORR. Boundary sum versus fresh nontrace P-SMOKE '
                      f'differs by{100*trial["relative_difference"]:.3f}%;{disposition}. '
                      'The production marker header and current artifacts remain unchanged.']
    incomplete = base/'boundary_cache_retune_b4_v1/incomplete_selection.json'
    if incomplete.exists():
        coverage = read(incomplete)
        lines += ['', 'Verified current72/208 B4 cached-boundary qualification passes, '
                  f'but only{len(coverage["eligible"])}/6 fixed candidates have both '
                  'production and boundary builds without spill. Candidate5 trace '
                  'is legal,but its nontrace build spills8/8B and never executes. '
                  'The other two rejected trace variants spill52/104B and96/180B. '
                  'Three legal variants each pass model5/5 and three matched-coordinate '
                  'byte-checked trace processes. Coverage below four prohibits an '
                  'assembled selection;the existing72/216 profile is qualified separately.']
    incomplete = base/'boundary_cache216_b8_v1/incomplete_selection.json'
    if incomplete.exists():
        coverage = read(incomplete)
        if coverage['gpu_executed'] or coverage['eligible'] != [0, 1, 2]:
            raise ValueError('B8 compile-only rejection differs from its stated coverage')
        lines += ['', 'Verified CPU-only B8 register-profile3 follow-up preserves the six '
                  'existing coordinate variants. Only0/1/2 have zero-spill production and '
                  'boundary builds;3/4/5 spill. Coverage remains3/6,below four,so no new '
                  'B8 GPU work or assembled stage selection is performed.']
    raw_tuned = base/'boundary_cache216_b4_v1/final_result.json'
    if raw_tuned.exists():
        result = read(raw_tuned)
        pair = result['pair']
        pending = not (base/'boundary_cache216_b4_admitted_v1/final_result.json').exists()
        lines += ['', 'Verified B4 fixed-candidate72/216 prototype on historical GPU3: '
                  f'P-PAIR{pair["reference_ms"]:.6f}→{pair["candidate_ms"]:.6f}ms;'
                  f'gain{100*(1-pair["ratio"]):.3f}% versus{100*pair["threshold"]:.3f}% noise. '
                  f'P-DNN{result["measurement"]["candidate_ms"]:.6f}ms. '
                  + ('This remains a prototype;complete four-program selection and fresh '
                     'explicit-profile component gates are pending. It does not replace '
                     'the current accepted B4 row.' if pending else
                     'Production admission and subsequent allocation are recorded separately '
                     'in the current artifact catalog.')]
    preparation = base/'first_ready_b2_preparation_v1/result.json'
    if preparation.exists():
        result = read(preparation)
        if not result['production_proof']['passed']:
            raise ValueError('first-task trace prototype changes production instructions')
        lines += ['', 'Verified CPU-only B2 trace prototype checks first_ready only on the '
                  'first local GEMM task;its clockpoint and four-marker layout are preserved. '
                  'Production instruction words exactly match the accepted control;trace '
                  'compilation has zero spill. Hardware qualification is pending;no '
                  'performance gain or valid stage windows are inferred.']
    retune = base/'phase72_retune_b8_gpu0_v1/incomplete_selection.json'
    if retune.exists():
        qualification = read(retune.parent/'trace_qualification.json')
        rejection = read(retune)
        assert qualification['passed']
        lines += ['', 'Verified profile2 B8 boundary overhead is'
                  f'{100*qualification["relative_difference"]:.3f}% on GPU0. '
                  f'Only{len(rejection["eligible"])}/6 existing variants are nonspilling;'
                  'each completes three fresh20/50 traces. Four spilled variants are '
                  'never executed. Required candidate coverage is incomplete;no assembled '
                  'selection or admission is claimed. GPU2 partial traces remain separate '
                  'and are not mixed with the GPU0 windows.']
    phase72 = base/'scalar_phase72_arithmetic_b8_v1/gpu2_diagnostic_v1/experiment.json'
    if phase72.exists():
        regression = read(base/'register72_host_v1/regression.json')
        equivalent = read(base/'register72_integrated_b8_v1/equivalence.json')
        assert regression['passed'] == 7 and equivalent['same_instruction_lines']
        lines += ['', 'Verified optional profile2 uses72/208 registers and explicit scalar phase '
                  'selection only for that profile. Profiles0/1 retain their prior paths and '
                  'the default remains0. Scalar/alias and P1 component cohorts each pass '
                  'unit5/5 and sanitizer1/1/5/5;B2/B4/B8/B16 whole models each pass5/5 '
                  'without spill. B8/B16 SASS is328576/395776B with zero local-memory '
                  'instructions. The integrated B8 kernel has identical instruction bytes '
                  'to its prototype. Seven related regression checks, real profile codegen '
                  'and invalid-setting checks pass. GPU2 comparisons remain diagnostics. '
                  'Complete GPU3 pairs subsequently admit B4/B8;exact-binary ledgers and '
                  'three-process P-DNN results are in the current catalog. B2 fails its '
                  'noise gate and B16 fails the firstRepeat3% gate. B8/B16 first-time '
                  'Repeat admission still requires3% plus noise;'
                  'B2/B4 already use accepted Repeat and require a gain beyond noise. '
                  'Fixed BF16 goals and original anchors remain unchanged.']
    spill_lines = []
    for name in ('repeat_arithmetic_middle_v1/b8', 'repeat_arithmetic_b8_reg56_v1'):
        log = base/name/'native/model.so.ptxas.log'
        if log.exists():
            spills = list(map(int, re.findall(r'(\d+) bytes spill (?:stores|loads)', log.read_text())))
            if any(spills):
                spill_lines.append(name+':'+','.join(str(n) for n in spills)+'B stores/loads')
    if spill_lines:
        lines += ['', 'Verified B8 arithmetic Repeat plus P1 normalization fails the compilation '
                  'gate under64/216 and prescribed56/224 ('+';'.join(spill_lines)+'). '
                  'Neither spilled binary is executed;profile2 later resolves this spill gate.']
    middle = base/'repeat_arithmetic_b4_reg56_v1/experiment.json'
    if middle.exists():
        decision = read(middle)
        pair = decision['pair']
        lines += ['', 'Verified B4 arithmetic Repeat plus P1 normalization retains the accepted '
                  'coordinates. The64/216 profile gains2.89%,above2.63% noise but below '
                  'DT-CODE3%,so it is rejected. Prescribed56/224 compiles without spill '
                  'and passes strict model5/5. Its GPU3 P-PAIR '
                  f'{pair["reference_ms"]:.6f}→{pair["candidate_ms"]:.6f}ms gives '
                  f'{100*(1-pair["ratio"]):.2f}% gain against '
                  f'{100*pair["threshold"]:.2f}% noise;the final P-DNN is recorded separately. '
                  'Arithmetic and P1 component units pass5/5 and sanitizer1/1/5/5 separately.']
        final = middle.parent/'g22_followup_final_v1/final_result.json'
        if final.exists():
            final = read(final)
            assert decision['accepted'] and not final['smoke_regression']
            measured = final['measurement']
            lines += ['', f'Verified B4 admitted P-DNN median{measured["candidate_ms"]:.6f}ms,'
                      f'p90{measured["p90_ms"]:.6f}ms,process range{measured["process_range_ms"]:.6f}ms. '
                      'SASS and identities are retained with the default56/224 profile;no production '
                      'header changes are needed. It remains above the fixed BF16 target and retains '
                      'the original GPU3 anchor without re-anchoring.']
    rejected_combination = base/'arithmetic_side_norm_b2_v1/native/model.so.ptxas.log'
    if rejected_combination.exists():
        spills = re.findall(r'(\d+) bytes spill (?:stores|loads)', rejected_combination.read_text())
        if any(map(int, spills)):
            lines += ['', 'Verified B2 P1 normalization plus the admitted arithmetic loop spills '
                      + '/'.join(spills)+'B stores/loads under64/216. The compilation gate rejects '
                      'it before execution;the admitted arithmetic-only B2 profile remains selected.']
    for batch in (2, 4):
        folder = base/f'current_selection_b{batch}_v1'
        if not (folder/'selection/result.json').exists():
            continue
        qualification = read(base/f'current_retune_b{batch}_v1/trace_qualification.json')
        selection = read(folder/'selection/result.json')
        lines += ['', f'Verified B{batch} current-profile boundary overhead '
                  f'{100*qualification["relative_difference"]:.3f}% exceeds5%;'
                  'those windows do not select stage coordinates. Four distinct nontrace '
                  'whole programs use only original Phase Z candidates and the admitted control. '
                  'Every candidate passes strict model5/5;spilled builds are never executed. '
                  f'G22 P-SEL winner={selection["winner"]},'
                  f'tie threshold={100*selection["threshold"]:.3f}%. '
                  'The admitted control remains selected unless a separate fresh P-PAIR '
                  'passes its unchanged noise gate.']
    equivalence = base/'partial_layout_equivalence_accepted_v1.json'
    if equivalence.exists():
        checked = read(equivalence)
        assert checked['passed'] and checked['passes'] == '5/5'
        assert len({r['pid'] for r in checked['processes']}) == 5
        assert len({r['binary_sha256'] for r in checked['processes']}) == 1
        assert len({r['output_hash'] for r in checked['processes']}) == 1
        assert all(read(Path(r['guard']))['code'] == 0 for r in checked['processes'])
        lines += ['', 'Verified original thread-major and opt-in vector-major split-K layouts '
                  'produce identical output bytes on six frozen shapes:odd rows,S3/4/6/16 '
                  'and TK64/128. The first five accepted fresh original-layout processes '
                  'use one frozen binary (four on GPU2,one on GPU3),each with guard0;'
                  'all match the accepted coalesced-suite output hash. This is a functional '
                  'allocation deviation with no mixed-card performance comparison.']
    resumed = base/'resume_gpu3_only_20261110_v1/result.json'
    if resumed.exists():
        lines += ['', 'Verified resumption: all new GPU tests use physical GPU3 only. '
                  'Previously completed component/model receipts are reused;historical '
                  'GPU0–2 queues remain inactive. The four pending candidates complete '
                  'fresh same-card pairs;none clears its acceptance gate. Accepted '
                  'profiles,original anchors and frozen Phase Z results are retained.']
    diagnosis = base/'ffn2_split_attribution_gpu3_v1/diagnosis.json'
    if diagnosis.exists():
        groups = read(diagnosis)['groups']
        a, b = groups['control'], groups['candidate']
        lines += ['', 'Verified rejected B8 FFN2 split2/TMA attribution on GPU3: '
                  f'FFN2 windows sum{a["stage_ns"]:.3f}→{b["stage_ns"]:.3f}us;'
                  f'main-loop CTA maxima sum{a["main_loop_max_cta_ns"]:.3f}→'
                  f'{b["main_loop_max_cta_ns"]:.3f}us;split-K maxima sum'
                  f'{b["splitk_max_cta_ns"]:.3f}us and epilogue maxima sum'
                  f'{a["epilogue_max_cta_ns"]:.3f}→{b["epilogue_max_cta_ns"]:.3f}us. '
                  'Segment maxima are nonadditive diagnostic attribution. The fixed '
                  'coalesced-partial follow-up has separate component/model/pair gates;'
                  'a main-loop improvement alone does not admit either combination.']
    async_rejection = base/'pp64_async_side_preparation_v1/component_rejection.json'
    if async_rejection.exists():
        failure = read(async_rejection)
        assert not failure['accepted'] and not failure['timing_executed']
        lines += ['', 'Verified private asynchronous PP side acquisition: normal unit5/5, '
                  'memcheck1/1,initcheck1/1 and racecheck5/5 pass on GPU3, but '
                  'synccheck0/1 reports8 Missing init errors. Reject before model or '
                  'timing;the partial cohort is not a complete17 gate. Default0 full '
                  'B16 instruction words match the accepted binary. The isolated '
                  'follow-up restores direct acquisition by all128 WG consumers;'
                  'its gates remain separate.']
    allwait_rejection = base/'pp64_async_side_allwait_preparation_v1/component_rejection.json'
    if allwait_rejection.exists():
        failure = read(allwait_rejection)
        assert not failure['accepted'] and not failure['timing_executed']
        lines += ['', 'Verified direct acquisition by all128 consumers also fails: '
                  'unit5/5,mem1/1,init1/1,race5/5 pass on GPU3; synccheck0/1 '
                  'reports1024 Missing init errors. Reject before model/timing. '
                  'The next private refinement retains the previously complete '
                  'named160 handoff and makes only the producer arrival nonblocking;'
                  'its full protocol and end-to-end gates are independent.']
    nonblocking = base/'pp64_arrive_side_b16_v2/experiment.json'
    if nonblocking.exists():
        trial = read(nonblocking)
        lines += ['', 'Verified nonblocking named160 PP side handoff passes two fresh '
                  'GPU3 component cohorts17/17 each and strict model5/5. Default0 '
                  'full B16 instruction words remain identical to the incumbent. '
                  f'GPU3 full-model pair ratio{trial["pair"]["ratio"]:.6f};'
                  f'noise{100*trial["pair"]["threshold"]:.3f}%,DT-PP minimum3%. '
                  'The synchronization repair is functional evidence;performance '
                  'and production admission follow the independent end-to-end gate.']
    current = args.out/'DNN4_BERT_current.csv'
    integrated = base/'coalesced_cyclic_b1_integrated_v1/final_result.json'
    if integrated.exists():
        final = read(integrated)
        pair, measured = final['pair'], final['measurement']
        lines += ['', 'Verified opt-in cyclic slot indexing and vector-major normal split-K '
                  'partials retain the admitted B1 coordinates,56/224 register roles and '
                  'ascending slice addition order. Integrated component gates pass17/17 '
                  'on GPU3;the extended S2/TK128/odd-row suite passes17/17 on GPU2. '
                  'The full model passes strict FP32 and cross-process byte checks5/5. '
                  f'G22 P-PAIR{pair["reference_ms"]:.6f}→{pair["candidate_ms"]:.6f}ms '
                  f'passes{100*pair["threshold"]:.3f}% noise;P-DNN '
                  f'{measured["candidate_ms"]:.6f}ms,p90{measured["p90_ms"]:.6f}ms,'
                  f'process range{measured["process_range_ms"]:.6f}ms. '
                  'The exact accepted binary has a P-TRACE1 ledger. '
                  'The same B2 mechanism fails its paired noise gate and is not enabled there. '
                  'Defaults,original anchors,CNN results and frozen20-cell Phase Z remain unchanged.']
    if current.exists():
        with current.open() as stream:
            latest = list(csv.DictReader(stream))
        wins = [r['batch'] for r in latest if float(r['median_ms']) < float(r['baseline_ms'])]
        target_status = ('Fixed BF16 targets reached for B'+',B'.join(wins)+'. '
                         if wins else 'All fixed BF16 targets remain unmet. ')
        cards = ','.join('GPU'+gpu for gpu in sorted({r['gpu'] for r in latest}))
        lines += ['', f'Verified current accepted artifacts ({cards};50/200;three-process P-DNN). '
                  'B1/B2/B4/B8/B16 are follow-ups that retain '
                  'the original anchor without re-anchoring. '+target_status+
                  'DNN4_BERT_current.csv records complete identities,profiles and evidence paths;'
                  'DNN4_BERT_current_ledger.csv preserves all315 current stage rows with exact '
                  'accepted reference hashes and coordinates.', '',
                  '| Batch | GPU | Median ms | p90 ms | Process range ms | Fixed BF16 ms | Ratio |',
                  '| --- | --- | --- | --- | --- | --- | --- |']
        for row in latest:
            lines.append(f'| {row["batch"]} | {row["gpu"]} | {float(row["median_ms"]):.6f} | '
                         f'{float(row["p90_ms"]):.6f} | {float(row["process_range_ms"]):.6f} | '
                         f'{float(row["baseline_ms"]):.4f} | {float(row["ratio"]):.4f} |')
        lines += ['', 'Verified largest three stages for each current unmet BERT cell '
                  '(P-TRACE1 attribution only;segments can overlap and are not timing results). '
                  'Each ledger checks the exact accepted reference binary SHA256.', '',
                  '| Batch | Largest stage | Second | Third |',
                  '| --- | --- | --- | --- |']
        for row in latest:
            if float(row['median_ms']) < float(row['baseline_ms']):
                continue
            with Path(row['ledger_source']).open() as stream:
                stages = list(csv.DictReader(stream))
            descriptions = []
            for stage in sorted(stages, key=lambda s: float(s['stage_ns']), reverse=True)[:3]:
                segments = sorted(((key.removesuffix('_max_cta_ns'), float(value))
                                   for key, value in stage.items() if key.endswith('_max_cta_ns')),
                                  key=lambda item: item[1], reverse=True)[:2]
                descriptions.append(f's{stage["stage"]} {stage["body"]} '
                    f'{float(stage["stage_ns"])/1000:.2f}us ('+
                    ','.join(f'{name}={ns/1000:.2f}' for name, ns in segments)+')')
            lines.append(f'| {row["batch"]} | '+' | '.join(descriptions)+' |')
    failed_pp = base/'pp64_ffn1_b16_v1/experiment.json'
    if failed_pp.exists() and read(failed_pp).get('status') == 'component_rejected':
        lines.append('\nVerified fixed WG64 PP/P1/TMA candidate compiles without spill and both normal '
                     'unit cohorts pass5/5, but ordering synccheck reports1024 errors and residual-chain '
                     'racecheck reports21 errors. Reject before model correctness or timing; '
                     'partial sanitizer cohorts are not complete gates. Frozen failures remain raw; '
                     'current profiles and targets are unchanged.')
    (args.out/'DNN4_BERT_optimization.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(comparisons=len(rows), report=str(args.out/'DNN4_BERT_optimization.md'))))


if __name__ == '__main__':
    main()
