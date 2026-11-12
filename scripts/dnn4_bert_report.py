#!/usr/bin/env python3
"""Publish guarded BERT optimization comparisons separately from final cells."""
import argparse
from collections import Counter
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
              ('cached-boundary216 B4 admitted continuation',
               base/'boundary_cache216_b4_admitted_v1/experiment.json'),
              ('Current prefetch1 B16', base/'current_prefetch_b16_v1/experiment.json'),
              ('FFN2 register store B8', base/'ffn2_reg_b8_v1/experiment.json'),
              *[(f'phase72 B{batch} GPU2 diagnostic',
                 base/f'scalar_phase72_arithmetic_b{batch}_v1/gpu2_diagnostic_v1/experiment.json')
                for batch in (2, 4, 8, 16)],
              *[(f'phase72 B{batch} G22',
                 base/f'scalar_phase72_arithmetic_b{batch}_v1/experiment.json')
                for batch in (2, 4, 8, 16)]]
    continuation = base/'resume_allcards_20261112_v1'
    if not (base/'boundary_cache216_b4_admitted_v1/experiment.json').exists():
        probes.append(('fixed-candidate216 B4 revalidation', continuation/'b4_experiment.json'))
    probes += [('existing whole-program216 B8', continuation/'b8_experiment.json'),
               ('partial TN96 store B8 prototype', continuation/'store96_experiment.json'),
               ('partial TN96 matrix store B8 prototype', continuation/'store96_matrix_experiment.json'),
               ('QKV four-slot B2', continuation/'qkv_slots4_experiment.json'),
               ('QKV seven-slot B4', continuation/'qkv_slots7_experiment.json'),
               ('QKV plus attention stores B2', continuation/'qkv_attention_experiment.json'),
               ('first WGMMA initialization B16', continuation/'mma_zero_experiment.json'),
               ('fixed H800 grid B16', continuation/'fixed_grid_experiment.json'),
               ('fixed grid producer recomputation B8', continuation/'grid_recount_experiment.json'),
               ('producer plus four-slot QKV B2', continuation/'recount_qkv4_experiment.json'),
               *[(f'producer recomputation B{b}', continuation/f'grid_recount_b{b}_experiment.json') for b in (2,4,16)],
               ('paired residual read B16 prototype', continuation/'residual_experiment.json'),
               ('paired residual read B16 GPU0 diagnostic', continuation/'residual_gpu0_experiment.json')]
    probes += [('full-tile B8', continuation/'full_tiles_b8_v2/experiment.json'),
               ('full-tile B16', continuation/'full_tiles_b16_selector_v1/experiment.json')]
    probes += [(f'full-tile B{b}', continuation/f'full_tiles_b{b}_v1/experiment.json')
               for b in (2, 4)]
    probes += [('full-tile pipeline B2', continuation/'pipeline_full_b2_v1/experiment.json'),
               ('full-bounds prefetch1 B8', continuation/'full_tiles_prefetch1_b8_v1/experiment.json'),
               ('full-bounds pipeline prefetch1 B2', continuation/'pipeline_prefetch1_b2_v1/experiment.json'),
               ('full-tile stores B8', continuation/'full_store_bundle_b8_v1/experiment.json'),
               ('full-tile stores B4', continuation/'full_store_bundle_b4_v1/experiment.json'),
               ('pipeline reductions B2', continuation/'pipeline_reduction_b2_v1/experiment.json'),
               ('full-store bounds B8', continuation/'full_store_bounds_b8_v1/experiment.json'),
               ('full-profile cyclic B4', continuation/'cyclic_full_b4_v1/experiment.json'),
               ('GELU quartet B16', continuation/'gelu_quartet_b16_v1/experiment.json'),
               ('relative descriptors B16', continuation/'descriptor_base_b16_v2/experiment.json'),
               ('two-row statistics plus recompute B16', continuation/'side_rows_b16_recompute_v3/experiment.json'),
               ('repaired catalog candidate5 B16', continuation/'candidate5_recompute_b16_v2/experiment.json'),
               ('statistics read ahead B16', continuation/'stats_ahead_b16_v1/experiment.json'),
               ('shared terminal FFN2 B16', continuation/'optional_stats_b16_v2/experiment.json'),
               ('M-first task traversal B16', continuation/'mfirst_b16_v1/experiment.json'),
               ('FFN2 fixed split2 B4', continuation/'ffn2_split2_b4_v1/experiment.json'),
               ('main-loop pair B4', continuation/'loop_pair_b4_v1/experiment.json'),
               ('repaired profile3 catalog candidate2 B8', continuation/'candidate2_profile3_b8_v1/experiment.json'),
               ('residual warp handoff B16', continuation/'residual_warp_b16_v1/experiment.json'),
               ('direct statistics B16', continuation/'direct_norm_b16_v3/experiment.json')]
    rows = []
    choices = []
    for name, path in probes:
        if not path.exists():
            continue
        experiment = read(path)
        if name in ('full-tile stores B8', 'full-tile stores B4',
                    'pipeline reductions B2', 'direct statistics B16',
                    'full-store bounds B8', 'full-profile cyclic B4', 'GELU quartet B16',
                    'relative descriptors B16', 'two-row statistics plus recompute B16',
                    'repaired catalog candidate5 B16', 'statistics read ahead B16',
                    'main-loop pair B4', 'repaired profile3 catalog candidate2 B8',
                    'residual warp handoff B16', 'shared terminal FFN2 B16',
                    'M-first task traversal B16', 'FFN2 fixed split2 B4'):
            for unit_path in experiment['unit_gates']:
                unit = read(Path(unit_path))
                processes = unit['processes']
                tools = Counter(p['name'].split('-')[0] for p in processes)
                if (not unit['passed'] or len({p['pid'] for p in processes}) != 17 or
                        tools != Counter(unit=5, memcheck=1, initcheck=1,
                                         racecheck=5, synccheck=5) or
                        any(p['exit_code'] or read(Path(p['guard']))['code']
                            for p in processes)):
                    raise ValueError('current-profile follow-up lacks its complete component gate')
            if name == 'FFN2 fixed split2 B4':
                # A split change reassociates FP32 additions; P-CORR still
                # requires the same reference threshold and deterministic bytes.
                model = read(path.parent/'model/result.json')
                preparation = read(Path(experiment['preparation']))
                if (not model['passed'] or model['passes'] != '5/5' or
                        not read(Path(preparation['control_proof']))['passed']):
                    raise ValueError('split-K follow-up lacks its model or control gate')
            elif not read(path.parent/'model_bit_proof.json')['passed']:
                raise ValueError('current-profile follow-up lacks incumbent output equivalence')
            if name in ('pipeline reductions B2', 'full-store bounds B8',
                        'full-profile cyclic B4', 'GELU quartet B16', 'relative descriptors B16',
                        'two-row statistics plus recompute B16', 'repaired catalog candidate5 B16',
                        'statistics read ahead B16', 'main-loop pair B4',
                        'repaired profile3 catalog candidate2 B8', 'residual warp handoff B16',
                        'shared terminal FFN2 B16', 'M-first task traversal B16',
                        'FFN2 fixed split2 B4'):
                bit = read(path.parent/'unit_bit_proof.json')
                control = read(Path(bit['control']))
                candidate = read(Path(bit['candidate']))
                if (not bit['passed'] or not control['passed'] or
                        len({p['pid'] for p in control['processes']}) != 5 or
                        control['output_hash'] != candidate['output_hash']):
                    raise ValueError('pipeline reductions lack control equivalence')
            if name == 'full-tile stores B4':
                reuse = read(Path(experiment['component_reuse']))
                fixtures = read(Path(reuse['fixtures']))
                if (not reuse['passed'] or not fixtures['passed'] or
                        reuse['units'] != experiment['unit_gates'] or
                        any(row['profile'] != reuse['profile'] for row in fixtures['rows']) or
                        read(Path(reuse['admission']))['status'] != 'admitted'):
                    raise ValueError('B4 store bundle lacks matching inherited component profiles')
            if name in ('direct statistics B16', 'full-store bounds B8', 'GELU quartet B16',
                        'relative descriptors B16', 'two-row statistics plus recompute B16',
                        'statistics read ahead B16', 'main-loop pair B4',
                        'residual warp handoff B16', 'shared terminal FFN2 B16',
                        'M-first task traversal B16'):
                preparation = read(Path(experiment['preparation']))
                if (not read(Path(preparation['default_proof']))['passed'] or
                        not read(path.parent/'unit_bit_proof.json')['passed']):
                    raise ValueError('private implementation requires default and unit equivalence')
        if name.startswith('full-tile '):
            unit_paths = experiment.get('unit_gates')
            if unit_paths is None:
                unit_paths = [p['candidate'] for p in experiment['unit_bit_proofs']]
            for unit_path in unit_paths:
                unit = read(Path(unit_path))
                if (not unit['passed'] or len(unit['processes']) != 17 or
                        any(p['exit_code'] or read(Path(p['guard']))['code']
                            for p in unit['processes'])):
                    raise ValueError('full-tile comparison lacks complete component gates')
        admitted_continuation = name == 'cached-boundary216 B4 admitted continuation'
        gpu0_continuation = admitted_continuation and experiment['gpu'] == 0
        if admitted_continuation:
            sys.path.insert(0, str(ROOT/'scripts'))
            from dnn4_bert_current import validate_boundary_selection, validate_followup_allocation
            final = read(path.parent/'final_result.json')
            validate_followup_allocation(path.parent, path.parent.name, experiment, final)
            validate_boundary_selection(path.parent, experiment['library'], experiment['reference'])
            if not experiment['accepted']:
                raise ValueError('admitted comparison lacks its completed acceptance')
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
                    'FFN1 PP nonblocking side B16', 'Current coalesced B2',
                    'cached-boundary216 B4 admitted continuation',
                    'existing whole-program216 B8', 'partial TN96 store B8 prototype',
                    'partial TN96 matrix store B8 prototype',
                    'QKV four-slot B2', 'QKV seven-slot B4', 'QKV plus attention stores B2',
                    'first WGMMA initialization B16',
                    'fixed H800 grid B16',
                    'fixed grid producer recomputation B8',
                    'paired residual read B16 prototype',
                    'paired residual read B16 GPU0 diagnostic'):
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
        if name in ('Current prefetch1 B16', 'full-bounds prefetch1 B8',
                    'full-bounds pipeline prefetch1 B2'):
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
        if name == 'paired residual read B16 prototype':
            pair_path = path.parent/'residual_pair/result.json'
        elif name == 'paired residual read B16 GPU0 diagnostic':
            pair_path = path.parent/'residual_pair_gpu0_diagnostic/result.json'
        if not pair_path.exists() and (path.parent/'winner_pair/result.json').exists():
            pair_path = path.parent/'winner_pair/result.json'
        if not pair_path.exists():
            pair_path = path.parent/f'pair_gpu{experiment["gpu"]}/result.json'
        first = read(pair_path)
        comparisons = [(('User-authorized GPU0 continuation;same-card pair;fixed GPU3 target '
                         'retained;not re-anchored' if gpu0_continuation else
                         f'GPU{first["gpu"]} same-card diagnostic'), pair_path)]
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
                accepted=(pair['gpu'] != 0 or gpu0_continuation) and pair['ratio']<1-max(pair['threshold'],
                    experiment.get('minimum_gain', 0),
                    .03 if name.startswith('Repeat') or name=='shared terminal FFN2 B16' or
                        (name.startswith('phase72') and experiment['batch'] in (8, 16)) else 0), scope=scope,
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
                  'boundary builds;3/4/5 spill. Coverage remains3/6,below four;these '
                  'partial windows cannot assemble stage coordinates. A separate whole-program '
                  'fallback may compare the three legal programs with the incumbent after '
                  'strict model5/5;an ambiguous selection retains the incumbent.']
    raw_tuned = base/'boundary_cache216_b4_v1/final_result.json'
    if raw_tuned.exists():
        result = read(raw_tuned)
        pair = result['pair']
        pending = not (base/'boundary_cache216_b4_admitted_v1/final_result.json').exists()
        lines += ['', 'Verified B4 fixed-candidate72/216 prototype on historical GPU3: '
                  f'P-PAIR{pair["reference_ms"]:.6f}→{pair["candidate_ms"]:.6f}ms;'
                  f'gain{100*(1-pair["ratio"]):.3f}% versus{100*pair["threshold"]:.3f}% noise. '
                  f'P-DNN{result["measurement"]["candidate_ms"]:.6f}ms. '
                  + ('This remains a prototype;the current continuation separately checks '
                     'four-program selection,explicit-profile components,independent paired '
                     'timing and final admission. It does not replace the current accepted '
                     'B4 row.' if pending else
                     'Production admission and subsequent allocation are recorded separately '
                     'in the current artifact catalog.')]
    preparation = base/'first_ready_b2_preparation_v1/result.json'
    if preparation.exists():
        result = read(preparation)
        if not result['production_proof']['passed']:
            raise ValueError('first-task trace prototype changes production instructions')
        qualified = base/'first_ready_b2_gpu0_v2/trace_qualification.json'
        status = ('Hardware qualification is pending;no performance gain or valid stage '
                  'windows are inferred.')
        skipped = base/'first_ready_b2_gpu0_v2/result.json'
        if skipped.exists() and read(skipped).get('status') == 'skipped_no_selection_coverage':
            if read(skipped)['trace_processes_executed'] != 0:
                raise ValueError('skipped B2 trace unexpectedly executed on hardware')
            status = ('Hardware qualification was skipped before all three trace processes: '
                      'all48 normal GEMM stages have at most one task per CTA,so the first-task '
                      'predicate removes no repeated baseline checks. The fresh GPU0 reference '
                      'smoke0.572192ms is diagnostic only;no new performance is admitted.')
        if qualified.exists():
            result = read(qualified)
            status = (f'Three independent GPU0 boundary processes give '
                      f'{100*result["relative_difference"]:.3f}% overhead against a fresh '
                      'same-card reference;'+('passes5%;eligible for fixed-candidate selection.'
                      if result['passed'] else 'fails5%;these windows never select coordinates.'))
        lines += ['', 'Verified B2 trace prototype checks first_ready only on the '
                  'first local GEMM task;its clockpoint and four-marker layout are preserved. '
                  'Production instruction words exactly match the accepted control;trace '
                  'compilation has zero spill. '+status]
    fixed = base/'first_ready_retune_b2_preparation_v1/compiled.json'
    if fixed.exists():
        result = read(fixed)
        if result['gpu_executed'] or result['eligible'] != [0, 1]:
            raise ValueError('B2 fixed-candidate CPU coverage differs from its evidence')
        lines += ['', 'Verified CPU-only B2 first-task trace preparation keeps all six '
                  'existing coordinate variants and the accepted72/216 profile. Only0/1 '
                  'have zero-spill production and boundary builds;2/3/4/5 spill. Coverage2/6 '
                  'prohibits assembled selection;baseline trace qualification is independent.']
    component = base/'resume_gpu0_intermittent_20261111_v1/alias_unit/result.json'
    if component.exists():
        result = read(component)
        if not result['passed'] or len(result['processes']) != 17 or result['gpu'] != 0:
            raise ValueError('GPU0 alias prerequisite lacks its complete17 processes')
        lines += ['', 'Verified explicit72/216 alias component on GPU0 passes unit5/5 '
                  'and sanitizer mem1/1,init1/1,race5/5,sync5/5 in seventeen independent '
                  'guarded reservations with released gaps. The resource.py generic entry '
                  'was renamed worker.py after two model jobs failed before measurement '
                  'because it shadowed the standard library;failed logs remain raw. '
                  'No numerical or timing thresholds change,and completed component work '
                  'is reused after the server restart.']
    role = base/'role_operands_b8_preparation_v1/result.json'
    if role.exists():
        result = read(role)
        if result['accepted'] or result['gpu_executed'] or any(r['legal'] for r in result['rows']):
            raise ValueError('producer operand compile rejection differs from its evidence')
        lines += ['', 'Verified private B8 producer operand lifetime trial keeps side bindings '
                  'inside P1 and preserves the fixed coordinates. Candidates3/4 still spill '
                  '24/32B and28/52B stores/loads versus20/28B and24/48B before the change. '
                  'Reject at the CPU gate;no trace builds,hardware execution or production change.']
    counts = base/'local_counts_b2_preparation_v1/result.json'
    if counts.exists():
        result = read(counts)
        if result['accepted'] or result['gpu_executed'] or any(r['legal'] for r in result['rows']):
            raise ValueError('stage index lifetime rejection differs from its evidence')
        lines += ['', 'Verified B2 fixed4 producer index lifetime drafts preserve coordinates '
                  'and arithmetic. Recomputing task counts reduces spill8/8B to4/4B;also '
                  'recomputing task addresses returns to8/8B. Both fail the zero-spill '
                  'gate and never execute on GPU. The main header is unchanged.']
    authorization = continuation/'authorization.json'
    if authorization.exists():
        allocation = read(authorization)
        if allocation['allowed_gpu_indices'] != [0, 1, 2, 3]:
            raise ValueError('four-card continuation differs from explicit user allocation')
        lines += ['', 'Stated2026-11-12 allocation:all four H800 cards0–3 are authorized;'
                  'intermittent gaps are no longer required. New schedules preserve the '
                  'six-sample guard,300s occupancy cooldown and /root/python entry paths. '
                  'B4/B8 timing uses original GPU3;functional checks may use the other '
                  'cards. Previously paused queues stay retired and completed checks are reused.']
    relocation = continuation/'b8_corr2_relocation.json'
    idle_change = continuation/'idle6_transition.json'
    if idle_change.exists():
        observed = read(continuation/'idle_observation_v1/result.json')
        transition = read(idle_change)
        if (not observed['cards']['3']['owners_free']
                or observed['cards']['3']['used_mib'] != [6.0] * 6
                or transition['old_state']['component']['exit_code'] != 75):
            raise ValueError('GPU3 idle recalibration lacks six owner-free samples')
        lines += ['', 'Verified GPU3 idle footprint changed from4MiB to6MiB: '
                  'six owner-free samples5s apart record6MiB and idle SM345MHz. '
                  'A separate guard entry updates only that measured floor;original '
                  'workers remain frozen. Locks,six-sample preflight,foreign-process '
                  'rejection,cooldowns and measurement thresholds remain in force. '
                  'The pending body cohort has no accepted processes;its replacement '
                  'inherits the cooldown. Completed component/model cohorts are reused. '
                  'Timing continues to record actual start/end SM clocks.']
    if relocation.exists():
        moved = read(relocation)
        if (moved['completed_processes_discarded'] != 0
                or any(s['attempts'] or s['status'] != 'pending'
                       for s in moved['states'].values())):
            raise ValueError('B8 relocation was not a wholly unstarted cohort')
        lines += ['', 'Verified foreign GPU1 occupancy prevents dispatch of one B8 '
                  'functional cohort. All five jobs have zero attempts and no guarded '
                  'children when its controllers retire. The cohort relocates to GPU0 '
                  'under new original guards;the other two live cohorts and all '
                  'completed checks are retained. Performance timing stays on GPU3.']
    residual = base/'residual_pair_b16_preparation_v1/result.json'
    if residual.exists() and read(residual)['status'] == 'prepared':
        compiled = read(residual)
        lines += ['', 'Verified private fixed-coordinate B16 residual-read prototype compiles '
                  f'without spill;SASS{compiled["model"]["sass_bytes"]}B. One aligned shared32 '
                  'read supplies each adjacent BF16 pair while preserving per-element '
                  'arithmetic. CPU compilation establishes feasibility only;complete '
                  'unit/sanitizer,model5/5 and same-GPU3 whole-model gates decide admission.']
    for suite, label in [('side_finish', 'explicit72/216 P1 normalization'),
                         ('residual_unit', 'private paired-residual-read')]:
        receipt = continuation/suite/'result.json'
        if not receipt.exists():
            continue
        result = read(receipt)
        processes = result['processes']
        expected = Counter(unit=5, memcheck=1, initcheck=1, racecheck=5, synccheck=5)
        if (not result['passed'] or len(processes) != 17
                or len({p['pid'] for p in processes}) != 17
                or Counter(p['name'].split('-')[0] for p in processes) != expected
                or len({p['binary_sha256'] for p in processes}) != 1
                or len({p['output_hash'] for p in processes}) != 1
                or any(p['exit_code'] or read(Path(p['guard']))['code'] for p in processes)):
            raise ValueError('four-card functional receipt lacks its complete17 processes')
        lines += ['', f'Verified {label} component on GPU{result["gpu"]}:unit5/5,'
                  'memcheck1/1,initcheck1/1,racecheck5/5,synccheck5/5;seventeen distinct '
                  'native processes have identical output hashes and clean guard receipts. '
                  + ('Four completed unit processes are reused;the remaining thirteen '
                     'run under one original guarded reservation after the inherited cooldown. '
                     if suite == 'side_finish' else
                     'All seventeen run under one original guarded reservation. ')
                  + 'These are functional prerequisites;whole-model correctness and '
                  'same-GPU3 timing decide performance admission.']
    for suite in ('b8_corr1', 'b8_corr2', 'b8_corr3', 'residual_corr', 'store96_corr', 'store96_matrix_corr', 'qkv_slots4_corr', 'qkv_slots7_corr', 'qkv_attention_corr', 'mma_zero_corr', 'fixed_grid_corr', 'grid_recount_corr', 'grid_recount_b2_corr', 'grid_recount_b4_corr', 'grid_recount_b16_corr', 'recount_qkv4_corr'):
        receipt = continuation/suite/'result.json'
        if not receipt.exists():
            continue
        result = read(receipt)
        processes = result['processes']
        if (not result['passed'] or result['passes'] != '5/5'
                or len(processes) != 5 or len({p['pid'] for p in processes}) != 5
                or any(p['exit_code'] or read(Path(p['guard']))['code'] for p in processes)):
            raise ValueError('continuation model receipt lacks five accepted processes')
        minimum = min(output['cosine_min'] for case in result['cases']
                      for output in case['outputs'])
        cards = result.get('gpu_indices', [result['gpu']])
        gpu_label = '/'.join(map(str, cards))
        lines += ['', f'Verified {suite} on GPU{gpu_label}:strict FP32 model5/5,'
                  f'minimum cosine{minimum:.8f},identical output bytes across all five '
                  'fresh processes. These completed correctness checks do not establish '
                  'whole-model performance or production admission.']
    paired = continuation/'b4_experiment.json'
    if paired.exists() and not read(paired)['accepted']:
        decision = read(paired); pair = decision['pair']
        lines += ['', 'Verified B4 fixed-candidate revalidation completes fresh GPU3 '
                  'P-SEL8/8 and independent P-PAIR6/6. The selected candidate measures '
                  f'{pair["candidate_ms"]:.6f}ms against{pair["reference_ms"]:.6f}ms;'
                  f'gain{100*(1-pair["ratio"]):.3f}% is below'
                  f'{100*pair["threshold"]:.3f}% process-noise threshold. Retain all six '
                  'processes and the earlier prototype evidence;do not replace the '
                  'admitted B4 result or rerun this pair to seek acceptance.']
    fixed_residual = base/'residual_pair_b8_candidate3_preparation_v1/result.json'
    if fixed_residual.exists():
        result = read(fixed_residual)
        if result['legal'] or result['gpu_executed'] or result['spill_bytes'] != [20, 28]:
            raise ValueError('fixed B8 residual-read rejection differs from its compile evidence')
        lines += ['', 'Verified CPU-only residual-pair follow-up on the existing fixed '
                  'B8 candidate3 retains its coordinates and72/216 profile. It still spills '
                  '20/28B stores/loads,matching the original candidate,and never executes '
                  'on GPU. The three legal whole-program alternatives remain unchanged.']
    whole_b8 = continuation/'b8_result.json'
    if whole_b8.exists() and read(whole_b8)['status'] == 'retain_current':
        result = read(whole_b8); selected = result['selection']
        if (selected['gpu'] != 3 or len(selected['measurements']) != 8
                or any(r['guard']['code'] for r in selected['measurements'])):
            raise ValueError('B8 retention lacks its complete same-card selection')
        timings = ','.join(f'{r["arm"]}={r["median_ms"]:.6f}ms' for r in selected['ranking'])
        lines += ['', 'Verified B8 nontrace whole-program P-SEL8/8 on GPU3: '+timings+
                  f';tied={selected["tied"]},noise{100*selected["threshold"]:.3f}%. '+
                  ('Retain the incumbent because the winner is ambiguous. '
                   if selected['tied'] else 'The incumbent wins and remains selected. ')+
                  'Incomplete stage-window coverage supplies no tie priority or assembled selection.']
    partial_store = base/'store96_profile3_model_preparation_v1/result.json'
    if partial_store.exists() and read(partial_store)['status'] == 'prepared':
        result = read(partial_store)
        proof = read(base/'store96_model_preparation_v1/control_proof.json')
        if not proof['passed'] or result['changed_stages'] != list(range(4, 60, 5)):
            raise ValueError('partial TN96 store must preserve its fixed12-stage scope')
        lines += ['', 'Verified private partial TN96 store extends section6.2 under the '
                  'user-authorized implementation discretion:only the first64 columns use '
                  'TMA;the remaining32 use generic stores,with the full-box bound enforced '
                  'inside each WG tile. The12 B8 FFN1 stages change Slots7 to6 to fit '
                  '218112B including staging;Slots7 would require246784B and is illegal. '
                  'The current72/208 whole-model build spills4/20B and never executes. '
                  f'The existing72/216 profile compiles without spill,SASS{result["model"]["sass_bytes"]}B. '
                  'Regeneration with unchanged coordinates/profile matches all41072 encoded '
                  'instruction words of the incumbent. Solver enumeration is unchanged;'
                  'exact-profile body/order gates,model5/5 and a same-GPU3 gain above3% '
                  'and noise are required before enabling this extension.']
        for suite in ('store96_body', 'store96_profile3_ordering', 'store96_profile3_body',
                      'store96_matrix_ordering', 'store96_matrix_body'):
            source = continuation/suite/'result.json'
            if not source.exists():
                continue
            unit = read(source)
            if not unit['passed'] or len(unit['processes']) != 17:
                raise ValueError('partial store component cohort is incomplete')
            lines += [f'Verified {suite}:unit5/5,mem1/1,init1/1,race5/5,sync5/5 '
                      f'on GPU{unit["gpu"]};all seventeen fresh processes have identical '
                      'output hashes. Functional evidence does not establish E2E benefit.']
        attribution = continuation/'store96_attribution.json'
        if attribution.exists():
            summary = read(attribution)
            ffn1 = next(r for r in summary['groups'] if r['group'] == 'ffn1')
            old, new = ffn1['reference_us'], ffn1['candidate_us']
            lines += ['Verified rejected partial-store attribution:the12 FFN1 windows sum'
                      f'{old["stage_ns"]:.3f}→{new["stage_ns"]:.3f}us;main-loop CTA maxima sum'
                      f'{old["main_loop_max_cta_ns"]:.3f}→{new["main_loop_max_cta_ns"]:.3f}us;'
                      f'epilogue{old["epilogue_max_cta_ns"]:.3f}→{new["epilogue_max_cta_ns"]:.3f}us;'
                      f'store{old["store_max_cta_ns"]:.3f}→{new["store_max_cta_ns"]:.3f}us. '
                      'The historical same-GPU3 reference and fresh candidate trace are '
                      'diagnostic;segment maxima overlap. These data do not support a '
                      'large main-loop penalty from losing one slot;the store reduction '
                      'is small and total epilogue increases. Nontrace pair rejects the candidate.']
        matrix = base/'store96_matrix_model_preparation_v1/result.json'
        if matrix.exists() and read(matrix)['status'] == 'prepared':
            compiled = read(matrix)
            scope = read(continuation/'store96_matrix_scope.json')
            if (compiled['changed_stages'] != list(range(4, 60, 5))
                    or len(scope['existing_tma_stages']) != 23):
                raise ValueError('partial matrix-store follow-up changed its scope')
            lines += ['Verified CPU-only follow-up retains those coordinates and72/216 '
                      'roles while enabling the existing matrix-store1 path;SASS'
                      f'{compiled["model"]["sass_bytes"]}B,zero spill. The flag affects '
                      'both12 partial FFN1 stores and23 existing TMA stages. This tests '
                      'the observed epilogue cost without a slot/tile scan;separate '
                      'matrix-store1 component/model gates and the predeclared3%-plus-noise '
                      'whole-model gate decide acceptance.']
        conditional = base/'store96_b16_model_preparation_v1/result.json'
        if conditional.exists() and read(conditional)['status'] == 'prepared':
            compiled = read(conditional)
            if compiled['changed_stages'] != list(range(6, 60, 5)) or not compiled['control_proof']['passed']:
                raise ValueError('conditional B16 partial store scope differs')
            lines += ['Verified CPU-only B16 preparation applies the same partial store '
                      'to11 repeated QKV stages,keeping the first NT1 QKV and the admitted '
                      'profile unchanged. The control matches all44736 encoded words;'
                      f'the candidate has zero spill and{compiled["model"]["sass_bytes"]}B SASS. '
                      'No GPU checks or performance are claimed;this follow-up is conditional '
                      'on B8 mechanism qualification and needs separate matrix-store1 gates.']
    qkv = base/'qkv_slots4_b2_model_preparation_v1/result.json'
    qkv_unit = base/'qkv_slots4_b2_units_v1/result.json'
    if qkv.exists() and qkv_unit.exists() and read(qkv)['status'] == 'prepared':
        compiled, resource = read(qkv), read(qkv_unit)
        if compiled['changed_stages'] != list(range(6, 60, 5)) or not compiled['control_proof']['passed']:
            raise ValueError('B2 four-slot candidate changes more than repeated QKV')
        if resource['rows'][1]['payload_bytes'] != 111616 or resource['rows'][1]['spanning']:
            raise ValueError('B2 four-slot resource boundary differs')
        lines += ['', 'Verified CPU-only B2 pipeline follow-up applies the seed one-arena '
                  'maximum to the admitted128x64 QKV tile:11 repeated stages change '
                  'Slots3 to4;the first QKV and all other coordinates/profile remain fixed. '
                  'The canonical resource function gives87040/111616/136192B for3/4/5 '
                  'slots;four fits one arena and five spans. Current QKV attribution '
                  'averages8.33us main loop with4.55us slot wait. The fixed candidate '
                  f'has zero spill and{compiled["model"]["sass_bytes"]}B SASS;'
                  'regenerated control matches all33120 encoded words. This is one '
                  'seed-rule choice,not a parameter scan;component17/model5 and '
                  'independent same-GPU3 E2E gain above noise decide acceptance.']
    qkv4 = base/'qkv_slots7_b4_model_preparation_v1/result.json'
    if qkv4.exists() and read(qkv4)['status'] == 'prepared':
        compiled = read(qkv4)
        resource = read(base/'qkv_slots7_b4_units_v1/result.json')
        if (compiled['changed_stages'] != list(range(6, 60, 5))
                or not compiled['control_proof']['passed']
                or resource['rows'][2]['payload_bytes'] != 214016
                or resource['rows'][3]['fits']):
            raise ValueError('B4 seed-rule spanning candidate differs from its evidence')
        lines += ['', 'Verified CPU-only B4 pipeline follow-up changes only11 repeated '
                  '128x96 QKV stages from3 to7 slots. Prompt4.5 selects spanning because '
                  'the one-arena maximum is3;seven slots fit214016B and eight do not fit. '
                  'The previous FFN2 already spans and attention cannot preissue,so this '
                  'change loses no existing adjacent B-page preissue. The72/208 profile '
                  f'is unchanged;zero spill,SASS{compiled["model"]["sass_bytes"]}B,control '
                  'regeneration matches all35296 encoded words. Complete component/model '
                  'checks and an independent same-GPU3 pair are required for acceptance.']
    for prefix, label in [('store96_matrix', 'partial TN96 matrix-store B8'),
                          ('qkv_slots4', 'four-slot QKV B2'),
                          ('qkv_slots7', 'seven-slot QKV B4'),
                          ('qkv_attention', 'QKV plus attention vector stores B2'),
                          ('mma_zero', 'first WGMMA initialization B16'),
                          ('fixed_grid', 'fixed H800 grid B16'),
                          ('grid_recount', 'fixed grid producer recomputation B8'),
                          ('recount_qkv4','producer plus four-slot QKV B2'),
                          *[(f'grid_recount_b{b}',f'producer recomputation B{b}') for b in (2,4,16)]]:
        result = continuation/(prefix+'_experiment.json')
        if result.exists() and not read(result)['accepted']:
            experiment = read(result); pair = experiment['pair']
            lines += ['', f'Verified {label} completes component/model gates and GPU3 '
                      f'P-PAIR6/6:reference{pair["reference_ms"]:.6f}ms,candidate'
                      f'{pair["candidate_ms"]:.6f}ms,gain{100*(1-pair["ratio"]):.3f}%,noise'
                      f'{100*pair["threshold"]:.3f}%,minimum gain'
                      f'{100*experiment.get("minimum_gain", 0):.1f}%. Reject;retain the '
                      'incumbent and all measured processes. No final collection or '
                      'production admission follows this pair.']
    for prefix in ('qkv_slots4', 'qkv_slots7'):
        path = continuation/(prefix+'_attribution.json')
        if not path.exists():
            continue
        attribution = read(path); metrics = attribution['metrics']
        if attribution['group'] != '11 repeated QKV':
            raise ValueError('ring attribution group differs from its coordinate scope')
        values = ';'.join(f'{label}{metrics[key]["reference_us"]:.3f}'
                          f'→{metrics[key]["candidate_us"]:.3f}us'
                          for label, key in [('windows', 'stage_ns'),
                                             ('main', 'main_loop_max_cta_ns'),
                                             ('slot wait', 'slot_wait_max_cta_ns'),
                                             ('first ready', 'first_ready_ns')])
        lines += ['', f'Verified B{attribution["batch"]} ring attribution across11 QKV '
                  f'stages:{values}. Historical accepted GPU3 trace and fresh exact-candidate '
                  'trace are diagnostic;segment maxima overlap. They do not override '
                  'the independent nontrace pair or qualify a new performance result.']
    if (continuation/'store96_b16_conditional_result.json').exists():
        lines += ['Verified conditional B16 partial-store work remains CPU-only because '
                  'the B8 mechanism did not pass its end-to-end gate;no B16 GPU job is queued.']
    zero = base/'mma_zero_b16_preparation_v2/result.json'
    if zero.exists() and read(zero)['status'] == 'prepared':
        prepared = read(zero)
        if not prepared['control_proof']['passed'] or prepared['control_proof']['words'] != 44736:
            raise ValueError('first-MMA prototype lacks its default0 instruction proof')
        lines += ['', 'Verified private default-off first-WGMMA initialization uses the '
                  'CUTLASS explicit first-slot prologue:each m64 fragment starts with '
                  'ScaleOutZero,then retains the original k16 order,slot and release '
                  'protocol. The dynamic-predicate prototype leaves clearing instructions '
                  'and never executes on GPU. The explicit prologue reduces static CS2R '
                  'zero instructions289→56;SASS357888→362880B,zero spill,maxR207. Default0 '
                  'matches all44736 incumbent instruction words. Coordinates and the '
                  'accepted B16 profile stay fixed;end-to-end benefit is unproven.']
        bit_proof = continuation/'mma_zero_unit_bit_proof.json'
        if bit_proof.exists():
            proof = read(bit_proof)
            if not proof['passed'] or proof['reference_passes'] != '5/5' or proof['candidate_passes'] != '17/17':
                raise ValueError('first-MMA component byte comparison is incomplete')
            lines += ['Verified five fresh control processes and the candidate unit5/5 '
                      'plus mem1/1,init1/1,race5/5,sync5/5 have identical output hashes '
                      'on GPU0. The five shapes cover GELU,residual statistics,TM256 '
                      'two-fragment ownership,TK128/split2,odd rows and ring wrap. '
                      'These component checks do not establish an end-to-end gain.']
        model_proof = continuation/'mma_zero_model_bit_proof.json'
        if model_proof.exists():
            proof = read(model_proof)
            if not proof['passed']:
                raise ValueError('first-MMA whole-model byte comparison failed')
            lines += ['Verified B16 strict model5/5 also matches the incumbent output '
                      'bytes. The independent GPU3 pair rejects this private candidate; '
                      'the production headers and accepted B16 result remain unchanged.']
    fixed_grid = base/'fixed_grid_b16_preparation_v1/result.json'
    if fixed_grid.exists() and read(fixed_grid)['status'] == 'prepared':
        prepared = read(fixed_grid)
        if not prepared['control_proof']['passed']:
            raise ValueError('fixed H800 grid lacks its default0 instruction proof')
        lines += ['', 'Verified private fixed132 H800 grid specialization retains '
                  'the B16 coordinates,register profile,task ownership and arithmetic '
                  'order. It folds runtime task division and prefetch partitioning; '
                  'SASS357888→343168B,static RCP97→76,zero spill. Default0 matches '
                  'all44736 incumbent instruction words. This extends the prompt\'s '
                  'index specialization under the user-authorized implementation '
                  'discretion;separate functional and E2E gates decide admission.']
        host = fixed_grid.parent/'host_guard/result.json'
        if host.exists() and read(host)['passed']:
            lines += ['Verified five fresh CPU-only host processes reject grids '
                      '0/1/131/133/264 in both serving modes before changing the '
                      'epoch. GPUs are hidden and no valid-grid launch is attempted.']
        for suite in ('body', 'attention'):
            receipt = continuation/('fixed_grid_'+suite+'_unit')/'result.json'
            if receipt.exists():
                unit = read(receipt)
                if not unit['passed'] or len(unit['processes']) != 17:
                    raise ValueError('fixed grid unit gate incomplete')
                lines += [f'Verified fixed-grid {suite} on GPU{unit["gpu"]}: '
                          'unit5/5,mem1/1,init1/1,race5/5,sync5/5;output bytes '
                          'match the frozen control. These are functional gates.']
        b8 = base/'fixed_grid_b8_preparation_v1/result.json'
        if b8.exists() and read(b8)['status'] == 'rejected_compile':
            lines += ['Verified CPU-only B8 portability check spills20/20B under '
                      'its current72/208 profile. It is rejected before GPU '
                      'execution;no B8 hardware result is claimed.']
        b8_profile3 = base/'fixed_grid_b8_profile3_preparation_v1/result.json'
        if b8_profile3.exists() and read(b8_profile3)['status'] == 'rejected_compile':
            lines += ['Verified existing72/216 also spills20/20B for fixed-grid B8; '
                      'that CPU-only follow-up is rejected without a register sweep.']
        admitted = base/'fixed_grid_b16_admitted_v1/admission.json'
        if admitted.exists():
            receipt = read(admitted)
            if receipt['status'] != 'admitted':
                raise ValueError('fixed grid admission is incomplete')
            lines += ['Verified fixed-grid B16 is admitted after main on/off '
                      'instruction equivalence,repository fixture equivalence, '
                      'host rejection5/5,strict model5/5,independent GPU3 pair and '
                      'an exact63-stage ledger. P-DNN median'
                      f'{receipt["median_ms"]:.6f}ms,p90{receipt["p90_ms"]:.6f}ms,'
                      f'process range{receipt["process_range_ms"]:.6f}ms. '
                      'Other four cells and original anchors remain unchanged. '
                      'CMake regeneration is blocked by a missing installed '
                      'MLIRCAPIDebug object;the unchanged certified cached archive '
                      'and rebuilt codegen object pass real codegen14profiles/'
                      '8invalid-settings checks,41admission tests and two related '
                      'CTest checks. The full CMake build is not claimed.']
    recount = base/'grid_recount_b8_preparation_v1/result.json'
    if recount.exists() and read(recount)['status'] == 'prepared':
        prepared = read(recount)
        if not prepared['control_proof']['passed']:
            raise ValueError('producer recomputation lacks its default0 proof')
        lines += ['', 'Verified private B8 producer repair recomputes the CTA index '
                  'at task/address uses. Earlier fixed-grid spills all occur after '
                  'producer deallocation to72registers;the larger consumer budget '
                  'does not resolve them. Current72/208 coordinates/profile now '
                  'compile with zero spill,SASS328576→311680B,RCP77→56. Default0 '
                  'matches all41072 current B8 words and all42896 admitted B16 '
                  'words. This is a compiler register-lifetime repair;independent '
                  'whole-model performance decides acceptance.']
        for suite in ('body', 'attention'):
            receipt = continuation/('grid_recount_'+suite+'_unit')/'result.json'
            if receipt.exists():
                unit = read(receipt)
                if not unit['passed'] or len(unit['processes']) != 17:
                    raise ValueError('producer recomputation unit gate incomplete')
                lines += [f'Verified producer-recomputation {suite} on GPU{unit["gpu"]}: '
                          'unit5/5,mem1/1,init1/1,race5/5,sync5/5;output bytes '
                          'match the frozen shape controls. These are functional gates.']
    fixed_b2 = base/'fixed_grid_b2_preparation_v1/result.json'
    if fixed_b2.exists() and read(fixed_b2)['status'] == 'rejected_compile':
        lines += ['', 'Verified fixed-grid-only B2 under its current72/216 '
                  'profile spills144/188B. It is rejected at the CPU gate and '
                  'does not execute on GPU.']
    guards = base/'grid_recount_default_guard_v1'
    guarded_names = ('b1_off', 'b2_off', 'b4_off', 'b8_off', 'b16_off', 'b8_on')
    if all((guards/name/'result.json').exists() for name in guarded_names):
        if not all(read(guards/name/'result.json')['passed'] for name in guarded_names):
            raise ValueError('producer default guard changes accepted instructions')
        lines += ['', 'Verified the original disabled BlockIndex helper changes '
                  'B2 instructions and is rejected before candidate compilation or '
                  'GPU execution. The repaired compile-time macro restores the '
                  'literal original expression when disabled. All five current '
                  'BERT kernels match their original instruction words;enabled B8 '
                  'matches all38960 words of the independently qualified prototype. '
                  'No numerical or timing process is repeated for this source repair.']
    recount_admission = base/'grid_recount_b8_admitted_v1/admission.json'
    if recount_admission.exists():
        admission = read(recount_admission)
        if admission['status'] != 'admitted':
            raise ValueError('producer recomputation admission is incomplete')
        lines += ['', 'Verified B8 producer recomputation is integrated and admitted: '
                  'main on/off kernels match all38960/41072 tested instruction words;'
                  'committed body/attention fixtures match the two17-process gates;'
                  'the unchanged fixed-grid runtime retains its five-process host '
                  'rejection proof. Real codegen16profiles/9invalid settings and '
                  '46admission tests pass. GPU3 independent pair gains'
                  f'{100*(1-admission["pair_ratio"]):.3f}% versus'
                  f'{100*admission["noise"]:.3f}% noise;P-DNN'
                  f'{admission["median_ms"]:.6f}ms,p90{admission["p90_ms"]:.6f}ms,'
                  f'process range{admission["process_range_ms"]:.6f}ms. '
                  'The exact tested binary and63-stage ledger are retained. '
                  'Other cells and original anchors remain unchanged.']
    recount_b2 = base/'grid_recount_b2_preparation_v2/result.json'
    if recount_b2.exists() and read(recount_b2)['status'] == 'prepared':
        lines += ['', 'Verified CPU-only B2 uses the same producer repair with '
                  'current72/216 coordinates/profile:zero spill,SASS264960→250752B.'
                  'The repaired default path matches all33120 incumbent words. '
                  'Separate functional and independent GPU3 timing gates decide '
                  'acceptance;code size alone does not admit this candidate.']
    recount_b4 = base/'grid_recount_b4_admitted_v1/admission.json'
    if recount_b4.exists():
        admission = read(recount_b4)
        if admission['status'] != 'admitted':
            raise ValueError('B4 producer admission is incomplete')
        lines += ['', 'Verified B4 reuses the exact profile2 body/attention17-process '
                  'gates and completes fresh model5/5 with unchanged output bytes. '
                  'Main on/off33072/35296 instruction words match the paired '
                  'candidate/incumbent;the exact GPU3 trace has63 stages. '
                  f'Independent pair gains{100*(1-admission["pair_ratio"]):.3f}% '
                  f'versus{100*admission["noise"]:.3f}% noise. Admitted P-DNN='
                  f'{admission["median_ms"]:.6f}ms,p90{admission["p90_ms"]:.6f}ms,'
                  f'process range{admission["process_range_ms"]:.6f}ms. '
                  'Other cells and original anchors remain unchanged.']
    recount_candidates = base/'recount_b8_existing_candidates_v1/result.json'
    if recount_candidates.exists():
        candidate_builds = read(recount_candidates)
        if candidate_builds['legal'] != candidate_builds['total'] or candidate_builds['total'] != 6:
            raise ValueError('recomputed B8 candidate legality differs')
        lines += ['', 'Verified CPU-only B8 check:all six exact existing repaired '
                  'P-TUNE coordinate programs compile without spills under the '
                  'currently admitted72/208 producer profile. No new coordinate '
                  'or register sweep is introduced. Legal compilation alone '
                  'does not select a program;boundary qualification,strict model '
                  'checks and an independent current-incumbent pair remain required.']
    trace_cta = base/'trace_cta_b8_uniform_v2/result.json'
    if trace_cta.exists():
        builds = read(trace_cta)
        legal = sorted(row['name'] for row in builds['rows']
                       if row['name'].startswith('candidate') and row['status'] == 'prepared')
        assert legal == ['candidate0', 'candidate1', 'candidate3', 'candidate4', 'candidate5']
        for name, words in [('nontrace_proof', 38960), ('attribution_proof', 44288)]:
            proof = read(trace_cta.parent/name/'result.json')['proof']
            assert proof['passed'] and proof['words'] == words
        unit = read(continuation/'trace_cta_unit/result.json')
        assert unit['passed'] and len(unit['processes']) == 17
        lines += ['', 'Verified private boundary-only CTA recomputation passes GPU0 '
                  'unit5/5 and sanitizer1/1/5/5. All38960 production and44288 '
                  'full-trace instruction words are preserved. The uniform boundary '
                  'source makes five of six existing candidates legal;candidate2 '
                  'still spills8/8B. Reusing one CTA value also leaves8/8B and is '
                  'rejected before GPU execution. No complete six-way P-TUNE result '
                  'is claimed;missing windows are not synthesized.']
        qualification = continuation/'trace_cta_b8_qualification_v2/result.json'
        if qualification.exists():
            q = read(qualification)
            lines += ['', f'Verified private B8 boundary qualification:GPU3 '
                      f'overhead{100*q["relative_difference"]:.3f}%,passed={q["passed"]}. '
                      'Three fresh20/50 trace processes compare exact output bytes '
                      'with the nontrace reference;one fresh50/200 reference supplies '
                      'the timing comparison. This is diagnostic qualification, '
                      'not performance admission.']
    proposal = base/'recount_b8_five_tune_v2/tune_selection.json'
    if proposal.exists():
        selected = read(proposal)
        assert selected['coverage'] == 5 and selected['missing'] == 2
        choices = [group['candidate'] for group in selected['groups']]
        lines += ['', f'Verified B8 diagnostic windows cover five of six existing '
                  f'programs, three fresh processes each; six groups choose {choices}. '
                  'The missing candidate2 remains in the nontrace selection. '
                  'An untied selection and an independent current-incumbent pair '
                  'are required; this is not complete six-way P-TUNE coverage.']
    catalog = base/'fixed_grid_b16_existing_candidates_v1/result.json'
    if catalog.exists():
        builds = read(catalog)
        legal = [row['candidate'] for row in builds['rows'] if row['status'] == 'prepared']
        assert legal == [1, 2, 3]
        lines += ['', 'Verified current-profile B16 existing candidates1/2/3 compile '
                  'without spills;0/4/5 are rejected before hardware execution. '
                  'The whole-program fallback compares those three with the incumbent '
                  'and requires an untied selection followed by an independent pair. '
                  'No boundary assembly or P-TUNE completion is claimed.']
    for batch, name in ((8, 'recount_b8_five_selection_v2'),
                        (16, 'fixed_grid_b16_whole_selection')):
        selected_path = continuation/name/'selection/result.json'
        if selected_path.exists():
            selected = read(selected_path)
            assert (selected['gpu'] == 3 and len(selected['measurements']) == 8
                    and [row['arm'] for row in selected['measurements']] == list('ABCDDCBA')
                    and all(row['guard']['code'] == 0 for row in selected['measurements']))
            ranking = ', '.join(f'{row["arm"]}={row["median_ms"]:.6f}ms'
                                for row in selected['ranking'])
            lines += ['', f'Verified current-profile B{batch} nontrace selection: '
                      f'{ranking}; tied={selected["tied"]}, '
                      f'noise={100*selected["threshold"]:.3f}%. '
                      'Selection medians are diagnostic; the accepted catalog changes '
                      'only after an independent pair, final measurement and ledger audit.']
    optional = base/'repeat_roles_preparation_v1/result.json'
    if optional.exists():
        regression = read(base/'repeat_roles_cpu_regression_v1/result.json')
        assert regression['passed'] and len(regression['rows']) == 4
        lines += ['', 'Verified private Repeat codegen reserves operand roles used '
                  'only by first or last repetitions. Existing optional-map candidates '
                  'B8/B16 now use319488/359424B of SASS without spills. All five current '
                  'generated sources remain identical; four real codegen cases reject '
                  'optional-map and alias corruption. These are CPU results, not '
                  'performance admission.']
        rejection = continuation/'repeat_roles_v1_rejection.json'
        if rejection.exists():
            lines += ['', 'Verified the first optional-map fixture fails initcheck '
                      'on generic reads after TMA output, consistent with the documented '
                      '2025.2.1 TMA initialization-shadow limitation. The rejected cohort '
                      'is preserved. The replacement poisons every output before every '
                      'launch and checks all intermediate values against positive CPU '
                      'references; its CUDA instruction words are unchanged. This '
                      'workaround retains the tool coverage limitation.']
        for batch in (8, 16):
            unit_path = continuation/f'repeat_roles_units_v2/b{batch}/result.json'
            if unit_path.exists():
                unit = read(unit_path)
                assert unit['passed'] and len(unit['processes']) == 17
                lines += ['', f'Verified optional Repeat B{batch} ordering fixture '
                          f'passes unit5/5 and sanitizer1/1/5/5 on GPU{unit["gpu"]}. '
                          'Whole-model byte equivalence and both code-size and '
                          'current-incumbent performance gates remain separate.']
            pair_path = continuation/f'repeat_roles_followup_v2/b{batch}/code_experiment.json'
            if pair_path.exists():
                experiment = read(pair_path)
                pair = experiment['pair']
                proof = read(pair_path.parent/'model_bit_proof.json')
                assert proof['passed'] and pair['gpu'] == 3
                assert [row['arm'] for row in pair['measurements']] == list('ABABAB')
                assert all(row['guard']['code'] == 0 for row in pair['measurements'])
                lines += ['', f'Verified private optional Repeat B{batch} same-coordinate '
                          f'P-PAIR: {pair["reference_ms"]:.6f} to '
                          f'{pair["candidate_ms"]:.6f}ms, gain '
                          f'{100*(1-pair["ratio"]):.3f}%, noise '
                          f'{100*pair["threshold"]:.3f}%, DT-CODE minimum3%. '
                          f'Passes this prerequisite={experiment["accepted"]}. '
                          'The private optional-role validator supplies identity checks; '
                          'the unchanged main contract does not support these artifacts. '
                          'This remains outside the production catalog.']
    epical = continuation/'bert_epical_current_v1/result.json'
    if epical.exists():
        recheck = read(epical)
        assert len(recheck['rows']) == 3
        values = ', '.join(f'{row["case"]["chain"]}='
                           f'{row["median_ns"]["A"]:.1f}→{row["median_ns"]["B"]:.1f}ns'
                           for row in recheck['rows'])
        lines += ['', f'Verified matched GPU3 epilogue recheck: {values}. '
                  'Original56/224 and current72/216 P1/matrix implementations each '
                  'use five fresh processes,132 CTAs and three tasks. These are '
                  'coefficient measurements, not new end-to-end gains; constants '
                  'remain uninstalled.']
    model_audit = base/'coordinate_prediction_current_v1/result.json'
    if model_audit.exists():
        audit = read(model_audit)
        proof = read(model_audit.parent/'original_rank0_proof.json')
        assert proof['passed'] and proof['stages'] == 63
        ratios = ', '.join(f'B{row["batch"]}={row["actual_over_predicted"]:.3f}'
                           for row in audit['rows'])
        lines += ['', f'Verified current-coordinate model audit, actual/predicted: '
                  f'{ratios}. The CPU utility reproduces all63 original rank0 '
                  'stage predictions exactly and reuses current accepted P-DNN '
                  'observations; it does not claim new P-MODEL traces. Every cell '
                  'exceeds1.3. Lower epilogue coefficients alone would worsen '
                  'underprediction, so no epilogue-only table is installed. '
                  'Inferred next implementation check: eliminate runtime tail '
                  'conditions for dimensions known to contain only full tiles.']
    boundary_audit = base/'current_b8_boundary_model_audit_v1/result.json'
    if boundary_audit.exists():
        audit = read(boundary_audit)
        groups = ', '.join(f'{row["group"]} '
                           f'{row["predicted_us"]:.1f}→{row["measured_us"]:.1f}µs'
                           for row in audit['groups'])
        lines += ['', f'Verified reuse of pre-full-tile B8 qualified boundary windows '
                  f'(prediction→observation, group sums): {groups}. '
                  'The existing three20/50 processes passed4.0288% overhead '
                  'qualification on GPU3. FFN1 has the largest underprediction. '
                  'This is an incumbent-coordinate diagnostic, not a new rank0 '
                  'P-MODEL gate; boundary terms alone do not assign the gap to '
                  'mainloop or epilogue. No coefficient is changed.']
    for batch, preparation, folder in [
            (8, 'full_tiles_preparation_v2/b8', 'full_tiles_b8_v2'),
            (16, 'full_tiles_b16_selector_v1', 'full_tiles_b16_selector_v1')]:
        prepared = base/preparation
        if not (prepared/'result.json').exists():
            continue
        if read(prepared/'result.json')['status'] != 'prepared':
            continue
        before = read(prepared/'flag0/result.json')
        after = read(prepared/'flag1/result.json')
        dest = continuation/folder
        completed = []
        for suite in ['body', 'order', 'repeat_even', 'repeat_odd']:
            gate = dest/suite/('individual' if suite == 'order' else 'flag1')/'result.json'
            if gate.exists():
                result = read(gate)
                assert result['passed'] and len(result['processes']) == 17
                assert all(read(Path(row['guard']))['code'] == 0 for row in result['processes'])
                completed.append(suite)
        lines += ['', f'Verified private full-tile B{batch} CPU preparation: '
                  f'SASS {before["sass_bytes"]}→{after["sass_bytes"]}B, zero spills; '
                  'disabled kernels preserve current instruction words. '
                  + ('B16 also rematerializes the first/last Repeat selector to '
                     'remove a measured4B consumer predicate spill. ' if batch == 16 else '')
                  + f'Complete unit5/sanitizer1/1/5/5 suites: {", ".join(completed) or "none"}. '
                  'The initial B8 ordering cohort exceeded its aggregate1200s limit '
                  'and is rejected; its replacement retains all200-launch cases '
                  'with separately guarded processes. These are implementation '
                  'and correctness results; no performance admission is implied.']
        if (dest/'model_bit_proof.json').exists():
            proof = read(dest/'model_bit_proof.json')
            model = read(dest/'model/result.json')
            assert proof['passed'] and model['passed'] and model['passes'] == '5/5'
            lines += ['', f'Verified full-tile B{batch} strict model5/5 on '
                      f'GPU{model["gpu"]}, with exact current-incumbent output bytes.']
        if (dest/'experiment.json').exists():
            experiment = read(dest/'experiment.json')
            pair = experiment['pair']
            assert all(row['guard']['code'] == 0 for row in pair['measurements'])
            lines += ['', f'Verified full-tile B{batch} GPU3 pair ratio '
                      f'{pair["ratio"]:.6f}, noise {100*pair["threshold"]:.3f}%; '
                      f'passes the independent gain gate={experiment["accepted"]}. '
                      'Main integration, final measurement and an exact-binary '
                      'ledger remain separate admission requirements.']
        admission_path = base/f'full_tiles_b{batch}_admitted_v1/admission.json'
        if admission_path.exists():
            admitted = read(admission_path)
            assert admitted['status'] == 'admitted'
            lines += ['', f'Verified full-tile B{batch} admission: GPU3 P-DNN '
                      f'{admitted["median_ms"]:.6f}ms, p90 '
                      f'{admitted["p90_ms"]:.6f}ms, process range '
                      f'{admitted["process_range_ms"]:.6f}ms. Main emitted kernels '
                      'and focused fixtures match the tested instruction words; '
                      'disabled defaults are preserved. The exact measured binary, '
                      'strict model gate and63-stage ledger are retained. Eight '
                      'corrupted admission receipts are rejected in CPU checks. '
                      'Other cells, fixed goals and original anchors are unchanged.']
    small_proof = base/'full_tiles_main_small_codegen_v1/result.json'
    if small_proof.exists():
        emitted = read(small_proof)
        assert emitted['passed'] and all(r['passed'] for r in emitted['rows'])
        for batch in (2, 4):
            model_path = continuation/f'full_tiles_small_models_v1/b{batch}/result.json'
            model = read(model_path)
            bit = read(model_path.parent/'model_bit_proof.json')
            assert model['passed'] and model['passes'] == '5/5' and bit['passed']
        lines += ['', 'Verified fixed full-tile B2/B4 candidates each pass strict '
                  'whole-model5/5 on GPU0 and preserve incumbent output bytes. '
                  'Real main codegen matches31360/31424 encoded64-bit instruction '
                  'words of the prepared binaries, with zero spill and unchanged '
                  'coordinates/profiles. B2 has a separate72/216 body cohort '
                  '(unit5,mem1,init1,race5,sync5); its model uses only register '
                  'stores. B4 requires the matching72/208 B8 body and ordering '
                  'cohorts. Independent GPU3 performance and exact-binary ledger '
                  'checks remain separate admission requirements.']
        admitted_path = base/'full_tiles_b4_admitted_v1/admission.json'
        if admitted_path.exists():
            admitted = read(admitted_path)
            assert admitted['status'] == 'admitted'
            lines += ['', f'Verified full-tile B4 admission: GPU3 P-DNN '
                      f'{admitted["median_ms"]:.6f}ms, p90 '
                      f'{admitted["p90_ms"]:.6f}ms, process range '
                      f'{admitted["process_range_ms"]:.6f}ms. Its independent '
                      f'pair gains {100*(1-admitted["pair_ratio"]):.3f}% against '
                      f'{100*admitted["noise"]:.3f}% noise. The main emitted '
                      'kernel matches the measured binary; complete B8 body '
                      'and ordering cohorts are reused only after matching '
                      'their full72/208 feature profiles. Nine corrupted '
                      'receipts, including mismatched inherited profiles, '
                      'are rejected. The exact63-stage GPU3 ledger, fixed '
                      'target and original anchor are retained.']
    isolated = continuation/'bert_stage_probe_constants_gpu3_v2/result.json'
    if isolated.exists():
        measured = read(isolated)['rows']
        assert len(measured) == 4 and all(r['passes'] == '5/5' for r in measured)
        values = ', '.join(f'B{r["batch"]} FFN{r["ffn"]} '
                           f'{r["window_ns"]/1000:.3f}us' for r in measured)
        lines += ['', 'Verified GPU3 isolated FFN diagnostics: '+values+'. '
                  'Each fixed production shape/profile uses132 CTAs, five fresh '
                  '20/50 processes, real TMA and unchanged epilogue arithmetic; '
                  'integer model shape/stride constants are preserved and every output '
                  'is checked against an independent constant-input FP32 reference. '
                  'The qualified pre-full-tile B8 FFN1 boundary mean is24.728us versus '
                  '12.096us isolated. The workloads differ in cache state, code '
                  'context, bindings and stage gates; this gap is not assigned '
                  'to one cause. These historical probes used synthetic '
                  'epsilon1e-5; emitted BERT uses1e-12. No F4 constant or '
                  'end-to-end result is replaced.']
    for folder, contrast in [('bert_data_context_gpu0_v1', '128MiB data-write'),
                             ('bert_code_context_v1', '1MiB straight-line-code')]:
        context = continuation/folder/'result.json'
        if not context.exists():
            continue
        values = read(context)['rows']
        assert len(values) == 2 and all(r['passes'] == '5/5' for r in values)
        hot, cold = values
        assert hot['gpu'] == cold['gpu']
        assert all(read(Path(r['guard']))['code'] == 0 for r in values)
        lines += ['', f'Verified GPU{hot["gpu"]} {contrast} contrast for the same '
                  f'isolated B8 FFN1 kernel: hot {hot["window_ns"]/1000:.3f}us '
                  f'and cold {cold["window_ns"]/1000:.3f}us, ratio '
                  f'{cold["window_ns"]/hot["window_ns"]:.6f}. Each context uses '
                  'five fresh20/50 processes and checks every output against '
                  'the constant-input FP32 reference. All6304 encoded64-bit '
                  'Probe words match the original diagnostic; eviction runs '
                  'outside its measured consumer window. This same-card '
                  'contrast does not establish the complete model gap or '
                  'replace F4 coefficients or end-to-end admission. Both '
                  'sides use the historical synthetic epsilon1e-5.']
    correction = base/'bert_probe_epsilon_v1/literal_proof.json'
    if correction.exists():
        proof = read(correction)
        assert proof['passed'] and proof['changed_words'] == 7
        lines += ['', 'Verified probe audit corrects its default epsilon from '
                  'synthetic1e-5 to the emitted BERT1e-12. The old setting '
                  'reproduces all6304 original instruction words; the corrected '
                  'build changes only seven FP32 FADD immediates, retaining '
                  'the opcode/register/scheduling words. Prior within-binary '
                  'contrasts remain synthetic diagnostics, not exact production '
                  'constants. Accepted model builds and installed coefficients '
                  'were never changed by these probes.']
    for folder, epsilon in [('bert_pattern_gpu0_v2', '1e-5'),
                            ('bert_pattern_epsilon_gpu0_v1', '1e-12')]:
        pattern = continuation/folder/'result.json'
        if not pattern.exists():
            continue
        uniform, varying = read(pattern)['rows']
        assert uniform['passes'] == varying['passes'] == '5/5'
        measurements = [p['measurement'] for r in (uniform, varying)
                        for p in r['processes']]
        assert all(m['reference_cosine'] >= .99999 and
                   m['reference_relative'] <= .01 for m in measurements)
        lines += ['', f'Verified GPU0 fixed FFN1 input-pattern contrast at '
                  f'epsilon{epsilon}: uniform {uniform["window_ns"]/1000:.3f}us '
                  f'and varying {varying["window_ns"]/1000:.3f}us. The same '
                  'kernel uses32 deterministic BF16 row/column patterns; '
                  'every output passes the independent FP32/GELU reference '
                  'with cosine≥0.99999 and maximum relative error≤0.01. '
                  'Each pattern has identical output bytes across five fresh '
                  '20/50 processes. This synthetic-data diagnostic does not '
                  'replace a whole-model measurement or coefficient.']
    lock_rejection = continuation/'full_tiles_default_lock_rejection.json'
    if lock_rejection.exists():
        lines += ['', 'Verified full-tile ordering dispatch correction: the first '
                  'individual-process wrapper omitted the device environment on '
                  'its scheduler subprocess, which replaced per-job lock paths '
                  'with one shared default lock. All of those individual ordering '
                  'results are excluded. The replacement passes the per-device '
                  'environment and checks the actual lock before each CUDA child; '
                  'all17 processes are recollected. Grouped body, Repeat and model '
                  'gates already held the correct device locks. The earlier foreign '
                  'GPU-lock and NUMA-setting diagnoses were not established.']
    timeout = continuation/'full_tiles_b16_selector_v1/order/device_locked_v1/timeout_hold.json'
    if timeout.exists():
        ordering_complete = (continuation/
            'full_tiles_b16_selector_v1/order/individual/result.json').exists()
        lines += ['', 'Verified full-tile B16 individual racecheck durations vary: '
                  'two complete guarded processes took603s and825s with zero errors; '
                  'a third reached the900s child timeout and has no valid result. '
                  'Its subsequent ownership rejection does not establish foreign '
                  'interference. The timed-out attempt is excluded; nine completed '
                  'processes were retained and only the eight missing checks '
                  'were scheduled with line-buffered output and a3600s child limit. The frozen '
                  'binary, cases,200 iterations, guard and correctness thresholds '
                  'are unchanged. ' +
                  ('The replacement ordering gate now completes all17 processes; '
                   'performance admission remains a separate requirement.'
                   if ordering_complete else
                   'No complete ordering gate or performance gain is claimed '
                   'while this remains unfinished.')]
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
                  'The earlier B2 comparison fails its paired noise gate;the later full-pipeline '
                  'combination has a separate admission below. '
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
    store_admission = base/'full_store_bundle_b8_admitted_v1/admission.json'
    if store_admission.exists():
        admission = read(store_admission)
        lines += ['', 'Verified current B8 combines existing matrix and attention vector stores '
                  'with the admitted full-tile profile. Three exact-profile component cohorts '
                  'each pass17/17;strict model5/5 preserves incumbent bytes. Main real codegen '
                  'matches all35232 encoded instruction words and the63-stage ledger identifies '
                  f'the measured binary. GPU3 P-DNN{admission["median_ms"]:.6f}ms,'
                  f'p90{admission["p90_ms"]:.6f}ms. Ten corrupted admission receipts are rejected. '
                  'Other cells,default flags and fixed anchors remain unchanged.']
    reduction_admission = base/'pipeline_reduction_b2_admitted_v1/admission.json'
    if reduction_admission.exists():
        admission = read(reduction_admission)
        lines += ['', 'Verified current B2 combines full-tile/grid/recompute with existing '
                  'coalesced partials and cyclic slots at unchanged coordinates and72/216 registers. '
                  'Eleven exact/odd-tail fixtures pass control5 and candidate17 with identical bytes;'
                  'strict model5 preserves incumbent bytes. Main real codegen matches29984 encoded '
                  'words;the exact measured binary has a63-stage GPU3 ledger. '
                  f'P-DNN{admission["median_ms"]:.6f}ms,p90{admission["p90_ms"]:.6f}ms. '
                  f'Pair gain{100*(1-admission["pair_ratio"]):.5f}% narrowly exceeds '
                  f'noise{100*admission["noise"]:.5f}%;this variability remains in the receipt. '
                  'Twelve corrupted admission receipts are rejected;fixed targets are unchanged.']
    direct = continuation/'direct_norm_b16_v3/experiment.json'
    if direct.exists():
        trial = read(direct)
        lines += ['', 'Verified private P1 direct-statistics reads preserve ascending sums, '
                  'normalization arithmetic,shared allocation and synchronization. Control5 and '
                  'candidate17 cover ten exact/odd-tail fixtures;model5 preserves incumbent bytes. '
                  f'GPU3 pair ratio{trial["pair"]["ratio"]:.6f},'
                  f'noise{100*trial["pair"]["threshold"]:.3f}%. '
                  'The candidate has no end-to-end gain and is rejected;private headers restored. '
                  'The initial illegal synthetic shape failed compilation in both arms and was '
                  'replaced before GPU execution with the actual legal QKV shapes.']
    repaired_attribution = continuation/'candidate5_recompute_b16_v2/attribution.json'
    if repaired_attribution.exists():
        audit = read(repaired_attribution.with_name('attribution_audit.json'))
        assert audit['passed']
        attribution = read(repaired_attribution)
        groups = {r['group']: r for r in attribution['rows']}
        changes = ', '.join(f'{name}+'
                            f'{groups[name]["stage_delta_ns"]/1000:.3f}us'
                            for name in ('projection', 'ffn2', 'qkv'))
        lines += ['', 'Verified formerly spilled B16 catalog candidate5 compiles without spill '
                  'after the existing task-index repair;14 exact/odd-tail fixtures pass control5 '
                  'and candidate17, and strict model5 preserves incumbent bytes. '
                  'Its independent GPU3 pair regresses15.282% and is rejected. '
                  f'Exact-binary P-TRACE1 group-window increases are {changes}. '
                  'Projection/FFN2 epilogue CTA-maxima sums rise73.280→146.304us '
                  'and76.608→146.304us. These diagnostic maxima are nonadditive; '
                  'no P-DNN or current-artifact replacement follows the rejection.']
    aggregation_repair = continuation/'stats_ahead_b16_v1/model_aggregation_repair.json'
    if aggregation_repair.exists():
        repair = read(aggregation_repair)
        assert repair['passed']
        metric = next(r for r in repair['metrics']
                      if r['output'] == 'tanh' and r['metric'] == 'max_error')
        lines += ['', 'Verified B16 statistics-read-ahead model checks produce identical bytes '
                  'in all five processes and all six execution cases, with unchanged FP32 gates. '
                  f'One reported tanh maximum error varies from {metric["min"]:.12f} to '
                  f'{metric["max"]:.12f}. The original wrapper incorrectly required the entire '
                  'metric dictionary to match. A CPU-only finalizer validates each process against '
                  'the original threshold and requires identical output hashes; raw checks and the '
                  'initial assertion are retained. Thirteen corrupted cohorts are rejected. '
                  'No hardware result or accuracy threshold was changed.']
    loop_attribution = continuation/'loop_pair_b4_v1/attribution.json'
    if loop_attribution.exists():
        groups = read(loop_attribution)['rows']
        gemm = [r for r in groups if r['group'] not in ('attention', 'small')]
        before = sum(r['control']['main_loop_max_cta_ns'] for r in gemm)/1000
        after = sum(r['candidate']['main_loop_max_cta_ns'] for r in gemm)/1000
        lines += ['', 'Verified B4 fixed two-iteration main-loop unrolling passes control5, '
                  'candidate17 on12 exact/odd-K/odd-M fixtures and strict model5 with unchanged bytes. '
                  'The GPU3 pair regresses5.880% and is rejected. '
                  f'P-TRACE1 GEMM main-loop CTA-maxima sums increase {before:.3f}→{after:.3f}us; '
                  'the intended segment did not improve. These maxima are nonadditive diagnostic '
                  'attribution. The private header is restored.']
    optional_prep = base/'optional_stats_b16_v2/result.json'
    if optional_prep.exists() and read(optional_prep).get('status') == 'prepared':
        preparation = read(optional_prep)
        sizes = [r['model']['sass_bytes'] for r in preparation['rows']]
        lines += ['', 'Verified private terminal FFN2 sharing compiles without spill; '
                  f'SASS decreases {sizes[0]}→{sizes[1]} bytes. '
                  'Default0 matches all40320 current B16 instruction words. The first attempt '
                  'compiled but failed the old artifact contract; its raw failure is retained. '
                  'The explicit optional-statistics contract checks identical geometry, arithmetic '
                  'and the absent terminal output;13 corrupted sources are rejected. '
                  'Production kernels and defaults remain unchanged.']
        model = continuation/'optional_stats_b16_v2/model/result.json'
        body = continuation/'optional_stats_b16_v2/body/result.json'
        if model.exists() and body.exists():
            assert read(model)['passed'] and read(model)['passes']=='5/5'
            assert read(body)['passed'] and len(read(body)['processes'])==17
            assert read(continuation/'optional_stats_b16_v2/model_bit_proof.json')['passed']
            lines.append('Verified ten cases form five normal/terminal pairs and cover exact/odd rows, TMA/reg stores, '
                         'TM64 merging and split-K. Control5 and candidate17 have identical bytes; '
                         'strict model5 matches the incumbent. The separate full-model pair requires '
                         'at least3% gain and the measured noise threshold before admission.')
        comparison = continuation/'optional_stats_b16_v2/experiment.json'
        if comparison.exists():
            experiment = read(comparison)
            pair = experiment['pair']
            lines.append(f'Verified GPU3 pair {pair["reference_ms"]:.6f}→'
                         f'{pair["candidate_ms"]:.6f}ms, noise{100*pair["threshold"]:.3f}%. '
                         'The code-size reduction provides no end-to-end gain; reject. '
                         'Private headers and verifier are restored; raw evidence is retained.')
    residual = continuation/'residual_warp_b16_v1/experiment.json'
    if residual.exists():
        trial = read(residual)
        order = read(residual.parent/'order/result.json')
        assert order['passed'] and len(order['processes']) == 17
        lines += ['', 'Verified residual warp handoff passes control5, body17, model5 and '
                  'the complete three-stage ordering cohort17. The ordering cohort uses GPU0/3 '
                  'and retains each process card and guard receipt. '
                  f'GPU3 pair ratio{trial["pair"]["ratio"]:.6f}, '
                  f'noise{100*trial["pair"]["threshold"]:.3f}%; no significant gain. '
                  'The private header is restored; the current artifact is retained.']
    traversal = continuation/'mfirst_b16_v1/experiment.json'
    if traversal.exists():
        trial = read(traversal)
        lines += ['', 'Verified fixed M-first task traversal passes the CPU mapping proof, '
                  'control5/body17 and strict model5 with unchanged output bytes. '
                  f'GPU3 pair ratio{trial["pair"]["ratio"]:.6f}, '
                  f'noise{100*trial["pair"]["threshold"]:.3f}%; reject. '
                  'Private backend and codegen changes are restored; no P-DNN or admission follows.']
    split_attribution = continuation/'ffn2_split2_b4_v1/attribution.json'
    if split_attribution.exists():
        trial = read(split_attribution.parent/'experiment.json')
        groups = {r['group']: r for r in read(split_attribution)['rows']}
        ffn2 = groups['ffn2']
        lines += ['', 'Verified fixed B4 FFN2 TM128/TN64/TK128/S2/Slots4 register-store '
                  'candidate passes eight-case control5/body17 and strict model5 with unchanged '
                  'FP32 thresholds and deterministic candidate bytes. Its S1-to-S2 reassociation '
                  'permits differences from incumbent bytes. '
                  f'GPU3 pair ratio{trial["pair"]["ratio"]:.6f}, '
                  f'noise{100*trial["pair"]["threshold"]:.3f}%; reject. '
                  'P-TRACE1 FFN2 main-loop CTA-maxima sums decrease '
                  f'{ffn2["control"]["main_loop_max_cta_ns"]/1000:.3f}→'
                  f'{ffn2["candidate"]["main_loop_max_cta_ns"]/1000:.3f}us, '
                  f'with {ffn2["candidate"]["splitk_max_cta_ns"]/1000:.3f}us of split-K maxima. '
                  'FFN2 epilogue maxima increase '
                  f'{ffn2["control"]["epilogue_max_cta_ns"]/1000:.3f}→'
                  f'{ffn2["candidate"]["epilogue_max_cta_ns"]/1000:.3f}us. '
                  'The three largest group-window increases are FFN1, QKV and projection; '
                  'these trace maxima are nonadditive diagnostics. Current coordinates are retained.']
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
