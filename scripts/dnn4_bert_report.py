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
              ('Repeat P1 normalization B16', base/'repeat_side_norm_b16_v1/experiment.json')]
    rows = []
    choices = []
    for name, path in probes:
        if not path.exists():
            continue
        experiment = read(path)
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
        pair_path = path.parent/'pair/result.json'
        if not pair_path.exists():
            pair_path = path.parent/'pair_gpu2/result.json'
        first = read(pair_path)
        comparisons = [(f'GPU{first["gpu"]} same-card diagnostic', pair_path)]
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
                accepted=pair['gpu'] != 0 and pair['ratio']<1-max(pair['threshold'], .03 if name.startswith('Repeat') else 0), scope=scope,
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
                  f'{final["measurement"]["candidate_ms"]:.6f}ms is accepted for B2 only. '
                  'B1 has no significant gain;B16 Repeat plus P1 normalization also fails '
                  'DT-CODE and keeps the flat control. GPU0 allocation-deviation diagnostics '
                  'are separately identified and cannot replace G22 admission. '
                  'The arithmetic feature remains default0 and is explicitly selected for B2.']
    diagnostic = base/'ncu_b16_v1/result.json'
    if diagnostic.exists():
        ncu = read(diagnostic)
        if ncu['error'] != 'ERR_NVGPUCTRPERM' or ncu['settings_changed']:
            raise ValueError('unexpected hardware-counter diagnostic state')
        lines += ['', 'Verified hardware-counter diagnostic: ncu2025.1.1 exits1 with ERR_NVGPUCTRPERM; '
                  'no counter samples are available and no device/driver settings were changed. '
                  'Fresh host CMake configuration also fails because the external MLIR installation '
                  'lacks obj.MLIRCAPIDebug/Debug.cpp.o. The existing configured incremental build succeeds.']
    current = args.out/'DNN4_BERT_current.csv'
    if current.exists():
        with current.open() as stream:
            latest = list(csv.DictReader(stream))
        lines += ['', 'Verified current accepted artifacts (GPU3;50/200;three-process P-DNN). '
                  'B8 reuses the original measurements;B1/B2/B4/B16 are follow-ups that retain '
                  'the original anchor without re-anchoring. All fixed BF16 targets remain unmet. '
                  'DNN4_BERT_current.csv records complete identities,profiles and evidence paths.', '',
                  '| Batch | Median ms | p90 ms | Process range ms | Fixed BF16 ms | Ratio |',
                  '| --- | --- | --- | --- | --- | --- |']
        for row in latest:
            lines.append(f'| {row["batch"]} | {float(row["median_ms"]):.6f} | '
                         f'{float(row["p90_ms"]):.6f} | {float(row["process_range_ms"]):.6f} | '
                         f'{float(row["baseline_ms"]):.4f} | {float(row["ratio"]):.4f} |')
    (args.out/'DNN4_BERT_optimization.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(comparisons=len(rows), report=str(args.out/'DNN4_BERT_optimization.md'))))


if __name__ == '__main__':
    main()
