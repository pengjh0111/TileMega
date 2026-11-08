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
              ('P1 normalization B16 G22', base/'side_norm_models_v2/b16/g22_v1/experiment.json')]
    rows = []
    choices = []
    for name, path in probes:
        if not path.exists():
            continue
        experiment = read(path)
        if name.startswith('P1 normalization'):
            unit = read(base/'side_norm_unit_v2/raw/dnn4-focused-f3-gates/attempt1/cohort/result.json')
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
                accepted=pair['ratio']<1-max(pair['threshold'], .03 if name.startswith('Repeat') else 0), scope=scope,
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
             'GPU1 diagnostics guide optimization; G22 diagnostics compare on GPU3 against rank0 and do not '
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
              'The prototype headers are restored. Verified B8 has incompatible canonical buffer '
              'aliases;codegen now retains its exact flat program instead of failing. Four real '
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
              'GPU3 foreign occupancy delayed additional same-card G22 probes; only our pending '
              'schedulers were cancelled, and fresh candidate/control pairs moved to authorized GPU2. '
              'The separately identified B1 follow-up does not silently replace the earlier frozen20-cell collection. '
              'That BERT batch collection '
              'completed on GPU3: B1/B2/B4/B8/B16 =0.614016/0.695504/0.790144/1.043264/1.463168ms. '
              'None beats its frozen BF16 target. CNN collection is complete and paused; '
              'existing CNN results and all60 baseline arms are reused. GPU ownership remains '
              'qualified by the container PID namespace. See DNN4_BERT_optimization.csv for identities, '
              'clocks, correctness receipts and raw timing sources.']
    (args.out/'DNN4_BERT_optimization.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(comparisons=len(rows), report=str(args.out/'DNN4_BERT_optimization.md'))))


if __name__ == '__main__':
    main()
