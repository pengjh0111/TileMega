#!/usr/bin/env python3
"""Publish guarded BERT optimization comparisons separately from final cells."""
import argparse
import csv
import json
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
              ('in-situ B1', base/'in_situ_b1_gpu1_v1/experiment.json')]
    rows = []
    choices = []
    for name, path in probes:
        if not path.exists():
            continue
        experiment = read(path)
        identity = verify(Path(experiment['library']))
        gate_path = Path(experiment['gate']).parent.parent/'result.json'
        gate = read(gate_path)
        if gate['status'] != 'passed' or gate['passes'] != '5/5' or any(
                not o.get('output_sha256') for c in gate['cases'] for o in c['outputs']):
            raise ValueError('BERT comparison requires five fresh byte-hashed FP32 gates')
        comparisons = [('GPU1 diagnostic', path.parent/'pair/result.json')]
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
                accepted=pair['ratio']<1-pair['threshold'], scope=scope,
                correctness='P-CORR5/5 with cross-process output byte hashes',
                artifact_id=identity['artifact_id'], binary_sha256=identity['binary_sha256'],
                source_digest=identity['source']['source_digest'],
                clocks=json.dumps([{k:m[k] for k in ('start_sm_clock_mhz','end_sm_clock_mhz')}
                                   for m in pair['measurements']], separators=(',', ':')),
                evidence='verified', qualification=pair['qualification'],
                gate=str(gate_path), source=str(source)))
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
             'Verified diagnostic comparisons below reuse the current frozen production coordinates and inputs. '
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
              'Final BERT batch collection continues on GPU3 after completion of all15 CNN cells and the '
              'CNN anchors. Existing CNN results and all60 baseline arms are reused. GPU ownership remains '
              'qualified by the container PID namespace. See DNN4_BERT_optimization.csv for identities, '
              'clocks, correctness receipts and raw timing sources.']
    (args.out/'DNN4_BERT_optimization.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(comparisons=len(rows), report=str(args.out/'DNN4_BERT_optimization.md'))))


if __name__ == '__main__':
    main()
