#!/usr/bin/env python3
"""MX-4 baseline table from scripts/mx_baselines.sh output (<base>, runs/mx4/base).

LLMs: <base>/llm/<system>/<model>_rand/B<b>/round<r>/B<b>/measurements.json
(vLLM, SGLang, torch.compile); per round TPOT, TTFT and output tokens/s, the
cell value is the median over rounds (MX4 prompt §9 protocol). A round without
a result is reported with its guard reason (e.g. the 15-minute timed-job limit).
DNNs: <base>/dnn/torchcompile/<model>/Ball/round<r>/measurements.json, one row
per batch; TensorRT has no C550 build and is reported as unavailable.
Usage: baselines.py --base runs/mx4/base --llm A.csv --dnn B.csv
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--base', type=Path, required=True)
ap.add_argument('--llm', type=Path, required=True)
ap.add_argument('--dnn', type=Path, required=True)
a = ap.parse_args()

SYSTEMS = ('vllm', 'sglang', 'torchcompile')


def reason(round_dir):
    guard = round_dir / 'guard' / 'guard_result.json'
    return json.loads(guard.read_text()).get('reason', '') if guard.is_file() else 'not run'


llm = []
for system in SYSTEMS:
    for model_dir in sorted((a.base / 'llm' / system).glob('*')):
        model = model_dir.name.removesuffix('_rand')
        for batch_dir in sorted(model_dir.glob('B*'), key=lambda p: int(p.name[1:])):
            b = int(batch_dir.name[1:])
            runs, notes = [], []
            for round_dir in sorted(batch_dir.glob('round*')):
                m = round_dir / batch_dir.name / 'measurements.json'
                if (round_dir / '.done').is_file() and m.is_file():
                    runs.append(json.loads(m.read_text()))
                else:
                    notes.append('%s: %s' % (round_dir.name, reason(round_dir)))
            med = (lambda k: round(1000 * statistics.median(r[k] for r in runs), 4)) if runs else (lambda k: '')
            llm.append(dict(model=model, batch=b, system=system, rounds=len(runs),
                            tpot_ms=med('tpot_seconds'),
                            tpot_ms_rounds=' '.join('%.4f' % (1000 * r['tpot_seconds']) for r in runs),
                            ttft_ms=med('ttft_seconds'), e2e_ms=med('e2e_seconds'),
                            tok_s=round(statistics.median(r['output_tokens_per_second'] for r in runs), 2) if runs else '',
                            note='; '.join(notes)))

dnn = []
for model_dir in sorted((a.base / 'dnn' / 'torchcompile').glob('*')):
    rounds = [json.loads(p.read_text()) for p in sorted(model_dir.glob('Ball/round*/measurements.json'))]
    batches = sorted({r['batch'] for d in rounds for r in d['rows']})
    for b in batches:
        rows = [r for d in rounds for r in d['rows'] if r['batch'] == b]
        dnn.append(dict(model=model_dir.name, batch=b, rounds=len(rows),
                        torchcompile_ms=round(statistics.median(r['torchcompile_ms'] for r in rows), 4),
                        torchcompile_ms_rounds=' '.join('%.4f' % r['torchcompile_ms'] for r in rows),
                        eager_ms=round(statistics.median(r['eager_ms'] for r in rows), 4),
                        fullgraph=rows[0]['fullgraph'], tensorrt_ms='',
                        tensorrt_note='no TensorRT build for the C550 (MACA 3.8.2)'))

for path, data in ((a.llm, llm), (a.dnn, dnn)):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
    print(path, len(data), 'rows')
