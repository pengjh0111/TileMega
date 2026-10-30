#!/usr/bin/env python3
"""Native solver measure command: P-SEL over the compiled top-K (prompt §2.3).

Invoked by `tilemega compile --solve ... --measure-cmd` as
`sm90r3_select.py OPTIONS --candidates FILE --out DIR`. Blocks until the
guarded scheduler has measured A B C D D C B A, then writes DIR/result.json.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from sm90d1_common import ROOT, PYTHON, read, sha, write
from sm90r4_dispatch import run_unit

LLM = ROOT / 'scripts/sm90r3_llm.py'
POLICY = ROOT / 'configs/bench/sm90r2_guard_policy.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--prefill', required=True,
                   help="the prefill binary, or 'prompt' for the prompt mode (§8 A3.5)")
    p.add_argument('--gpu', required=True, help="a card index, or 'auto' for the first free card")
    p.add_argument('--name', required=True)
    p.add_argument('--concurrency', default='serial', choices=('serial', 'trial', 'verified'))
    p.add_argument('--candidates', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True)
    rows = [r for r in read(a.candidates) if r['status'] == 'compiled']
    if len(rows) < 2:
        raise ValueError('P-SEL needs at least two compiled candidates')
    # 'prompt' is a sentinel, not a path: it must survive into the spec.
    prefill = 'prompt' if str(a.prefill) == 'prompt' else str(Path(a.prefill).resolve())
    arms = {f"c{r['rank']}": dict(decode=r['binary'], prefill=prefill) for r in rows}
    spec = dict(name=a.name, kind='sel', model=a.model, batch=a.batch, gpu=a.gpu,
                concurrency=a.concurrency, arms=arms,
                predicted_ns={f"c{r['rank']}": r['predicted_ns'] for r in rows})
    write(out / 'spec.json', spec)
    if a.gpu == 'auto':
        # §1.5: a MoE P-SEL runs with the machine to itself.
        # §1.5 gives a MoE P-SEL the whole machine because R3 measured 4-15%
        # MoE variance on parallel cards. The user relaxed that on 2026-10-30:
        # --concurrency now chooses the claim, for MoE as for dense, and every
        # cell records the concurrency it was measured under.
        cls = {'trial': 'trial', 'verified': 'trial', 'serial': 'node'}[a.concurrency]
        write(out / 'psel.json', run_unit(out / 'spec.json', out / 'unit', cls))
    else:
        gpu = int(a.gpu)
        subprocess.check_call([PYTHON, str(LLM), 'queue', '--spec', str(out / 'spec.json'),
                               '--queue-dir', str(out / 'queue')], cwd=ROOT)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TILEMEGA_DEVICE_INDEX=str(gpu),
                   TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{gpu}.lock', TILEMEGA_NUMA_NODE=str(0 if gpu < 4 else 1),
                   PYTHONPATH=str(ROOT / 'python'))
        subprocess.check_call([PYTHON, '-m', 'tilemega.bench.scheduler', '--queue-dir', str(out / 'queue'),
                               '--out', str(out / 'scheduler'), '--policy', str(POLICY),
                               '--deadline-hours', '24'], cwd=ROOT, env=env)
        subprocess.check_call([PYTHON, str(LLM), 'collect', '--spec', str(out / 'spec.json'),
                               '--state', str(out / 'scheduler/state.json'), '--out', str(out / 'psel.json')],
                              cwd=ROOT)
    psel = read(out / 'psel.json')
    winner = psel['winner']
    binary = arms[winner]['decode']
    write(out / 'result.json', dict(schema=1, protocol='P-SEL', winner_rank=int(winner[1:]),
                                    winner_binary_sha256=sha(binary), measured_order=psel['measured_order'],
                                    arms=psel['arms'], predicted_ns=spec['predicted_ns'],
                                    top_gap_pct=psel['top_gap_pct'], threshold_pct=psel['threshold_pct'],
                                    tie_broken_by_prediction=psel['tie_broken_by_prediction']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
