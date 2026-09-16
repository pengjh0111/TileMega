#!/usr/bin/env python3
"""R5's frozen branch rule: four rotate reference cells, median path shares."""
import argparse
import csv
import statistics
from pathlib import Path


def decide(rows):
    refs=[r for r in rows if r['model'] in ('gqa2','mha4') and int(r['placement'])==5]
    if {(r['model'],int(r['seq'])) for r in refs}!={(m,s) for m in ('gqa2','mha4') for s in (4,128)} or len(refs)!=4:
        raise ValueError('FORK5 requires exactly four distinct rotate reference cells')
    load=statistics.median(float(r['phase_cp_load_wait_share_ns']) for r in refs)
    fixed=statistics.median(float(r['phase_cp_setup_share_ns'])+float(r['phase_cp_epilogue_share_ns']) for r in refs)
    math=statistics.median(float(r['phase_cp_mainloop_share_ns']) for r in refs)
    rule=1 if load>=.40 else 2 if fixed>=.40 else 3
    return f'FORK5 rule={rule} load_share={load:.3f} fixed_share={fixed:.3f} math_share={math:.3f} cells={len(refs)}'

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('analysis',type=Path)
    args=ap.parse_args()
    with args.analysis.open() as f: print(decide(list(csv.DictReader(f,delimiter='\t'))))
