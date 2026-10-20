#!/usr/bin/env python3
"""Rewrite every geometry's `stages` in a pin file to one depth.

MX3-2 item 5 wants the same cell, same geometry, at depth 1 (the serial
mainloop) against the measured best depth. Only `stages` changes, so the
comparison isolates the pipeline from the tile choice.
"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('pin', type=Path)
ap.add_argument('--stages', type=int, required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--residency', type=int,
                help='CTAs per AP; mma16 decode attention needs more shared memory '
                     'than four CTAs can share at D=128')
a = ap.parse_args()


def fits(tile_m, tile_n, stages, waves=2):
    """solver::ServingGemmDepthFits: depth 1 always, deeper within 32 VGPRs."""
    loads = tile_n // 16 // waves + tile_m // 16
    return stages == 1 or stages * loads * 4 <= 32


doc = json.loads(a.pin.read_text())
clamped = 0
for case in doc['cases']:
    if a.residency:
        case['residency'] = a.residency
    for g in case['geometries']:
        want = a.stages
        while want > 1 and not fits(g['tile_m'], g['tile_n'], want):
            want -= 1
            clamped += 1
        g['stages'] = want
a.out.write_text(json.dumps(doc, indent=2) + '\n')
print('%s -> stages=%d (%d geometries clamped to fit registers) -> %s'
      % (a.pin.name, a.stages, clamped, a.out))
