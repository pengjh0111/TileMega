#!/usr/bin/env python3
"""One serving CSV across campaigns, each row labelled with what produced it.

MX3-0 item 2. The MX-1 CSV carried a `note` column that MX-2 backfilled with
MX-2's selected implementation while the measurements stayed MX-1's, so a
row-major 22.33 ms number was labelled `layout=tiled gemm=mma16` -- the
configuration that produced 15.99 ms. Labels are read here from each
campaign's own build command, never from another campaign's.
"""
import argparse, csv, json, re
from pathlib import Path

# Repository root, so a clone anywhere works (MX3-0 item 1).
REPO = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument('--root', type=Path, default=REPO / 'runs')
ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()

PEAK_BYTES_PER_S = 1.64e12
GEOMETRY = {
    'qwen3_0_6b': (1.11 * 2**30, 28, 8, 128),
    'qwen3_1_7b': (3.20 * 2**30, 28, 8, 128),
    'llama3_2_1b': (2.30 * 2**30, 16, 8, 64),
    'llama3_2_3b': (5.98 * 2**30, 28, 8, 128),
    'llama3_1_8b': (14.96 * 2**30, 32, 8, 128),
    'qwen3_30b_a3b_l24': (29.02 * 2**30, 24, 4, 128),
}
MOE = dict(layers=24, hidden=2048, experts=128, top_k=8, moe_inter=768)


def moe_weight_bytes(batch):
    m = MOE
    attention = (2 * m['hidden'] ** 2 + 2 * 4 * 128 * m['hidden']) * 2
    router = m['hidden'] * m['experts'] * 2
    per_expert = 3 * m['hidden'] * m['moe_inter'] * 2
    distinct = m['experts'] * (1 - (1 - 1 / m['experts']) ** (batch * m['top_k']))
    return m['layers'] * (attention + router + distinct * per_expert)


def mbu(model, batch, tpot_ms, context):
    key = model.replace('_rand', '')
    if key not in GEOMETRY or not tpot_ms or tpot_ms != tpot_ms:
        return ''
    weights, layers, kv_heads, head_dim = GEOMETRY[key]
    if key == 'qwen3_30b_a3b_l24':
        weights = moe_weight_bytes(batch)
    kv = 2 * batch * layers * kv_heads * head_dim * context * 2
    return '%.6f' % ((weights + kv) / (tpot_ms * 1e-3) / PEAK_BYTES_PER_S)


def label(cell: Path) -> dict:
    """Layout and implementation, read from this cell's own build command."""
    out = dict(layout='', impl='')
    command = cell / 'plan.so.build_command.txt'
    if not command.exists():
        return out
    text = command.read_text(errors='ignore')
    macros = {k: int(v) for k, v in
              re.findall(r'-D(TILEMEGA_[A-Z0-9_]+)=(-?\d+)', text)}
    out['layout'] = 'tiled' if macros.get('TILEMEGA_NONPAGED_TILED') else 'row'
    gemv = sorted(k.rsplit('_', 1)[-1] for k, v in macros.items()
                  if re.fullmatch(r'TILEMEGA_GEMM_IMPL_\d+', k) and v)
    bits = ['gemm=' + ('gemv:' + ','.join(gemv) if gemv else 'mma16')]
    if 'TILEMEGA_ATTENTION_IMPL' in macros:
        bits.append('attn=' + ('simt' if macros['TILEMEGA_ATTENTION_IMPL'] == 3 else 'mma16'))
    cu = cell / 'plan.so.cu'
    if cu.exists():
        tile = {}
        for line in cu.open():
            m = re.match(r'#define TILEMEGA_GEMM_TILE_([MNK]) (\d+)', line)
            if m:
                tile[m.group(1)] = m.group(2)
            m = re.match(r'#define TILEMEGA_GEMM_STAGES (\d+)', line)
            if m:
                tile['S'] = m.group(1)
            if len(tile) == 4:
                break
        if len(tile) == 4:
            bits.append('tile=%sx%sx%ss%s' % (tile['M'], tile['N'], tile['K'], tile['S']))
        if macros.get('TILEMEGA_GEMM_VARIANT_COUNT'):
            bits.append('variants=%d' % macros['TILEMEGA_GEMM_VARIANT_COUNT'])
    out['impl'] = ' '.join(bits)
    return out


rows = []
for campaign, base in (('mx1', a.root / 'mx1' / 'base'),):
    for path in sorted(base.glob('*/*/round*/B*/measurements.json')):
        d = json.loads(path.read_text())
        system, model, rnd = path.parts[-5], path.parts[-4], path.parts[-3]
        if 'stale' in rnd:
            continue
        rows.append(dict(campaign=campaign, system=system,
                         model=model.replace('_rand', ''), round=rnd,
                         batch=d['batch'], ttft_ms=d['ttft_seconds'] * 1e3,
                         tpot_ms=d['tpot_seconds'] * 1e3, e2e_s=d['e2e_seconds'],
                         tokens_per_s=d['output_tokens_per_second'],
                         max_tokens=d['max_tokens'], layout='n/a',
                         impl=(d.get('grouped_mm_error') and 'experts=eager' or
                               ('fullgraph=%s' % d['fullgraph'] if 'fullgraph' in d else ''))))

# tm-mx2-unpinned: the first single-HEAD rebuild, whose pins could not carry
# Ec/Rq or hold residency, so the search re-chose them (MX3-0). tm-mx2: the
# rebuild with all three pinned, which reproduces MX-2 to within 0.2%.
for campaign, pattern in (('mx1', a.root / 'mx1' / 'tm'), ('mx2', a.root / 'mx2' / 'tm'),
                          ('tm-mx2-unpinned', a.root / 'mx3' / 'tm'),
                          ('tm-mx2', a.root / 'mx3' / 'tm2'),
                          ('mx3', a.root / 'mx3' / 'final'),
                          # The rerun after the descriptor prefetch, the attention
                          # coordinate and the in-situ recalibration (26632fa26).
                          ('mx3-rerun', a.root / 'mx3' / 'final2'),
                          # Second rerun: direct split-K partial stores, recalibrated
                          # task bodies (c11ff37bc; 8B decode re-solved at 9bc090997).
                          ('mx3-rerun2', a.root / 'mx3' / 'final3')):
    for path in sorted(pattern.glob('*/B*/measure/round*/measurements.json')):
        d = json.loads(path.read_text())
        model, batch, rnd = path.parts[-5], path.parts[-4], path.parts[-2]
        cell = path.parents[2] / 'decode'
        tags = label(cell)
        rows.append(dict(campaign=campaign, system='tilemega', model=model, round=rnd,
                         batch=int(batch.lstrip('B')),
                         ttft_ms=d.get('ttft_seconds', float('nan')) * 1e3,
                         tpot_ms=d.get('tpot_seconds', float('nan')) * 1e3,
                         e2e_s=d.get('e2e_seconds', float('nan')),
                         tokens_per_s=d.get('output_tokens_per_second', float('nan')),
                         max_tokens=d.get('max_tokens', 0), **tags))

for row in rows:
    context = 64 + (row['max_tokens'] or 1) / 2
    row['mbu'] = mbu(row['model'], row['batch'], row['tpot_ms'], context)

order = ['campaign', 'system', 'model', 'batch', 'round', 'ttft_ms', 'tpot_ms',
         'e2e_s', 'tokens_per_s', 'mbu', 'max_tokens', 'layout', 'impl']
with a.out.open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=order)
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda r: (r['campaign'], r['system'],
                                                 r['model'], r['batch'], r['round'])))
print('%d rows -> %s' % (len(rows), a.out))
for c in ('mx1', 'mx2', 'tm-mx2-unpinned', 'tm-mx2', 'mx3'):
    tm = [r for r in rows if r['campaign'] == c and r['system'] == 'tilemega']
    layouts = sorted({r['layout'] for r in tm})
    print('  %s: %d tilemega rows, layouts %s' % (c, len(tm), layouts))
