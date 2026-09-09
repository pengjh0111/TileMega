#!/usr/bin/env python3
"""Offline task mapping sweep; never launches or changes a CUDA kernel."""
import argparse
import csv
from html import escape
import math
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', type=Path, default=Path('build-portable/tools/tilemega-affine-probe'))
    parser.add_argument('--out', type=Path, default=Path('docs/experiments/AFFINE_PROBE/balanced'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seq in (4, 128, 512):
        for workers in (16, 256):
            result = subprocess.run([str(args.probe), str(seq), str(workers), '256'],
                                    capture_output=True, text=True,
                                    env=dict(os.environ, TILEMEGA_ISL_AUDIT='1'))
            (args.out / f's{seq}_w{workers}.log').write_text(result.stdout + result.stderr)
            if result.returncode or 'ISL_CONTEXT remaining=0' not in result.stderr:
                raise RuntimeError(f'probe or ownership audit failed: seq={seq}, workers={workers}')
            case = [dict(field.split('=', 1) for field in line.split()[1:])
                    for line in result.stdout.splitlines() if line.startswith('SPAN ')]
            if len(case) != 13:
                raise RuntimeError(f'incomplete mapping matrix: {len(case)}')
            base = next(row for row in case if row['mode'] == 'stage_major')
            for row in case:
                queue, local = int(row['max_queue']), int(row['same_worker_edges'])
                # Exact integer counts, not rounded printed fractions.
                row['pareto'] = int(not any(
                    int(other['max_queue']) <= queue and int(other['same_worker_edges']) >= local
                    and (int(other['max_queue']) < queue or int(other['same_worker_edges']) > local)
                    for other in case))
                row['gate'] = int('_balanced_' in row['mode']
                                  and 5 * queue <= 6 * int(base['max_queue'])
                                  and local > int(base['same_worker_edges']))
                row['strict_dominance'] = int(queue <= int(base['max_queue'])
                                             and local > int(base['same_worker_edges']))
            rows.extend(case)
            print(f'seq={seq} workers={workers} gate={sum(r["gate"] for r in case)}', flush=True)
    with (args.out / 'mappings.tsv').open('w') as out:
        writer = csv.DictWriter(out, fieldnames=rows[0], delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="700" viewBox="0 0 1080 700">',
           '<rect width="1080" height="700" fill="white"/>',
           '<style>text{font:12px sans-serif} .point{fill:#999} .front{fill:#1674bd}</style>',
           '<text x="50" y="20">Offline Pareto frontier: x = longest queue (log10); y = same-worker fraction</text>']
    for panel, (seq, workers) in enumerate((s, w) for s in (4, 128, 512) for w in (16, 256)):
        case = [r for r in rows if int(r['seq']) == seq and int(r['workers']) == workers]
        left, top = 55 + (panel % 3)*355, 70 + (panel // 3)*320
        maximum = max(int(r['max_queue']) for r in case)
        ymax = max(float(r['same_worker_fraction']) for r in case)*1.1
        def xy(row):
            return (left + 280*math.log10(max(1, int(row['max_queue'])))/math.log10(maximum),
                    top + 230*(1 - float(row['same_worker_fraction'])/ymax))
        svg.append(f'<text x="{left}" y="{top-18}">seq={seq}, workers={workers}</text>')
        svg.append(f'<path d="M{left},{top} V{top+230} H{left+280}" stroke="black" fill="none"/>')
        for tick in (1, 10, 100, 1000):
            if tick > maximum: continue
            x = left + 280*math.log10(tick)/math.log10(maximum)
            svg.append(f'<text x="{x}" y="{top+248}">{tick}</text>')
        for i in range(5):
            value = ymax*i/4
            svg.append(f'<text x="{left-45}" y="{top+230-230*i/4}">{value:.3f}</text>')
        frontier = sorted((r for r in case if r['pareto']), key=lambda r: int(r['max_queue']))
        points = ' '.join(f'{x},{y}' for x, y in map(xy, frontier))
        svg.append(f'<polyline points="{points}" fill="none" stroke="#1674bd"/>')
        for row in case:
            x, y = xy(row)
            color = '#d53232' if row['mode'] == 'stage_major' else '#1674bd' if row['pareto'] else '#999'
            svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"><title>{escape(row["mode"])}</title></circle>')
            if row['mode'] == 'stage_major':
                svg.append(f'<text x="{x-35}" y="{y-8}" fill="#d53232">stage-major</text>')
    svg.append('</svg>')
    (args.out / 'pareto.svg').write_text('\n'.join(svg)+'\n')
    (args.out / 'status.txt').write_text('PASS: 6 offline instances, 78 mappings; no GPU claim\n')


if __name__ == '__main__':
    main()
