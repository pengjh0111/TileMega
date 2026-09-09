#!/usr/bin/env python3
"""Require the complete fresh-process matrix before summarizing depth data."""
import csv
from pathlib import Path
import re
import torch


def main():
    here = Path(__file__).resolve().parent
    out = here/'depth_results'
    rows = list(csv.DictReader((out/'depth.tsv').open(), delimiter='\t'))
    summaries = []
    for depth in (2,4,6,8,12,16):
        group = [r for r in rows if int(r['depth']) == depth]
        if sorted(int(r['round']) for r in group) != list(range(50)):
            raise RuntimeError(f'incomplete 50-process depth={depth}')
        for row in group:
            log = (out/f'l{depth}_r{row["round"]}.txt').read_text()
            if any(pattern not in log for pattern in ('l1_vs_l05_mismatch=0 max_abs=0',
                      'l2_vs_l1_mismatch=0 max_abs=0', 'l2_iter1_vs_iter0_mismatch=0 max_abs=0')):
                raise RuntimeError('inter-level or iteration comparison failed')
        if any(len({r[key] for r in group}) != 1 for key in ('hash','mismatch','max_abs','max_rel')):
            raise RuntimeError('depth results vary across processes')
        log = (out/f'l{depth}_r0.txt').read_text()
        work = here/f'depth_work/l{depth}'
        for index, buffer in re.findall(r'E2E_OUTPUT_DIFF index=(\d+) buffer=(\d+)',log):
            for path, dtype in ((work/f'dump/buffer_{buffer}.bin',torch.bfloat16),
                                (work/f'export/fixture/reference_{index}.bin',torch.bfloat16),
                                (work/f'export/fixture/reference_fp32_{index}.bin',torch.float32)):
                value = torch.frombuffer(bytearray(path.read_bytes()),dtype=dtype)
                if not torch.isfinite(value).all(): raise RuntimeError(f'nonfinite output: {path}')
        r = group[0]
        summaries.append({'depth':depth, 'processes':50, 'passes':sum(int(x['returncode'])==0 for x in group),
                          'mismatch':r['mismatch'], 'max_abs':r['max_abs'], 'max_rel':r['max_rel'],
                          'hash':r['hash'], 'all_outputs_finite':1, 'interlevel_and_iteration_zero':50})
    with (out/'summary.tsv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=summaries[0],delimiter='\t',lineterminator='\n')
        writer.writeheader();writer.writerows(summaries)
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="320" viewBox="0 0 900 320">',
         '<rect width="900" height="320" fill="white"/><style>text{font:12px sans-serif}</style>']
    for panel,key,maximum,title in ((0,'max_abs',.1,'Maximum absolute difference'),
                                     (1,'mismatch',200,'Elements outside unchanged tolerance')):
        left=60+panel*445;top=45
        svg.append(f'<text x="{left}" y="22">{title}</text>')
        svg.append(f'<path d="M{left},{top} V{top+220} H{left+340}" stroke="black" fill="none"/>')
        points=[]
        for r in summaries:
            x=left+(r['depth']-2)*340/14;y=top+220*(1-float(r[key])/maximum)
            points.append(f'{x},{y}')
            svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="#1674bd"/>')
            svg.append(f'<text x="{x-12}" y="{y-10}">{r[key]}</text>')
            svg.append(f'<text x="{x-4}" y="{top+240}">{r["depth"]}</text>')
        svg.append(f'<polyline points="{" ".join(points)}" stroke="#1674bd" fill="none"/>')
        svg.append(f'<text x="{left+145}" y="{top+265}">depth</text>')
    svg.append('</svg>')
    (out/'depth.svg').write_text('\n'.join(svg)+'\n')
    print('DEPTH_MATRIX complete=300 finite=300 interlevel_and_iteration_zero=300')


if __name__=='__main__':
    main()
