#!/usr/bin/env python3
"""Report page-ring stalls, realized chain bubbles, and adjacent-launch gaps."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import statistics


def rows(path: Path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def floor_at(points: Path, model: str, batch: int, past: int) -> float:
    source=rows(points)
    if 'floor_ns' in source[0]:
        exact={int(row['past']):float(row['floor_ns'])/1e6 for row in source}
        if past not in exact:raise ValueError('CG floor table lacks the requested step')
        return exact[past]
    pair=sorted((int(r['past']),float(r['floor_ms'])) for r in source
                if r['model']==model and int(r['batch'])==batch)
    if len(pair)!=3 or not pair[0][0]<=past<=pair[-1][0]:
        raise ValueError('CG floor table does not cover this step')
    left,right=(pair[0],pair[1]) if past<=pair[1][0] else (pair[1],pair[2])
    return left[1]+(right[1]-left[1])*(past-left[0])/(right[0]-left[0])


def analyze(page_trace: Path, *, model: str, batch: int, prompt_len: int,
            floor_points: Path, chain_analysis: Path|None=None):
    by_step={}
    for row in rows(page_trace):
        step=int(row['step'])
        begin=int(row['kernel_begin_ns']);end=int(row['kernel_end_ns'])
        if not begin and not end:continue
        if begin<=0 or end<begin:raise ValueError(f'invalid kernel stamps at step {step}')
        by_step.setdefault(step,[]).append(row)
    if not by_step:raise ValueError('page trace has no launched steps')
    measured_chain=None
    measured_chain_span=None
    if chain_analysis:
        chain=rows(chain_analysis)
        if len(chain)!=1:raise ValueError('expected one realized-chain trace')
        measured_chain=int(chain[0]['cp_nodes'])
        measured_chain_span=int(chain[0]['cp_chain_span_ns'])
        if measured_chain<=0:raise ValueError('empty realized chain')
    output=[];previous_end=None
    for step,ctas in sorted(by_step.items()):
        if len(ctas)!=len(by_step[next(iter(by_step))]):
            raise ValueError('incomplete CTA stamps')
        start=min(int(x['kernel_begin_ns']) for x in ctas)
        end=max(int(x['kernel_end_ns']) for x in ctas)
        gap_ns=start-previous_end if previous_end is not None else None
        past=prompt_len+step
        floor_ms=floor_at(floor_points,model,batch,past)
        span_ns=end-start
        dependency=sum(int(x['dependency_wait_ns']) for x in ctas)
        full=sum(int(x['page_full_ns']) for x in ctas)
        overlap=sum(int(x['full_and_wait_ns']) for x in ctas)
        if overlap>min(dependency,full):raise ValueError('page/dependency intersection exceeds a marginal')
        output.append(dict(model=model,batch=batch,step=step,past=past,
            ctas=len(ctas),kernel_span_ns=span_ns,floor_ns=round(floor_ms*1e6),
            kernel_over_floor=span_ns/(floor_ms*1e6),
            realized_chain_links=measured_chain,
            measured_chain_span_ns=measured_chain_span,
            measured_chain_over_floor=(measured_chain_span/(floor_ms*1e6)
                if measured_chain_span is not None else None),
            residual_bubble_ns_per_link=((span_ns-floor_ms*1e6)/measured_chain
                if measured_chain else None),
            dependency_wait_cta_ns=dependency,page_full_cta_ns=full,
            page_full_and_dependency_wait_cta_ns=overlap,
            page_full_and_dependency_wait_mean_cta_ns=overlap/len(ctas),
            launch_gap_ns=gap_ns))
        previous_end=end
    gaps=[r['launch_gap_ns'] for r in output if r['launch_gap_ns'] is not None]
    summary=dict(model=model,batch=batch,steps=len(output),
        chain_source=str(chain_analysis) if chain_analysis else None,
        floor_source=str(floor_points),page_source=str(page_trace),
        measured_chain_links=measured_chain,
        measured_chain_span_ns=measured_chain_span,
        launch_gap_mean_ns=statistics.mean(gaps) if gaps else None,
        launch_gap_p50_ns=statistics.median(gaps) if gaps else None,
        page_full_and_dependency_wait_cta_ns=sum(r['page_full_and_dependency_wait_cta_ns'] for r in output),
        residual_bubble_p50_ns_per_link=statistics.median(
            r['residual_bubble_ns_per_link'] for r in output) if measured_chain else None)
    return output,summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--page-trace',type=Path,required=True)
    ap.add_argument('--model',choices=['llama','qwen3'],required=True)
    ap.add_argument('--batch',type=int,required=True)
    ap.add_argument('--prompt-len',type=int,default=64)
    ap.add_argument('--floor-points',type=Path,default=Path('docs/experiments/SERVING_R10/report_tables/floor_points.tsv'))
    ap.add_argument('--floor-steps',type=Path,
                    help='exact per-step CG floor from tilemega inspect request-floor')
    ap.add_argument('--chain-analysis',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    samples,summary=analyze(args.page_trace,model=args.model,batch=args.batch,
        prompt_len=args.prompt_len,floor_points=args.floor_steps or args.floor_points,
        chain_analysis=args.chain_analysis)
    with (args.out/'page_chain.tsv').open('w') as stream:
        out=csv.DictWriter(stream,fieldnames=samples[0].keys(),delimiter='\t');out.writeheader();out.writerows(samples)
    (args.out/'page_chain_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))

if __name__=='__main__':main()
