#!/usr/bin/env python3
"""Read-only witness for the archived split16 consumer-coordinate mismatch.

This diagnostic reads archived CUDA only to explain a stopped runtime test;
the symbolic projection and the cost model do not consume this parser.
"""
import hashlib
import json
from pathlib import Path
import re


def main():
    repo = Path(__file__).resolve().parents[3]
    source = repo/'docs/experiments/BF16/raw_splitk/src/gqa2_k16.cu'
    plan_path = repo/'docs/experiments/BF16/raw_splitk/plan/gqa2_k16.json'
    fixture = repo/'docs/experiments/SEQSCAN/raw/fixture/gqa2_s512_p0/manifest.json'
    shape = json.loads(plan_path.read_text())['variants'][0]['uniform']
    seq = json.loads(fixture.read_text())['config']['seq']
    text = source.read_text()
    gemm = re.search(r'constexpr GemmDesc kGemms\[\] = \{\s*\{(\d+), (\d+),',text)
    edge = re.search(r'\{0u, 1u, StageDependency::Map::kWindow, (\d+)u, '
                     r'(-?\d+), (-?\d+), (\d+)u\}',text)
    if not gemm or not edge:
        raise RuntimeError('archived first GEMM/window not found')
    n,k = map(int,gemm.groups())
    div,scale,offset,count = map(int,edge.groups())
    tiles_m = (seq+shape['tile_m']-1)//shape['tile_m']
    tiles_n = (n+shape['tile_n']-1)//shape['tile_n']
    tiles = tiles_m*tiles_n
    chunks = min(shape['split_k'],(k+shape['tile_k']-1)//shape['tile_k'])
    witnesses = []
    missing_tasks = 0
    for task in range(tiles*chunks):
        # GemmStageTaskBody::RunLogicalTask: chunk-major, then M/N tiles.
        chunk,local = divmod(task,tiles)
        m = local//tiles_n
        required = set(range(m*shape['tile_m'],min((m+1)*shape['tile_m'],seq)))
        # Host materialization applies the emitted window directly to task.
        at = (task//div)*scale+offset
        polled = set(range(max(at,0),min(at+count,seq)))
        missing = required-polled
        if missing:
            missing_tasks += 1
            if len(witnesses)<3:
                witnesses.append(dict(task=task,chunk=chunk,tile_m=m,
                    required_rows=[min(required),max(required)],
                    window_rows=[min(polled),max(polled)] if polled else [],
                    missing_rows=len(missing)))
    result = dict(source=str(source.relative_to(repo)),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  seq=seq,n=n,k=k,shape=shape,window=dict(div=div,scale=scale,offset=offset,count=count),
                  consumer_tasks=tiles*chunks,undercovered_tasks=missing_tasks,witnesses=witnesses,
                  scope='static incoming-edge undercoverage before queue lifting; not an exclusive dynamic-cause proof')
    if not witnesses:
        raise RuntimeError('expected coordinate counterexample absent; diagnosis falsified')
    out = repo/'docs/experiments/EVENT_COST/runtime_projection/split_coordinate_witness.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
