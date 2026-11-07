#!/usr/bin/env python3
"""DT-SPILL: compile every legal F3 tile shape and record its register use.

One translation unit per (mode, TM, TN, chain class), each with the real
`-Xptxas -v` and the register contract the artifact build uses. A shape that
still spills after the 64-column epilogue blocking leaves the legal set.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
NVCC = '/usr/local/cuda/bin/nvcc'
# §9.2's epilogue classes, as the chains the four models actually use.
CHAINS = {
    'plain': ('Sm90F3Step<Sm90ForwardOp::kBias,0>', 0, False, False),
    'bias_act': ('Sm90F3Step<Sm90ForwardOp::kBias,0>,'
                 'Sm90F3Step<Sm90ForwardOp::kActivation,0,DmActivation::kRelu>', 1, False, False),
    'bias_res_act': ('Sm90F3Step<Sm90ForwardOp::kScale,0>,'
                     'Sm90F3Step<Sm90ForwardOp::kBias,1>,'
                     'Sm90F3Step<Sm90ForwardOp::kResidual,2>,'
                     'Sm90F3Step<Sm90ForwardOp::kActivation,0,DmActivation::kRelu>', 3, True, False),
    'deferred_ln': ('Sm90F3Step<Sm90ForwardOp::kDeferredLayerNorm,0,DmActivation::kRelu,1>',
                    2, False, True),
    'deferred_ln_gelu': ('Sm90F3Step<Sm90ForwardOp::kDeferredLayerNorm,0,DmActivation::kRelu,1>,'
                         'Sm90F3Step<Sm90ForwardOp::kActivation,0,DmActivation::kGeluErf>',
                         2, False, True),
    'residual_ln_rowstats': ('Sm90F3Step<Sm90ForwardOp::kResidualLN,0,DmActivation::kRelu,1>',
                             2, True, True),
}
SOURCE = '''
#include <tilemega/Backend/Sm90F3Gemm.cuh>
using namespace tilemega::backend;
using tilemega::codegen::DmActivation;
using Side=Sm90F3SideLayout<{tm},{tn},{residual},{vectors},{stats}>;
using Chain=Sm90F3Chain<Side,{steps}>;
struct Conv {{static constexpr int kQ=56,kP=56,kC=64,kS=3,kR=3,kStrideW=1,kStrideH=1,
    kPadW=1,kPadH=1,kDilationW=1,kDilationH=1,kW=56,kH=56;}};
using G=Sm90F3Gemm<{tm},{tn},2,{split},Sm90F3Mode::kNormal,{m},{n},{k},{n},Chain,Conv>;
__global__ __launch_bounds__(288,1) void probe(Sm90F3GemmOperands p,char* out) {{
  extern __shared__ __align__(1024) char shared[];
  auto* header=reinterpret_cast<Sm90F3Header*>(shared);
  auto* arena=reinterpret_cast<G::Element*>(shared+Sm90F3Budget::kArena0);
  if(threadIdx.x==0){{header->rings[0].Initialize();}}
  __syncthreads();
  Sm90F3Phase phase{{}};
  auto* side=shared+Sm90F3Budget::kArena1;
  if(threadIdx.x>=256)G::Produce(p,header->rings[0],arena,side,phase,0);
  else G::Consume(p,header->rings[0],arena,side,phase,header,int(threadIdx.x)/128);
  if(threadIdx.x==0)out[0]=1;
}}
'''


def compile_one(job, folder, keep):
    tm, tn, name, split = job
    steps, vectors, residual, stats = CHAINS[name]
    # Enough rows to fill the tile, and N a whole number of TN tiles.
    m = max(512, tm * 4)
    source = folder / f'spill_{tm}x{tn}_{name}_s{split}.cu'
    source.write_text(SOURCE.format(tm=tm, tn=tn, m=m, n=tn * 4, k=576 if (576 // 64) % split == 0 else 768, split=split, steps=steps,
                                    vectors=vectors, residual=str(residual).lower(),
                                    stats=str(stats).lower()))
    done = subprocess.run(
        [NVCC, '-std=c++17', '-O3', '-gencode', 'arch=compute_90a,code=sm_90a',
         '--expt-relaxed-constexpr', '-Xptxas', '-v',
         '-Xptxas', '--device-function-maxrregcount=224',
         '-c', str(source), '-o', str(source.with_suffix('.o')),
         '-I' + str(ROOT / 'include'), '-I' + str(ROOT / 'third_party/cutlass/include')],
        capture_output=True, text=True, cwd=ROOT)
    row = dict(tile_m=tm, tile_n=tn, chain=name, split_k=split)
    if done.returncode:
        row.update(status='compile_error', error=done.stderr[-500:])
        return row
    log = done.stderr
    spills = [int(x) for x in re.findall(r'(\d+) bytes spill stores', log)]
    registers = [int(x) for x in re.findall(r'Used (\d+) registers', log)]
    row.update(status='ok', spill_bytes=max(spills) if spills else 0,
               registers=max(registers) if registers else 0,
               legal=(max(spills) if spills else 0) == 0 and
                     (max(registers) if registers else 0) <= 224)
    if not keep:
        source.with_suffix('.o').unlink(missing_ok=True)
        source.unlink(missing_ok=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/dnn3/spill')
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--keep', action='store_true')
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=True)
    jobs = [(tm, tn, chain, 1)
            for tm in (64, 128, 256) for tn in (32, 64, 128, 256) for chain in CHAINS
            if not (tm == 256 and tn == 256)]
    jobs += [(128, 64, 'deferred_ln', 3), (64, 64, 'bias_res_act', 8)]
    rows = []
    with concurrent.futures.ThreadPoolExecutor(options.jobs) as pool:
        for row in pool.map(lambda j: compile_one(j, options.out, options.keep), jobs):
            rows.append(row)
            print(json.dumps(row), flush=True)
    (options.out / 'spill.json').write_text(json.dumps(
        dict(schema=1, rows=rows), indent=2) + '\n')
    bad = [r for r in rows if r.get('status') != 'ok' or not r.get('legal')]
    print(json.dumps(dict(shapes=len(rows), illegal=len(bad))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
