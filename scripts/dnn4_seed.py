#!/usr/bin/env python3
"""Generate §4.5 F3 seed coordinates using the body's canonical arena layout."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HOST_QUERY = r'''
#include <cstdio>
using namespace tilemega::backend;
using namespace tilemega::codegen::forward_f3;
template<class B> struct LayoutOf;
template<int TM,int TN,int Slots,int S,Sm90F3Mode Mode,int M,int N,int K,int NS,class Chain,class Conv,class Pool>
struct LayoutOf<Sm90F3Gemm<TM,TN,Slots,S,Mode,M,N,K,NS,Chain,Conv,Pool>> {
 static constexpr auto Get(int slots) {
  return Sm90F3MakeGemmLayout(TM,TN,slots,false,0,Chain::kNeedsResidual,
   Chain::Layout::kVectorBytes/(4*TN),Chain::kNeedsStats,
   2*Sm90F3Split<TM,TN>::kWgCols,Pool::kArea?Pool::kImages:0);
 }
};
template<int Slots,int S,int M,int N,int K,int NS,class Chain>
struct LayoutOf<Sm90F3GemmSwapAB<Slots,S,M,N,K,NS,Chain>> {
 static constexpr auto Get(int slots) {
  constexpr int rows=M<=8?8:16;
  return Sm90F3MakeGemmLayout(rows,128,slots,true,rows,false,
   Chain::Layout::kVectorBytes/(4*128),false,0,0);
 }
};
template<class B> void Show(int index) {
 for(int slots=2;slots<=8;++slots) {
  auto l=LayoutOf<B>::Get(slots);
  std::printf("%d %d %d %d %d\n",index,slots,l.end,int(!l.Spanning()),int(Sm90F3Budget::Fits(l.end)));
 }
}
'''


def split_for(stage, tm, tn):
    iterations = math.ceil(stage['k'] / 64)
    tiles = math.ceil(stage['m'] / tm) * math.ceil(stage['n'] / tn)
    return max([s for s in range(1, 17) if iterations % s == 0
                and tiles*s <= 132 and iterations//s >= 6] or [1])


def seed(stage):
    c = dict(body=stage['body'], mode=stage['mode'], tile_m=64, tile_n=64,
             split_k=1, slots=2, tk=64, store='reg', mc=0,
             tile_w=8, tile_h=8, channel_block=64, fused=False)
    if stage['body'] not in ('gemm', 'conv'):
        if stage['body'] not in ('layout', 'pool', 'global_pool', 'layer_norm',
                                 'embedding', 'encoder_attention'):
            raise ValueError('no DNN4 seed for body: ' + stage['body'])
        return c
    m, n, k = (stage[key] for key in ('m', 'n', 'k'))
    iterations = math.ceil(k / 64)
    divisors = [s for s in range(1, 33) if iterations % s == 0]
    if c['mode'] == 'swapab':
        c['tile_n'] = 128
        sufficient = [s for s in divisors if math.ceil(n / 128)*s >= 120]
        c['split_k'] = min(sufficient) if sufficient else max(divisors)
    elif c['mode'] == 'gather':
        c['tile_m'], c['tile_n'] = 128, 64
    elif c['mode'] == 'normal':
        if 'has_row_stats' not in stage:
            raise ValueError('regenerate the F3 plan with explicit row-stat metadata')
        tm = 64 if m <= 4096 else 256 if m >= 8192 and n <= 128 else 128
        padded_n = math.ceil(n / 32)*32
        choices = [tn for tn in (256, 128, 64, 32) if padded_n % tn == 0
                   and math.ceil(m / tm)*(padded_n//tn) >= 66
                   and (not stage['has_row_stats'] or tn % 64 == 0)]
        tn = max(choices or [64 if stage['has_row_stats'] else 32])
        c.update(tile_m=tm, tile_n=tn, split_k=split_for(stage, tm, tn))
    else:
        raise ValueError('unknown F3 mode')
    return c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--bridge', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=ROOT / 'runs/dnn4/build/main/tools/tilemega')
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError('retain seed evidence; choose an empty output directory')
    plan = json.loads(args.plan.read_text())
    if plan['abi'] != 'f3':
        raise ValueError('requires an F3 plan')
    coordinates = dict(schema=4, model='dnn', abi='f3', residency=1,
                       stages=[seed(stage) for stage in plan['stage_rows']])
    prototype = args.out / 'prototype.tmexec.json'
    prototype.write_text(json.dumps(coordinates, indent=2) + '\n')
    source = args.out / 'prototype.cu'
    command = [str(args.binary.resolve()), 'compile', str(args.bridge.resolve()), str(source),
        '--emit', 'serving', '--serving', 'forward', '--frontend', 'dnn',
        '--batch', str(plan['batch']), '--abi', 'f3', '--tmexec-config', str(prototype),
        '--runtime-target', str(ROOT / 'configs/targets/sm_90_h800.json')]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    exclusions = []
    with (args.out / 'codegen.log').open('w') as stream:
        while True:
            done = subprocess.run(command, env=environment, cwd=ROOT, capture_output=True, text=True)
            stream.write(done.stdout + done.stderr)
            stream.flush()
            if done.returncode == 0:
                break
            rejected = re.search(r'DT-SPILL excluded this F3 tile and chain at stage (\d+): (\d+)x(\d+) (\w+)',
                                 done.stderr + done.stdout)
            if rejected is None:
                raise RuntimeError('F3 seed codegen failed; retain ' + str(args.out / 'codegen.log'))
            index = int(rejected[1])
            c = coordinates['stages'][index]
            if c['tile_n'] <= 32 or c['mode'] != 'normal':
                raise ValueError('no smaller legal normal seed tile')
            exclusions.append(dict(stage=index, tile_m=c['tile_m'], tile_n=c['tile_n'],
                                   chain=rejected[4], reason='measured DT-SPILL exclusion'))
            minimum = 64 if plan['stage_rows'][index]['has_row_stats'] else 32
            if c['tile_n'] <= minimum:
                raise ValueError('no smaller legal row-stat seed tile')
            c['tile_n'] //= 2
            stage = plan['stage_rows'][index]
            c['split_k'] = split_for(stage, c['tile_m'], c['tile_n'])
            prototype.write_text(json.dumps(coordinates, indent=2) + '\n')
    (args.out / 'seed.exclusions.json').write_text(json.dumps(exclusions, indent=2) + '\n')
    matrices = [i for i, c in enumerate(coordinates['stages']) if c['body'] in ('gemm', 'conv')]
    host = args.out / 'layout_probe.cu'
    host.write_text(source.read_text().split('#include <tilemega/Codegen/tasks/Sm90F3Runtime.cuh>')[0]
        + HOST_QUERY + '\nint main(){\n' + ''.join(f'Show<Body_{i}>({i});\n' for i in matrices) + '}\n')
    executable = args.out / 'layout_probe'
    command = ['taskset', '-c', '136-143', 'nvcc', '-std=c++17', '-O3',
        '-gencode', 'arch=compute_90a,code=sm_90a', '--expt-relaxed-constexpr',
        '-Xptxas', '-v', '-I' + str(ROOT / 'include'),
        '-I' + str(ROOT / 'third_party/cutlass/include'), str(host), '-o', str(executable)]
    with (args.out / 'layout_probe.ptxas.log').open('w') as stream:
        subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
    measured = subprocess.check_output([str(executable)], env=environment, text=True)
    (args.out / 'layout_probe.txt').write_text(measured)
    layouts = []
    for index in matrices:
        choices = [list(map(int, row.split())) for row in measured.splitlines()
                   if int(row.split()[0]) == index]
        feasible = [row for row in choices if row[4]]
        fitting_arena = [row for row in feasible if row[3]]
        if not feasible:
            raise ValueError('stage does not fit F3 shared budget: ' + str(index))
        chosen = max(fitting_arena or feasible, key=lambda row: row[1])
        if chosen[1] < 4:
            chosen = max(feasible, key=lambda row: row[1])
        coordinates['stages'][index]['slots'] = chosen[1]
        layouts.append(dict(stage=index, slots=chosen[1], shared_bytes=chosen[2], spanning=not chosen[3]))
    (args.out / 'seed.tmexec.json').write_text(json.dumps(coordinates, indent=2) + '\n')
    (args.out / 'seed.layout.json').write_text(json.dumps(layouts, indent=2) + '\n')
    print(json.dumps(dict(batch=plan['batch'], stages=len(coordinates['stages']),
                         spanning=sum(row['spanning'] for row in layouts), coordinates=str(args.out / 'seed.tmexec.json'))))


if __name__ == '__main__':
    main()
