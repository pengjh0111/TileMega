#!/usr/bin/env python3
"""Generate the earlier DNN1 probe recipes without ignored campaign builders.

The source root is explicit: changed production headers require new measurements.
This command never initializes CUDA or enqueues measurements.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
RECIPES = {'base': {'unchanged': True},
 'live16': {'add_live16': True},
 'float4': {'unchanged': True},
 'tm32': {'catalogs': ['void Catalog() {\n'
                       '  Epilogues<32,32>();Epilogues<32,64>();Epilogues<32,128>();Epilogues<32,256>();\n'
                       '  '
                       'EpilogueCase<32,8,Sm90ForwardEpilogueStep<Sm90ForwardOp::kBias>,1,3>("scalar_2byte");\n'
                       '  '
                       'EpilogueCase<32,16,Sm90ForwardEpilogueStep<Sm90ForwardOp::kBias>,2>("vector_16byte");\n'
                       '  '
                       'EpilogueCase<32,24,Sm90ForwardEpilogueStep<Sm90ForwardOp::kBias>,2>("vector_16byte");\n'
                       '  RowstatCase<32,1>();RowstatCase<32,2>();RowstatCase<32,3>();RowstatCase<32,4>();\n'
                       '  '
                       'RowstatCase<32,6>();RowstatCase<32,8>();RowstatCase<32,12>();RowstatCase<32,16>();\n'
                       '  RowstatCase<32,24>();RowstatCase<32,48>();\n'
                       '}',
                       'void Catalog() {\n'
                       '#define SK(N) '
                       'SplitCase<32,N,2>();SplitCase<32,N,4>();SplitCase<32,N,8>();SplitCase<32,N,16>()\n'
                       '  SK(32);SK(64);SK(128);SK(256);\n'
                       '#undef SK\n'
                       '}']},
 'mma32_math': {'catalogs': ['void Catalog() '
                             '{Mma32Case<32>();Mma32Case<64>();Mma32Case<128>();Mma32Case<256>();}\n',
                             'void Catalog() {}'],
                'direct_primitive': 'dnn1_mma32_math_probe.cuh',
                'include': 'Sm90ForwardMma32.cuh'},
 'dwpw': {'catalogs': ['void Catalog() {\n'
                       '  FusedDwCase<16,1>();FusedDwCase<16,2>();FusedDwCase<32,1>();\n'
                       '  FusedDwCase<32,2>();FusedDwCase<64,1>();FusedDwCase<64,2>();\n'
                       '}',
                       'void Catalog() {}'],
          'direct_primitive': 'dnn1_dwpw_probe.cuh',
          'include': 'Sm90ForwardFused.cuh'},
 'residual_scaled': {'catalogs': ['void Catalog() {\n'
                                  '  '
                                  'EpilogueCase<64,32,Sm90ForwardEpilogueStep<Sm90ForwardOp::kResidual>>("residual_scaled");\n'
                                  '  '
                                  'EpilogueCase<64,64,Sm90ForwardEpilogueStep<Sm90ForwardOp::kResidual>>("residual_scaled");\n'
                                  '  '
                                  'EpilogueCase<64,128,Sm90ForwardEpilogueStep<Sm90ForwardOp::kResidual>>("residual_scaled");\n'
                                  '  '
                                  'EpilogueCase<64,256,Sm90ForwardEpilogueStep<Sm90ForwardOp::kResidual>>("residual_scaled");\n'
                                  '  '
                                  'EpilogueCase<128,32,Sm90ForwardEpilogueStep<Sm90ForwardOp::kResidual>>("residual_scaled");\n'
                                  '  '
                                  'EpilogueCase<128,64,Sm90ForwardEpilogueStep<Sm90ForwardOp::kResidual>>("residual_scaled");\n'
                                  '  '
                                  'EpilogueCase<128,128,Sm90ForwardEpilogueStep<Sm90ForwardOp::kResidual>>("residual_scaled");\n'
                                  '  '
                                  'EpilogueCase<128,256,Sm90ForwardEpilogueStep<Sm90ForwardOp::kResidual>>("residual_scaled");\n'
                                  '}\n',
                                  'void Catalog() {}']},
 'gated_dw': {'catalogs': ['void Catalog() {}',
                           'void Catalog() {\n'
                           '  DepthwiseCase<8,8,64>();DepthwiseCase<8,8,128>();\n'
                           '  DepthwiseCase<8,16,64>();DepthwiseCase<8,16,128>();\n'
                           '  DepthwiseCase<16,16,64>();DepthwiseCase<16,16,128>();\n'
                           '}'],
              'depthwise_primitive': 'dnn1_gated_dw_probe.cuh'},
 'global_pool': {'catalogs': ['void Catalog() {\n'
                              '#define GAP(P,C) SmallCase<Sm90ForwardGlobalPool<1>>("small_global_pool" #P '
                              '"_" #C,"kind=global_pool;parts=" #P ";channels=" #C,P,49,C,C,P*C)\n'
                              '  GAP(1,512);GAP(2,512);GAP(4,512);GAP(7,512);\n'
                              '  GAP(1,1024);GAP(2,1024);GAP(4,1024);GAP(7,1024);\n'
                              '  GAP(1,1280);GAP(2,1280);GAP(4,1280);GAP(7,1280);\n'
                              '  GAP(256,32);GAP(128,64);GAP(64,128);GAP(32,256);GAP(16,512);\n'
                              '#undef GAP\n'
                              '}\n',
                              'void Catalog() {}']},
 'global_pool_unique': {'catalogs': ['void Catalog() {\n'
                                     '#define GAP(P,C) '
                                     'SmallCase<Sm90ForwardGlobalPool<1>>("small_global_pool" #P "_" '
                                     '#C,"kind=global_pool;parts=" #P ";channels=" #C,P,49,C,C,P*C)\n'
                                     '  GAP(3,512);\n'
                                     '  GAP(3,1280);\n'
                                     '  GAP(4,256);\n'
                                     '  GAP(8,256);\n'
                                     '  GAP(16,128);\n'
                                     '  GAP(16,256);\n'
                                     '  GAP(32,128);\n'
                                     '  GAP(64,64);\n'
                                     '  GAP(256,64);\n'
                                     '  GAP(512,32);\n'
                                     '  GAP(1024,32);\n'
                                     '#undef GAP\n'
                                     '}\n',
                                     'void Catalog() {}']},
 'maxpool_flat': {'catalogs': ['void Catalog() {\n'
                               '  '
                               'SmallCase<Sm90ForwardMaxPool<3,2>>("small_pool64","kind=pool;window=3;stride=2;pixels=256;channels=64",32,32,64,64,256*64);\n'
                               '}\n',
                               'void Catalog() {}']}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def catalog_spans(source):
    spans = []
    for match in re.finditer(r'void Catalog\(\) \{', source):
        depth, end = 1, match.end()
        while depth:
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        spans.append((match.start(), end))
    if len(spans) != 2:
        raise ValueError('expected direct and segment catalogs')
    return spans


def generate_source(recipe, base):
    config = RECIPES[recipe]
    if config.get('unchanged'):
        return base
    source = base
    if config.get('add_live16'):
        return source.replace('  DW(8,8);DW(8,16);DW(16,16);\n',
            '  DW(8,8);DW(8,16);DW(16,16);\n'
            '  DepthwiseCase<8,8,64,16,1>();DepthwiseCase<8,8,64,16,2>();\n')
    if 'include' in config:
        source = source.replace('#include <cuda_runtime.h>',
            '#include <tilemega/Backend/'+config['include']+'>\n#include <cuda_runtime.h>')
    if 'depthwise_primitive' in config:
        start = source.index('template<int W,int H,int C,int Live=C,int Stride=1>void DepthwiseCase()')
        end = source.index('template<class Body>', start)
        source = source[:start]+(HERE/config['depthwise_primitive']).read_text()+source[end:]
    for index, (start, end) in reversed(list(enumerate(catalog_spans(source)))):
        body = config['catalogs'][index]
        if index == 0 and 'direct_primitive' in config:
            body = (HERE/config['direct_primitive']).read_text()+body
        source = source[:start]+body+source[end:]
    return source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe', choices=RECIPES, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--base-source', type=Path, default=HERE/'dnn1_constants.cu')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--build', action='store_true')
    p.add_argument('--nvcc', default='/usr/local/cuda/bin/nvcc')
    p.add_argument('--timeout-s', type=int, default=5400)
    a = p.parse_args()
    if not (a.source_root/'include/tilemega/Backend/Sm90ForwardEpilogue.cuh').is_file():
        raise ValueError('source-root lacks production headers')
    a.out.mkdir(parents=True, exist_ok=False)
    output = a.out/'dnn1_constants.cu'
    output.write_text(generate_source(a.recipe, a.base_source.read_text()))
    receipt = dict(status='CPU source generation; not calibration', recipe=a.recipe,
        base_sha256=sha(a.base_source), generated_sha256=sha(output),
        generator_sha256=sha(__file__), source_root=str(a.source_root.resolve()),
        scope='Run the existing guarded five-fresh-process workflow on GPUs0/5/7. '
              'Production header identity is recorded at build; source equivalence alone is not binary equivalence.')
    (a.out/'generation.json').write_text(json.dumps(receipt, indent=2)+'\n')
    if a.build:
        sys.path.insert(0, str(a.base_source.parent))
        import dnn1_constants as c
        c.SOURCE = output
        for header in ('Sm90ForwardMma32.cuh', 'Sm90ForwardFused.cuh'):
            if (a.source_root/'include/tilemega/Backend'/header).is_file():
                c.FORWARD_HEADERS += ('Backend/'+header,)
        c.build(argparse.Namespace(out=a.out/'build', source_root=a.source_root,
            implementation='fixed', nvcc=a.nvcc, timeout_s=a.timeout_s))


if __name__ == '__main__':
    main()
