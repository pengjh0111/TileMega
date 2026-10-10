#!/usr/bin/env python3
"""Numerically check an imported CG's generated forward library."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from identity_dm import resources, sha, source_snapshot


def run(binary, output, token_seq):
    import torch
    from tilemega.serving.plan import FORWARD, PlanLibrary
    torch.manual_seed(20261007)
    torch.backends.cuda.matmul.allow_tf32 = False
    library = PlanLibrary(binary)
    assert library.info.phase == FORWARD and library.info.seq == (token_seq or 1)
    assert library.info.capacity == library.info.past_lo == library.info.past_hi == 0
    assert {buffer.name for buffer in library.buffers} == {'x', 'weight', 'out'}
    records = []
    for batch in ((1,) if token_seq else (1, 8)):
        rows = token_seq or batch * 17
        x = torch.randn(rows, 128, device='cuda', dtype=torch.bfloat16)
        weight = torch.randn(64, 128, device='cuda', dtype=torch.bfloat16)
        actual = torch.empty(rows, 64, device='cuda', dtype=torch.bfloat16)
        reference = torch.nn.functional.linear(x.float(), weight.float())
        pointers = {name: value.data_ptr() for name, value in
                    (('x', x), ('weight', weight), ('out', actual))}
        with library.create(batch, pointers, 0) as plan:
            plan.set_steps([0])
            first = None
            for mode in (1, 2):
                plan.launch(0, mode, torch.cuda.current_stream().cuda_stream)
                torch.cuda.synchronize()
                error = (actual.float() - reference).abs()
                assert torch.all(error <= 1.6e-2 + 1.6e-2 * reference.abs())
                if first is None:
                    first = actual.clone()
                else:
                    assert torch.equal(first, actual)
                records.append(dict(batch=batch, mode=mode, passed=True,
                                    max_absolute_error=error.max().item()))
    output.write_text(json.dumps(dict(evidence='verified', passed=True,
        scope='CG-generated dense forward GEMM, numerical and L1/L2 equality; no model gate',
        token_seq=token_seq,
        cases=records), indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--token-seq', type=int, choices=(1, 16, 128, 4096), default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.run:
        run(args.run, args.out / 'result.json', args.token_seq)
        return
    compiler = args.root / 'build-dm/tools/tilemega'
    source = source_snapshot(args.root, compiler)
    cu = args.out / 'forward.cu'
    binary = args.out / 'forward.so'
    subprocess.run([str(args.root / 'build-dm/tilemega-unit'), 'forward_frontend',
                    *(['--emit-token', str(cu), str(args.token_seq)] if args.token_seq else
                      ['--emit', str(cu)])], check=True)
    support = ['lib/Target/TargetSpec.cpp', 'lib/Support/Json.cpp',
               'lib/Codegen/RuntimeTaskGraph.cpp', 'lib/Solver/PlanMaterialize.cpp',
               'lib/Dialect/CouplingGraph/PlacementPlan.cpp',
               'lib/Solver/BalancedPlacement.cpp', 'lib/Solver/ListScheduler.cpp']
    command = ['/usr/local/cuda/bin/nvcc', '-std=c++17', '-O2',
               '--expt-relaxed-constexpr', '-arch=sm_89', '-DTILEMEGA_ARCH_ID=890',
               '-DTILEMEGA_SERVING_BATCH_LO=1',
               f'-DTILEMEGA_SERVING_BATCH_HI={1 if args.token_seq else 8}',
               '-DTILEMEGA_SERVING_PAST_LO=0', '-DTILEMEGA_SERVING_PAST_HI=0',
               '-DTILEMEGA_MIDPOINT_REFINE=0', '-shared', '-Xcompiler=-fPIC',
               '-cudart', 'shared', '-Xptxas=-v', '-x', 'cu',
               '-I' + str(args.root / 'include'),
               '-I' + str(args.root / 'third_party/cutlass/include'),
               '-I' + str(args.root / 'third_party/cutlass/tools/util/include'),
               str(cu), *(str(args.root / name) for name in support), '-o', str(binary)]
    log = args.out / 'forward.ptxas.log'
    with log.open('w') as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    kernels = resources(log.read_text())
    identity = dict(schema='tilemega.dm1.test_identity.v1', source=source,
        cu_sha256=sha(cu), so_sha256=sha(binary), compiler_command=command,
        nvcc_version=subprocess.check_output([command[0], '--version'], text=True),
        macros=dict(re.findall(r'^#define\s+(\w+)\s+([^\n]+)', cu.read_text(), re.M)),
        architecture='sm_89', implementations=['ServingGemmTaskBody::RunDm'],
        execution=dict(phase='forward', modes=['L1', 'L2']), kernels=kernels,
        spill=any(row['spill'] for row in kernels.values()),
        test_source_sha256=sha(args.root / 'test/unit/forward_frontend_test.cpp'))
    for flag in command:
        if flag.startswith('-D'):
            name, _, value = flag[2:].partition('=')
            identity['macros'][name] = value or '1'
    Path(str(binary) + '.identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    subprocess.run([sys.executable, __file__, '--root', str(args.root),
                    '--out', str(args.out), '--run', str(binary),
                    *(['--token-seq', str(args.token_seq)] if args.token_seq else [])], check=True)
    if source['source_files'] != source_snapshot(args.root, compiler)['source_files']:
        raise RuntimeError('generated forward inputs changed during validation')
    print('Generated forward: numerical and L1/L2 equality passed', flush=True)


if __name__ == '__main__':
    main()
