#!/usr/bin/env python3
"""Check dense/page DM TaskBodies in bounded compilation bundles."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

from identity_dm import resources, sha

LOCK = '/root/r14_work/gpu.lock'


def source_bundle(path, fixture, shapes):
    code = '''#define TILEMEGA_DM_SUPPORT 1
#include <tilemega/Backend/ServingDmGemm.h>
#include <tilemega/Codegen/tasks/PagedGemmTaskBody.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
'''
    for i, (m, n, k) in enumerate(shapes):
        code += (f'#undef DM_TEST_TM\n#undef DM_TEST_TN\n#undef DM_TEST_TK\n'
                 f'#define DM_TEST_TM {m}\n#define DM_TEST_TN {n}\n#define DM_TEST_TK {k}\n'
                 f'namespace fixture{i} {{\n#include "{fixture}"\n}}\n')
    code += 'int main(){' + ''.join(f'if(fixture{i}::main())return 3;' for i in range(len(shapes)))
    path.write_text(code + 'return 0;}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--overlay', type=Path)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fixture = args.fixture or args.root/'test/unit/dm_small_task_gemm_test.cu'
    nvcc = '/usr/local/cuda/bin/nvcc'
    common = [nvcc, '-std=c++17', '-O2', '--expt-relaxed-constexpr']
    if args.overlay:
        common += ['-I'+str(args.overlay/'include')]
    common += ['-I'+str(args.root/'include'),
               '-I'+str(args.root/'third_party/cutlass/include'),
               '-I'+str(args.root/'third_party/cutlass/tools/util/include')]
    # Each narrow-N warp layout compiles on every target. Bundling reduces
    # repeated header parsing; every geometry still has its own CUDA entries.
    critical = [[(m, 16, k) for k in (16, 32, 64, 128)] for m in (16, 32, 64, 128)]
    critical += [[(16, 32, 16), (16, 64, 32)]]
    covered = {shape for group in critical for shape in group}
    others = [(m, n, k) for m in (16, 32, 64, 128)
              for n in (16, 32, 64, 128, 256) for k in (16, 32, 64, 128)
              if m*n <= 16384 and (n == 16 or k < 64) and (m, n, k) not in covered]
    groups = critical + [others[i:i+4] for i in range(0, len(others), 4)]
    if args.smoke:
        groups = groups[:1]
    sources = []
    dependencies = {Path(__file__).resolve(), Path(__file__).with_name('identity_dm.py').resolve()}
    for i, shapes in enumerate(groups):
        source = args.out/f'bundle{i}.cu'
        source_bundle(source, fixture.resolve(), shapes)
        sources.append(source)
        listing = subprocess.check_output(['flock', LOCK, *common, '-arch=sm_89',
            '-DTILEMEGA_ARCH_ID=890', '-DTILEMEGA_NONPAGED_TILED=1',
            '-DTILEMEGA_WEIGHT_LAYOUT_TILED=1', '-M', str(source)], text=True)
        for word in shlex.split(listing.replace('\\\n', ' '))[1:]:
            path = Path(word).resolve()
            if path.is_file():
                dependencies.add(path)
    hashes = {str(p): sha(p) for p in sorted(dependencies)}
    result = dict(evidence='verified', passed=False, inputs=hashes, groups=groups,
        builds=[], fresh_processes=[],
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.root, text=True).strip(),
        diff_sha256=hashlib.sha256(subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=args.root)).hexdigest(),
        compiler_version=subprocess.check_output([nvcc, '--version'], text=True), compiler_sha256=sha(nvcc),
        baselines=dict(vllm_version=subprocess.check_output(['/root/venv_vllm/bin/python', '-c',
            'from importlib.metadata import version; print(version("vllm"))'], text=True).strip()),
        scope='46 additional finite-chain dense/page TaskBody geometries; row/tiled weights, independent FP32 oracle, page wrap, tails, offsets, canaries and BF16 bitwise nonpaged/paged; im2col/rowgather/expert policies, complete chains and model gates pending')
    try:
        for i, source in enumerate(sources):
            for tiled in (0, 1):
                for arch in ((80, 89, 90, 100, 120) if i < len(critical) and not args.smoke else (89,)):
                    binary = args.out/f'bundle{i}-tiled{tiled}-sm_{arch}'
                    log = binary.with_suffix('.log')
                    defines = [f'-DTILEMEGA_ARCH_ID={arch*10}', f'-DTILEMEGA_NONPAGED_TILED={tiled}',
                               f'-DTILEMEGA_WEIGHT_LAYOUT_TILED={tiled}']
                    command = ['flock', LOCK, *common, f'-arch=sm_{arch}', *defines,
                               '-Xptxas=-v', str(source), '-o', str(binary)]
                    with log.open('w') as stream:
                        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
                    kernels = resources(log.read_text())
                    macros = binary.with_suffix('.macros.txt')
                    subprocess.run(['flock', LOCK, *common, f'-arch=sm_{arch}', *defines,
                        '-E', '-Xcompiler=-dM', str(source), '-o', str(macros)], check=True)
                    identity = dict(schema='tilemega.dm1.native-test.identity.v1',
                        source={key: result[key] for key in ('head', 'diff_sha256', 'inputs')},
                        cu_sha256=sha(source), binary_sha256=sha(binary),
                        nvcc_version=result['compiler_version'], compiler_sha256=result['compiler_sha256'],
                        compiler_command=command[2:],
                        macros=dict(re.findall(r'^#define\s+(\S+)(?:[ \t]+(.*))?$', macros.read_text(), re.M)),
                        macros_sha256=sha(macros), arch=f'sm_{arch}', baselines=result['baselines'],
                        execution=dict(pg=['l2', 'pages'], phase='body_unit', tiled_weights=bool(tiled)),
                        implementations=dict(gemm='ServingGemmTaskBody/ServingDmGemm',
                            paged_gemm='PagedGemmTaskBody', geometries=groups[i], stages=2, pages=2),
                        kernels=kernels, spill=any(row['spill'] for row in kernels.values()))
                    identity['artifact_id'] = hashlib.sha256(json.dumps(identity,
                        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                    Path(str(binary)+'.identity.json').write_text(json.dumps(identity, indent=2)+'\n')
                    result['builds'].append(dict(bundle=i, tiled=tiled, arch=arch, command=command,
                        binary_sha256=sha(binary), log_sha256=sha(log), artifact_id=identity['artifact_id'],
                        kernels=kernels, spill=identity['spill']))
        for tiled in (0, 1):
            for i in range(len(groups)):
                for process in range(50 if i == 0 and not args.smoke else 1):
                    binary = args.out/f'bundle{i}-tiled{tiled}-sm_89'
                    log = args.out/f'process-bundle{i}-tiled{tiled}-{process:02d}.log'
                    with log.open('w') as stream:
                        status = subprocess.run(['flock', LOCK, str(binary)], stdout=stream,
                            stderr=subprocess.STDOUT, timeout=7200).returncode
                    result['fresh_processes'].append(dict(bundle=i, tiled=tiled, process=process,
                        exit_code=status, log_sha256=sha(log)))
                    if status:
                        raise RuntimeError(f'small TaskBody GEMM failed; see {log}')
        if hashes != {str(p): sha(p) for p in sorted(dependencies)}:
            raise RuntimeError('small TaskBody GEMM inputs changed during validation')
        result['passed'] = True
    finally:
        result['pass_rate'] = {}
        for tiled in (0, 1):
            for i in range(len(groups)):
                rows = [r for r in result['fresh_processes'] if (r['bundle'], r['tiled']) == (i, tiled)]
                result['pass_rate'][f'bundle{i}-tiled{tiled}'] = dict(
                    passes=sum(r['exit_code'] == 0 for r in rows), attempts=len(rows), required=50 if i == 0 and not args.smoke else 1)
        (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print('DM small TaskBodies: dense/page numerical and fresh-process checks passed', flush=True)


if __name__ == '__main__':
    main()
