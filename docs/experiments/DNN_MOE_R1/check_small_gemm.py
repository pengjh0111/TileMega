#!/usr/bin/env python3
"""Check DM small-tile mainloops before exposing them to solver candidates."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

from identity_dm import resources, sha

LOCK = '/root/r14_work/gpu.lock'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.root / 'test/unit/dm_small_gemm_test.cu'
    nvcc = '/usr/local/cuda/bin/nvcc'
    common = [nvcc, '-std=c++17', '-O2', '--expt-relaxed-constexpr',
              '-I' + str(args.root / 'include'),
              '-I' + str(args.root / 'third_party/cutlass/include'),
              '-I' + str(args.root / 'third_party/cutlass/tools/util/include')]
    listing = subprocess.check_output(['flock', LOCK, *common, '-arch=sm_89',
        '-DTILEMEGA_ARCH_ID=890', '-M', str(source)], text=True)
    dependencies = {Path(__file__).resolve(), Path(__file__).with_name('identity_dm.py').resolve()}
    for word in shlex.split(listing.replace('\\\n', ' '))[1:]:
        path = Path(word).resolve()
        if path.is_file():
            dependencies.add(path)
    hashes = {str(p): sha(p) for p in sorted(dependencies)}
    configurations = [(m, n, k, 2) for m in (16, 32, 64, 128)
        for n in (16, 32, 64, 128, 256) for k in (16, 32, 64, 128)
        if m * n <= 16384 and (n == 16 or k < 64)]
    configurations += [(m, 16, k, 3) for m in (16, 32, 64, 128) for k in (16, 32)]
    architecture_cases = {(m, 16, k, 2) for m in (16, 32, 64, 128) for k in (16, 32, 64, 128)}
    architecture_cases |= {(16, 32, 16, 2), (16, 64, 32, 2)}
    result = dict(evidence='verified', passed=False, inputs=hashes, builds=[],
        fresh_processes=[], configurations=configurations,
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.root, text=True).strip(),
        diff_sha256=hashlib.sha256(subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=args.root)).hexdigest(),
        compiler_version=subprocess.check_output([nvcc, '--version'], text=True), compiler_sha256=sha(nvcc),
        scope='new TN16/TK16/TK32 dense/tiled-B mainloops only; independent FP32 oracle, tails, split offsets, canaries and bitwise row/tiled reads; paged ring, convolution and expert operand policies and solver integration pending')
    try:
        for m, n, k, stages in configurations:
            for arch in ((80, 89, 90, 100, 120) if (m, n, k, stages) in architecture_cases else (89,)):
                binary = args.out / f'm{m}-n{n}-k{k}-s{stages}-sm_{arch}'
                log = binary.with_suffix('.log')
                macros = [f'-DTILEMEGA_ARCH_ID={arch*10}', f'-DDM_TEST_TM={m}',
                          f'-DDM_TEST_TN={n}', f'-DDM_TEST_TK={k}', f'-DDM_TEST_STAGES={stages}']
                command = ['flock', LOCK, *common, f'-arch=sm_{arch}', *macros,
                           '-Xptxas=-v', str(source), '-o', str(binary)]
                with log.open('w') as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
                kernel_resources = resources(log.read_text())
                macro_path = binary.with_suffix('.macros.txt')
                subprocess.run(['flock', LOCK, *common, f'-arch=sm_{arch}', *macros,
                    '-E', '-Xcompiler=-dM', str(source), '-o', str(macro_path)], check=True)
                definitions = dict(re.findall(r'^#define\s+(\S+)(?:[ \t]+(.*))?$', macro_path.read_text(), re.M))
                identity = dict(schema='tilemega.dm1.native-test.identity.v1',
                    source={key: result[key] for key in ('head', 'diff_sha256', 'inputs')},
                    cu_sha256=sha(source), binary_sha256=sha(binary),
                    nvcc_version=result['compiler_version'], compiler_sha256=result['compiler_sha256'],
                    compiler_command=command[2:], macros=definitions, macros_sha256=sha(macro_path),
                    arch=f'sm_{arch}', execution=dict(pg=['l2', 'tiled'], phase='body_unit'),
                    implementations=dict(gemm='ServingDmGemm', tile_m=m, tile_n=n, tile_k=k,
                                         stages=stages, k_warp_splits=2 if m == n == 16 else 1),
                    kernels=kernel_resources, spill=any(row['spill'] for row in kernel_resources.values()))
                identity['artifact_id'] = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                Path(str(binary) + '.identity.json').write_text(json.dumps(identity, indent=2) + '\n')
                result['builds'].append(dict(tile_m=m, tile_n=n, tile_k=k, stages=stages,
                    arch=arch, command=command, binary_sha256=sha(binary), log_sha256=sha(log),
                    artifact_id=identity['artifact_id'], kernels=kernel_resources, spill=identity['spill']))
        for m, n, k, stages in configurations:
            binary = args.out / f'm{m}-n{n}-k{k}-s{stages}-sm_89'
            required = 50 if (m, n, stages) == (16, 16, 2) else 1
            for process in range(required):
                log = args.out / f'process-m{m}-n{n}-k{k}-s{stages}-{process:02d}.log'
                with log.open('w') as stream:
                    status = subprocess.run(['flock', LOCK, str(binary)], stdout=stream,
                        stderr=subprocess.STDOUT, timeout=7200).returncode
                result['fresh_processes'].append(dict(tile_m=m, tile_n=n, tile_k=k, stages=stages,
                    process=process, exit_code=status, log_sha256=sha(log)))
                if status:
                    raise RuntimeError(f'small GEMM failed; see {log}')
        if hashes != {str(p): sha(p) for p in sorted(dependencies)}:
            raise RuntimeError('small GEMM inputs changed during validation')
        result['passed'] = True
    finally:
        result['pass_rate'] = {}
        for m, n, k, stages in configurations:
            rows = [r for r in result['fresh_processes'] if
                    (r['tile_m'], r['tile_n'], r['tile_k'], r['stages']) == (m, n, k, stages)]
            result['pass_rate'][f'{m}x{n}x{k}_s{stages}'] = dict(
                passes=sum(r['exit_code'] == 0 for r in rows), attempts=len(rows),
                required=50 if (m, n, stages) == (16, 16, 2) else 1)
        (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print('DM small GEMM families and fresh-process K-warp reductions passed', flush=True)


if __name__ == '__main__':
    main()
