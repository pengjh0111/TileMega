#!/usr/bin/env python3
"""Verify sparse table and counted waits through the stage materializer."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from identity_dm import resources, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    nvcc = '/usr/local/cuda/bin/nvcc'
    names = ['test/unit/dm_stage_dependencies_test.cu',
             'lib/Target/TargetSpec.cpp', 'lib/Support/Json.cpp',
             'lib/Codegen/RuntimeTaskGraph.cpp', 'lib/Solver/PlanMaterialize.cpp',
             'lib/Dialect/CouplingGraph/PlacementPlan.cpp',
             'lib/Solver/BalancedPlacement.cpp', 'lib/Solver/ListScheduler.cpp']
    common = [nvcc, '-std=c++17', '-O2', '--expt-relaxed-constexpr',
              '-I' + str(args.root / 'include'),
              '-I' + str(args.root / 'third_party/cutlass/include'),
              '-I' + str(args.root / 'third_party/cutlass/tools/util/include')]
    dependencies = {Path(__file__).resolve()}
    for source in names:
        command = common + ['-arch=sm_80', '-DTILEMEGA_ARCH_ID=800', '-x', 'cu',
                            '-M', str(args.root / source)]
        listing = subprocess.check_output(command, text=True)
        for word in shlex.split(listing.replace('\\\n', ' '))[1:]:
            path = Path(word).resolve()
            if path.is_file(): dependencies.add(path)
    hashes = {str(path): sha(path) for path in sorted(dependencies)}
    result = dict(evidence='verified', passed=False, inputs=hashes, builds=[],
        execution_architecture=89, fresh_processes=[], support_builds=[],
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.root, text=True).strip(),
        diff_sha256=hashlib.sha256(subprocess.check_output(
            ['git', 'diff', '--binary', 'HEAD'], cwd=args.root)).hexdigest(),
        compiler_version=subprocess.check_output([nvcc, '--version'], text=True),
        compiler_sha256=sha(nvcc),
        scope='synthetic task bodies using the full table queue materializer, counted stage waits and isolated L1/L2 counter banks; 16 alternating epochs; kappa=1/4/16; real TaskBody and page gates pending')
    try:
        objects = []
        for source in names[1:]:
            obj = args.out / (Path(source).stem + '.o')
            log = args.out / (Path(source).stem + '.log')
            command = common + ['-arch=sm_80', '-DTILEMEGA_ARCH_ID=800',
                '-DTILEMEGA_DM_SUPPORT=1', '-x', 'cu', '-c', str(args.root / source), '-o', str(obj)]
            with log.open('w') as stream:
                subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
            objects.append(str(obj))
            result['support_builds'].append(dict(command=command, object_sha256=sha(obj), log_sha256=sha(log)))
        for kappa in (1, 4, 16):
            for arch in ((80, 89, 90, 100, 120) if kappa == 1 else (89,)):
                binary = args.out / f'stage-k{kappa}-sm_{arch}'
                command = common + [f'-arch=sm_{arch}', f'-DTILEMEGA_ARCH_ID={arch * 10}',
                    f'-DTILEMEGA_EVENT_KAPPA={kappa}', '-Xptxas=-v',
                    str(args.root / names[0]), *objects, '-o', str(binary)]
                log = args.out / f'stage-k{kappa}-sm_{arch}.log'
                with log.open('w') as stream:
                    subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
                kernels = resources(log.read_text())
                result['builds'].append(dict(arch=arch, kappa=kappa, command=command,
                    binary_sha256=sha(binary), kernels=kernels,
                    spill=any(row['spill'] for row in kernels.values())))
            for process in range(50):
                log = args.out / f'process-k{kappa}-{process:02d}.log'
                with log.open('w') as stream:
                    status = subprocess.run([str(args.out / f'stage-k{kappa}-sm_89')],
                        stdout=stream, stderr=subprocess.STDOUT, timeout=60).returncode
                result['fresh_processes'].append(dict(kappa=kappa, process=process, exit_code=status,
                                                      log_sha256=sha(log)))
                if status: raise RuntimeError(f'process {process}, kappa {kappa} failed; see {log}')
        if hashes != {str(path): sha(path) for path in sorted(dependencies)}:
            raise RuntimeError('stage dependency transitive inputs changed during validation')
        result['passed'] = True
    finally:
        passes = sum(row['exit_code'] == 0 for row in result['fresh_processes'])
        result['pass_rate'] = dict(passes=passes, attempts=len(result['fresh_processes']), required=150)
        (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Stage dependencies: five targets compiled; 150/150 fresh processes passed (50 per kappa)', flush=True)


if __name__ == '__main__':
    main()
