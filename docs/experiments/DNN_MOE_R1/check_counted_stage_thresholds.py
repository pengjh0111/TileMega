#!/usr/bin/env python3
"""Verify native counted threshold upload, waits, probes and counter banks."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

from identity_dm import resources, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    nvcc = '/usr/local/cuda/bin/nvcc'
    names = ['test/unit/dm_counted_stage_threshold_test.cu',
             'lib/Target/TargetSpec.cpp', 'lib/Support/Json.cpp',
             'lib/Codegen/RuntimeTaskGraph.cpp', 'lib/Solver/PlanMaterialize.cpp',
             'lib/Dialect/CouplingGraph/PlacementPlan.cpp',
             'lib/Solver/BalancedPlacement.cpp', 'lib/Solver/ListScheduler.cpp']
    common = [nvcc, '-std=c++17', '-O2', '--expt-relaxed-constexpr',
              '-I' + str(args.root / 'include'),
              '-I' + str(args.root / 'third_party/cutlass/include'),
              '-I' + str(args.root / 'third_party/cutlass/tools/util/include')]
    dependencies = {Path(__file__).resolve(),Path(__file__).with_name('identity_dm.py').resolve()}
    for source in names:
        command = common + ['-arch=sm_80', '-DTILEMEGA_ARCH_ID=800', '-x', 'cu',
                            '-M', str(args.root / source)]
        listing = subprocess.check_output(['flock','/root/r14_work/gpu.lock',*command], text=True)
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
        baselines={'vllm':'0.30.0'},
        scope='native model creation uploads [8,8,2] per-consumer thresholds and I2 ordering tables; waits and slot probes consume immutable thresholds, counter offsets and separate L1/L2 banks; synthetic bodies, 16 alternating epochs; real MoE and paged body gates pending')
    try:
        objects = []
        for source in names[1:]:
            obj = args.out / (Path(source).stem + '.o')
            log = args.out / (Path(source).stem + '.log')
            command = common + ['-arch=sm_80', '-DTILEMEGA_ARCH_ID=800',
                '-DTILEMEGA_DM_SUPPORT=1', '-x', 'cu', '-c', str(args.root / source), '-o', str(obj)]
            with log.open('w') as stream:
                subprocess.run(['flock','/root/r14_work/gpu.lock',*command], check=True, stdout=stream, stderr=subprocess.STDOUT)
            objects.append(str(obj))
            result['support_builds'].append(dict(command=command, object_sha256=sha(obj), log_sha256=sha(log)))
        for kappa in ((1,) if args.smoke else (1,4,16)):
            for arch in ((80,89,90,100,120) if kappa==1 and not args.smoke else (89,)):
                binary = args.out / f'stage-k{kappa}-sm_{arch}'
                command = common + [f'-arch=sm_{arch}', f'-DTILEMEGA_ARCH_ID={arch * 10}',
                    f'-DTILEMEGA_EVENT_KAPPA={kappa}', '-Xptxas=-v',
                    str(args.root / names[0]), *objects, '-o', str(binary)]
                log = args.out / f'stage-k{kappa}-sm_{arch}.log'
                with log.open('w') as stream:
                    subprocess.run(['flock','/root/r14_work/gpu.lock',*command], check=True, stdout=stream, stderr=subprocess.STDOUT)
                kernels = resources(log.read_text())
                macros=binary.with_suffix('.macros.txt')
                subprocess.run(['flock','/root/r14_work/gpu.lock',*common,f'-arch=sm_{arch}',
                    f'-DTILEMEGA_ARCH_ID={arch*10}',f'-DTILEMEGA_EVENT_KAPPA={kappa}',
                    '-E','-Xcompiler=-dM',str(args.root/names[0]),'-o',str(macros)],check=True)
                identity=dict(schema='tilemega.dm1.native-test.identity.v1',
                    source={key:result[key] for key in ('head','diff_sha256','inputs')},
                    cu_sha256=sha(args.root/names[0]),binary_sha256=sha(binary),
                    compiler_command=command,compiler_version=result['compiler_version'],
                    compiler_sha256=result['compiler_sha256'],
                    macros=dict(re.findall(r'^#define\s+(\S+)(?:[ \t]+(.*))?$',macros.read_text(),re.M)),
                    arch=f'sm_{arch}',baselines=result['baselines'],
                    execution=dict(pg='l2',phase='forward',executors=['L1','L2'],kappa=kappa),
                    implementations=['Create','WaitDmCountedDependency','ProbeTaskDependencies'],
                    kernels=kernels,spill=any(row['spill'] for row in kernels.values()))
                identity['artifact_id']=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                Path(str(binary)+'.identity.json').write_text(json.dumps(identity,indent=2)+'\n')
                result['builds'].append(dict(arch=arch, kappa=kappa, command=command,
                    binary_sha256=sha(binary), artifact_id=identity['artifact_id'], log_sha256=sha(log), kernels=kernels,
                    spill=any(row['spill'] for row in kernels.values())))
            for process in range(1 if args.smoke else 50):
                log = args.out / f'process-k{kappa}-{process:02d}.log'
                with log.open('w') as stream:
                    status = subprocess.run(['flock','/root/r14_work/gpu.lock',str(args.out / f'stage-k{kappa}-sm_89')],
                        stdout=stream, stderr=subprocess.STDOUT, timeout=60).returncode
                result['fresh_processes'].append(dict(kappa=kappa, process=process, exit_code=status,
                                                      log_sha256=sha(log)))
                if status: raise RuntimeError(f'process {process}, kappa {kappa} failed; see {log}')
        if hashes != {str(path): sha(path) for path in sorted(dependencies)}:
            raise RuntimeError('stage dependency transitive inputs changed during validation')
        result['passed'] = True
    finally:
        passes = sum(row['exit_code'] == 0 for row in result['fresh_processes'])
        result['pass_rate'] = dict(passes=passes, attempts=len(result['fresh_processes']), required=1 if args.smoke else 150)
        (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Native counted thresholds: PASS',flush=True)


if __name__ == '__main__':
    main()
