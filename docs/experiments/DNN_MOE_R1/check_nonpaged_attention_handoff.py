#!/usr/bin/env python3
"""Verify native nonpaged attention last-arriver and elided-stage execution."""
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
    parser.add_argument('--overlay', type=Path,
                        help='isolated headers; recorded in transitive input hashes')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    nvcc = '/usr/local/cuda/bin/nvcc'
    sources = ['test/unit/dm_nonpaged_attention_handoff_test.cu', 'lib/Target/TargetSpec.cpp',
        'lib/Support/Json.cpp', 'lib/Codegen/RuntimeTaskGraph.cpp',
        'lib/Solver/PlanMaterialize.cpp', 'lib/Dialect/CouplingGraph/PlacementPlan.cpp',
        'lib/Solver/BalancedPlacement.cpp', 'lib/Solver/ListScheduler.cpp']
    common = [nvcc, '-std=c++17', '-O2', '--expt-relaxed-constexpr',
              '-I' + str(args.root/'include'),
              '-I' + str(args.root/'third_party/cutlass/include'),
              '-I' + str(args.root/'third_party/cutlass/tools/util/include')]
    if args.overlay:
        common.insert(4,'-I'+str(args.overlay/'include'))
    dependencies = {Path(__file__).resolve(), Path(__file__).with_name('identity_dm.py').resolve(),
        ((args.overlay or args.root)/'include/tilemega/Codegen/executor/ServingPrefetch.cuh').resolve()}
    for source in sources:
        listing = subprocess.check_output(['flock', LOCK, *common, '-arch=sm_80',
            '-DTILEMEGA_ARCH_ID=800', '-x', 'cu', '-M', str(args.root/source)], text=True)
        for word in shlex.split(listing.replace('\\\n',' '))[1:]:
            path=Path(word).resolve()
            if path.is_file(): dependencies.add(path)
    hashes={str(path):sha(path) for path in sorted(dependencies)}
    result=dict(evidence='verified',passed=False,inputs=hashes,builds=[],support_builds=[],
        fresh_processes=[],head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.root,text=True).strip(),
        diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=args.root)).hexdigest(),
        compiler_version=subprocess.check_output([nvcc,'--version'],text=True),compiler_sha256=sha(nvcc),
        scope='native C ABI decode attention B=3 G=2 QPerKV=4 D=64 capacity=137 block=64; 16 poisoned epochs, independent FP32 reference, group-major stage/LA and L1/L2 bitwise, epoch tickets and elided barriers; no model performance')
    cases=(0,) if args.smoke else (0,1)
    result['baselines']=dict(vllm_version=subprocess.check_output(
        ['/root/venv_vllm/bin/python','-c','from importlib.metadata import version; print(version("vllm"))'],text=True).strip())
    try:
        objects=[]
        for source in sources[1:]:
            obj=args.out/(Path(source).stem+'.o');log=obj.with_suffix('.log')
            command=['flock',LOCK,*common,'-arch=sm_80','-DTILEMEGA_ARCH_ID=800',
                '-DTILEMEGA_DM_SUPPORT=1','-x','cu','-c',str(args.root/source),'-o',str(obj)]
            with log.open('w') as stream:subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
            objects.append(str(obj))
            result['support_builds'].append(dict(command=command,object_sha256=sha(obj),log_sha256=sha(log)))
        for phase in cases:
            for tile in ((1,) if args.smoke else (1,4,16,0)):
                for arch in ((89,) if args.smoke or tile!=1 else (80,89,90,100,120)):
                    binary=args.out/f'sync{phase}-kappa{tile}-sm_{arch}';log=binary.with_suffix('.log')
                    configuration=[f'-DTILEMEGA_SYNC_V3={phase}', f'-DTILEMEGA_EVENT_KAPPA={tile}',
                                   '-DTILEMEGA_EVENT_SOLO=1', '-DDM_TEST_PHASE=1', '-DTILEMEGA_L2_PREFETCH=1',
                                   '-DTILEMEGA_L2_PREFETCH_DEPTH=2', '-DTILEMEGA_L2_PREFETCH_BYTES=16384',
                                   '-DTILEMEGA_L2_PREFETCH_STRIDE=128']
                    command=['flock',LOCK,*common,f'-arch=sm_{arch}',f'-DTILEMEGA_ARCH_ID={arch*10}',
                        *configuration,'-Xptxas=-v',
                        str(args.root/sources[0]),*objects,'-o',str(binary)]
                    with log.open('w') as stream:subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
                    kernels=resources(log.read_text())
                    result['builds'].append(dict(sync_v3=phase,kappa=tile,arch=arch,command=command,
                        binary_sha256=sha(binary),log_sha256=sha(log),kernels=kernels,
                        spill=any(row['spill'] for row in kernels.values())))
                    macros=binary.with_suffix('.macros.txt')
                    subprocess.run(['flock',LOCK,*common,f'-arch=sm_{arch}',
                        f'-DTILEMEGA_ARCH_ID={arch*10}',*configuration,'-E','-Xcompiler=-dM',
                        str(args.root/sources[0]),'-o',str(macros)],check=True)
                    definitions=dict(re.findall(r'^#define\s+(\S+)(?:[ \t]+(.*))?$',macros.read_text(),re.M))
                    identity=dict(schema='tilemega.dm1.native-test.identity.v1',
                        source={k:result[k] for k in ('head','diff_sha256','inputs')},
                        cu_sha256=sha(args.root/sources[0]),binary_sha256=sha(binary),
                        nvcc_version=result['compiler_version'],compiler_sha256=result['compiler_sha256'],
                        compiler_command=command[2:],macros=definitions,macros_sha256=sha(macros),arch=f'sm_{arch}',
                        baselines=result['baselines'],
                        execution=dict(pg='l2',executor=['L1','L2'],loop=[],phase=1,sync_v3=phase,kappa=tile),
                        implementations=dict(gemm='ServingGemmTaskBody',
                            fixture=sources[0],tile_m=16,tile_n=64,tile_k=64,attention='IndependentAttentionTaskBody/AttentionMergeTaskBody', qperkv=4, head_dim=64, batch=3, groups=2, capacity=137, kv_block=64, handoff='EpochLastArriver/RunTask'),
                        placement=dict(prefetch_depth=2,prefetch_bytes=16384),
                        kernels=kernels,spill=any(row['spill'] for row in kernels.values()))
                    identity['artifact_id']=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                    result['builds'][-1]['artifact_id']=identity['artifact_id']
                    Path(str(binary)+'.identity.json').write_text(json.dumps(identity,indent=2)+'\n')
            for process in range(1 if args.smoke else 50):
                tile=1 if args.smoke else (1,4,16,0)[process%4]
                binary=args.out/f'sync{phase}-kappa{tile}-sm_89';log=args.out/f'process-sync{phase}-{process:02d}.log'
                with log.open('w') as stream:
                    status=subprocess.run(['flock',LOCK,str(binary)],stdout=stream,stderr=subprocess.STDOUT,timeout=7200).returncode
                result['fresh_processes'].append(dict(phase=phase,kappa=tile,process=process,exit_code=status,log_sha256=sha(log)))
                if status:raise RuntimeError(f'nonpaged attention handoff {phase}, process {process} failed; see {log}')
        if hashes!={str(path):sha(path) for path in sorted(dependencies)}:
            raise RuntimeError('nonpaged attention handoff inputs changed during validation')
        result['passed']=True
    finally:
        result['pass_rate']={str(phase):dict(passes=sum(row['exit_code']==0 for row in result['fresh_processes'] if row['phase']==phase),
            attempts=sum(row['phase']==phase for row in result['fresh_processes']),required=1 if args.smoke else 50) for phase in cases}
        (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Nonpaged attention: independent numeric and native stage/LA checks passed',flush=True)


if __name__=='__main__':main()
