#!/usr/bin/env python3
"""Verify produced activations in forward and prefill paged GEMM plans."""
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
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--multipage', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    nvcc = '/usr/local/cuda/bin/nvcc'
    if args.attention and args.multipage:parser.error('choose one fixture kind')
    sources = [('test/unit/dm_paged_multipage_phase_test.cu' if args.multipage else
        'test/unit/dm_paged_attention_test.cu' if args.attention else
        'test/unit/dm_paged_phase_test.cu'), 'lib/Target/TargetSpec.cpp',
        'lib/Support/Json.cpp', 'lib/Codegen/RuntimeTaskGraph.cpp',
        'lib/Solver/PlanMaterialize.cpp', 'lib/Dialect/CouplingGraph/PlacementPlan.cpp',
        'lib/Solver/BalancedPlacement.cpp', 'lib/Solver/ListScheduler.cpp']
    common = [nvcc, '-std=c++17', '-O2', '--expt-relaxed-constexpr',
              '-I' + str(args.root/'include'),
              '-I' + str(args.root/'third_party/cutlass/include'),
              '-I' + str(args.root/'third_party/cutlass/tools/util/include')]
    dependencies = {Path(__file__).resolve(), Path(__file__).with_name('identity_dm.py').resolve()}
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
        scope=('native PageStream forward/prefill plans with four-page B stages wrapping a five-slot ring; lookahead, produced activations, poisoned epochs, independent FP32/BF16-store oracles and bitwise L1/L2' if args.multipage else
            'paged QKV -> prefill attention -> o_proj with two query blocks, causal FP32 oracle, exact consumer table and bitwise L1/L2' if args.attention else
            'two dense GEMMs through native paged forward/prefill C ABI; produced activation, weight pages, poisoned intermediates and bitwise L1/L2') + '; binding-aware pages and model gates pending')
    cases=(0,) if args.attention else (2,0)
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
            for tile in ((16,) if args.multipage else (64,) if args.attention else (128,) if args.smoke else (128,16,32,64)):
                for arch in ((89,) if args.smoke or (not args.attention and not args.multipage and tile!=128) else (80,89,90,100,120)):
                    binary=args.out/f'phase{phase}-m{tile}-sm_{arch}';log=binary.with_suffix('.log')
                    configuration=([] if args.attention else
                        [f'-DDM_TEST_PHASE={phase}',f'-DDM_TEST_TILE_M={tile}'])
                    command=['flock',LOCK,*common,f'-arch=sm_{arch}',f'-DTILEMEGA_ARCH_ID={arch*10}',
                        *configuration,'-Xptxas=-v',
                        str(args.root/sources[0]),*objects,'-o',str(binary)]
                    with log.open('w') as stream:subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
                    kernels=resources(log.read_text())
                    result['builds'].append(dict(phase=phase,tile_m=tile,arch=arch,command=command,
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
                        execution=dict(pg='pages',executor=['L1','L2'],loop=False,
                            phase=0 if args.attention else phase),
                        implementations=dict(gemm='PagedGemmTaskBody',
                            attention='FusedAttentionTaskBody' if args.attention else None,
                            fixture=sources[0],tile_m=tile,tile_n=128 if args.multipage else 64,tile_k=128 if args.multipage else 64),
                        placement=dict(pages=5 if args.multipage else 2,page_bytes=8192,lookahead_bytes=32768),
                        kernels=kernels,spill=any(row['spill'] for row in kernels.values()))
                    identity['artifact_id']=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                    result['builds'][-1]['artifact_id']=identity['artifact_id']
                    Path(str(binary)+'.identity.json').write_text(json.dumps(identity,indent=2)+'\n')
            for process in range(1 if args.smoke else 50):
                tile=16 if args.multipage else 64 if args.attention else 128 if args.smoke else (16,32,64,128)[process%4]
                binary=args.out/f'phase{phase}-m{tile}-sm_89';log=args.out/f'process-phase{phase}-{process:02d}.log'
                with log.open('w') as stream:
                    status=subprocess.run(['flock',LOCK,'timeout','--signal=KILL','300',str(binary)],
                        stdout=stream,stderr=subprocess.STDOUT).returncode
                result['fresh_processes'].append(dict(phase=phase,tile_m=tile,process=process,exit_code=status,log_sha256=sha(log)))
                if status:raise RuntimeError(f'paged phase {phase}, process {process} failed; see {log}')
        if hashes!={str(path):sha(path) for path in sorted(dependencies)}:
            raise RuntimeError('paged phase inputs changed during validation')
        result['passed']=True
    finally:
        result['pass_rate']={str(phase):dict(passes=sum(row['exit_code']==0 for row in result['fresh_processes'] if row['phase']==phase),
            attempts=sum(row['phase']==phase for row in result['fresh_processes']),required=1 if args.smoke else 50) for phase in cases}
        (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Paged phases: native C ABI, numerical and bitwise L1/L2 checks passed',flush=True)


if __name__=='__main__':main()
