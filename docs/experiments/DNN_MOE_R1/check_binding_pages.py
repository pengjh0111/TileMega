#!/usr/bin/env python3
"""Check binding-gated synthetic pages before full expert-body integration."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
from identity_dm import resources, sha

LOCK='/root/r14_work/gpu.lock'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    source=args.root/'test/unit/dm_binding_pages_test.cu'
    nvcc='/usr/local/cuda/bin/nvcc'
    common=[nvcc,'-std=c++17','-O2','--expt-relaxed-constexpr',
        '-I'+str(args.root/'include'),'-I'+str(args.root/'third_party/cutlass/include')]
    listing=subprocess.check_output(['flock',LOCK,*common,'-arch=sm_80',
        '-DTILEMEGA_ARCH_ID=800','-M',str(source)],text=True)
    inputs={Path(__file__).resolve(),Path(__file__).with_name('identity_dm.py').resolve()}
    for word in shlex.split(listing.replace('\\\n',' '))[1:]:
        path=Path(word).resolve()
        if path.is_file():inputs.add(path)
    hashes={str(path):sha(path) for path in sorted(inputs)}
    result=dict(evidence='verified',passed=False,inputs=hashes,builds=[],processes=[],
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.root,text=True).strip(),
        diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=args.root)).hexdigest(),
        compiler_version=subprocess.check_output([nvcc,'--version'],text=True),compiler_sha256=sha(nvcc),
        baselines={'vllm':'0.30.0'},
        scope='synthetic binding gate/page ring, nonblocking lookahead probe, empty notifications and 32 epochs; full PageStream/expert TaskBodies and model gates pending')
    try:
        for arch in ((89,) if args.smoke else (80,89,90,100,120)):
            binary=args.out/f'binding-sm_{arch}';log=binary.with_suffix('.log')
            flags=[f'-arch=sm_{arch}',f'-DTILEMEGA_ARCH_ID={arch*10}']
            command=['flock',LOCK,*common,*flags,'-Xptxas=-v',str(source),'-o',str(binary)]
            with log.open('w') as stream:
                subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
            kernels=resources(log.read_text())
            macros=binary.with_suffix('.macros.txt')
            subprocess.run(['flock',LOCK,*common,*flags,'-E','-Xcompiler=-dM',str(source),'-o',str(macros)],check=True)
            identity=dict(schema='tilemega.dm1.native-test.identity.v1',
                source={k:result[k] for k in ('head','diff_sha256','inputs')},
                cu_sha256=sha(source),binary_sha256=sha(binary),compiler_command=command[2:],
                compiler_version=result['compiler_version'],compiler_sha256=result['compiler_sha256'],
                macros=dict(re.findall(r'^#define\s+(\S+)(?:[ \t]+(.*))?$',macros.read_text(),re.M)),
                baselines=result['baselines'],
                arch=f'sm_{arch}',execution=dict(pg='pages',phase='synthetic'),
                implementations=['BindingGate','MoeBindingView','PageRing'],
                kernels=kernels,spill=any(row['spill'] for row in kernels.values()))
            identity['artifact_id']=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            Path(str(binary)+'.identity.json').write_text(json.dumps(identity,indent=2)+'\n')
            result['builds'].append(dict(arch=arch,command=command,binary_sha256=sha(binary),
                log_sha256=sha(log),artifact_id=identity['artifact_id'],kernels=kernels,spill=identity['spill']))
        for process in range(1 if args.smoke else 50):
            log=args.out/f'process-{process:02d}.log'
            with log.open('w') as stream:
                status=subprocess.run(['flock',LOCK,str(args.out/'binding-sm_89')],
                    stdout=stream,stderr=subprocess.STDOUT,timeout=300).returncode
            result['processes'].append(dict(process=process,exit_code=status,log_sha256=sha(log)))
            if status:raise RuntimeError(f'binding page process {process} failed; see {log}')
        if hashes!={str(path):sha(path) for path in sorted(inputs)}:
            raise RuntimeError('binding page inputs changed during validation')
        result['passed']=True
    finally:
        result['pass_rate']=dict(passes=sum(row['exit_code']==0 for row in result['processes']),
            attempts=len(result['processes']),required=1 if args.smoke else 50)
        (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Binding pages: gate, indirect addresses, empty notifications and ring wraps passed',flush=True)


if __name__=='__main__':main()
