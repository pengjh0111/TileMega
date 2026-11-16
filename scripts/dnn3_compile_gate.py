#!/usr/bin/env python3
"""Compile measured F3 candidates with frozen sources and Hopper launch limits."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from tilemega.build.identity import generate,sha,snapshot,verify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu-gate',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True,help='immutable git checkout for all native inputs')
    p.add_argument('--compiler',type=Path)
    p.add_argument('--ranks',type=int,nargs='+',choices=range(4),default=list(range(4)))
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.source=args.source.resolve(strict=True);args.out=args.out.resolve()
    gate=json.loads(args.cpu_gate.read_text());ranks=sorted(set(args.ranks))
    if gate['status']!='passed' or len(gate['rows'])!=20 or any(len(r['candidates'])!=4 for r in gate['rows']):
        raise ValueError('require passed20-cell,80-candidate measured solve/codegen gate')
    expected_compiler=next((path,digest) for path,digest in gate['inputs'].items() if Path(path).name=='tilemega')
    compiler=(args.compiler or Path(expected_compiler[0])).resolve(strict=True)
    if sha(compiler)!=expected_compiler[1]:raise ValueError('compiler differs from the CPU solve gate')
    args.out.mkdir(parents=True,exist_ok=False)
    frozen_compiler=args.out/'tilemega';shutil.copy2(compiler,frozen_compiler)
    source_snapshot=args.out/'source.json';snapshot(args.source,source_snapshot,frozen_compiler)
    script_digest=sha(Path(__file__));rows=[]
    receipt=dict(evidence='verified',protocol='CPU native compilation; actual runtime residency and P-SEL separate',
        source=str(args.source),source_snapshot_sha256=sha(source_snapshot),
        cpu_gate=str(args.cpu_gate.resolve()),cpu_gate_sha256=sha(args.cpu_gate),
        script_sha256=script_digest,ranks=ranks,rows=rows,status='partial',gpu_executed=False)
    def save():
        (args.out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    for rank in ranks:
        for cell in gate['rows']:
            item=cell['candidates'][rank];cu=Path(item['source'])
            if item['rank']!=rank or sha(cu)!=item['sha256']:raise ValueError('candidate CUDA source changed')
            folder=args.out/f"{cell['model']}_b{cell['batch']}"/f'candidate-{rank}'
            folder.mkdir(parents=True);lib=folder/'model.so'
            for suffix in ('.cu','.plan.json','.tmexec.mlir'):
                origin=cu if suffix=='.cu' else Path(str(cu)+suffix)
                shutil.copyfile(origin,Path(str(lib)+suffix))
            command=['/usr/local/cuda/bin/nvcc','-rdc=true','-std=c++17','-O3',
                '-gencode','arch=compute_90a,code=sm_90a','--expt-relaxed-constexpr',
                '-shared','-Xcompiler','-fPIC','-cudart','shared','-Xptxas','-v',
                '--split-compile','1','-I'+str(args.source/'include'),
                '-I'+str(ROOT/'third_party/cutlass/include'),str(lib)+'.cu','-o',str(lib),
                '-Xptxas','--device-function-maxrregcount=224']
            Path(str(lib)+'.build_command.txt').write_text(shlex.join(command)+'\n')
            log=Path(str(lib)+'.ptxas.log')
            with log.open('w') as stream:
                done=subprocess.run(['taskset','-c','136-143',sys.executable,
                    str(ROOT/'scripts/run_sm90r1_build.py'),'--receipt',str(lib)+'.build_limits.json',
                    '--max-gib','48','--timeout','1800','--lock','/tmp/tilemega_dnn3_native_gate.lock',
                    '--',*command],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),
                    stdout=stream,stderr=subprocess.STDOUT)
            text=log.read_text();spills=list(map(int,re.findall(r'(\d+) bytes spill (?:stores|loads)',text)))
            registers=list(map(int,re.findall(r'Used (\d+) registers',text)))
            row=dict(model=cell['model'],batch=cell['batch'],rank=rank,out=str(folder),
                compile_exit=done.returncode,spill_max=max(spills,default=-1),
                register_max=max(registers,default=-1),status='rejected')
            if done.returncode==0 and spills and not any(spills) and registers and max(registers)<=224 and 'C7509' not in text:
                generate(lib,args.source,source_snapshot);identity=verify(lib)
                entries=[r for name,r in identity['resources'].items() if 'sm90_forward_f3_kernel' in name]
                if len(entries)!=1:raise ValueError('expected exactly one F3 entry resource record')
                entry_registers=entries[0]['registers'];row['entry_registers']=entry_registers
                # Hopper rounds9 warps to12 for the per-CTA register check;
                # 169+ static registers fail despite the224-register callee cap.
                if entry_registers<=168:
                    row.update(status='passed',artifact_id=identity['artifact_id'])
                else:row['reason']='288-thread Hopper CTA fails static register allocation'
            rows.append(row);save();print(json.dumps(row),flush=True)
            if sha(Path(__file__))!=script_digest:raise ValueError('native gate script changed')
    receipt['status']='passed' if all(r['status']=='passed' for r in rows) else 'failed'
    save()
    return int(receipt['status']!='passed')


if __name__=='__main__':
    raise SystemExit(main())
