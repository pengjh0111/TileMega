#!/usr/bin/env python3
"""Build and run the prescribed five-process DNN4 Phase P probes."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
TILES=[(64,32),(64,64),(128,64),(128,128),(128,256)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def inputs():
    files=[*sorted((ROOT/'include').rglob('*.h')),
           *sorted((ROOT/'include').rglob('*.cuh')),
           *sorted((ROOT/'scripts').glob('dnn4_probe_*.cu'))]
    return {str(p.relative_to(ROOT)):sha(p) for p in files}


def icache_asm(path):
    with path.open('w') as stream:
        for kb in [8,32,128,512,1024]:
            stream.write(f'template<> __device__ __noinline__ float Work<{kb}>(float x) {{\n')
            stream.write('float a=x+.1f,b=x+.2f,c=x+.3f,d=x+.4f,e=x+.5f,f=x+.6f,g=x+.7f,h=x+.8f;\n')
            stream.write('asm volatile(\n')
            for i in range(kb*1024//16):
                stream.write(f'"fma.rn.f32 %{i%8}, %{i%8}, %8, %9;\\n"\n')
            stream.write(': "+f"(a),"+f"(b),"+f"(c),"+f"(d),"+f"(e),"+f"(f),"+f"(g),"+f"(h) : "f"(.9999f),"f"(.00001f));\n')
            stream.write('return a+b+c+d+e+f+g+h;\n}\n')


def build(args):
    args.out.mkdir(parents=True,exist_ok=False)
    frozen=inputs();specs=[];skipped=[]
    for tm,tn in TILES:
        for slots in [2,3,4,6,8]:
            if slots*128*(tm+tn)>230400:
                skipped.append(dict(kind='pipe',tile_m=tm,tile_n=tn,slots=slots,
                                    reason='exceeds double-arena shared-memory budget'))
                continue
            specs.append(dict(kind='pipe',tile_m=tm,tile_n=tn,slots=slots))
    specs.append(dict(kind='pipe',tile_m=128,tile_n=128,slots=7))
    for tm,tn in TILES:
        for slots in [2,4,8]:
            specs.append(dict(kind='resident',tile_m=tm,tile_n=tn,slots=slots))
    specs.append(dict(kind='icache'))
    icache_asm(args.out/'dnn4_icache_ops.inc')

    def compile_one(item):
        index,spec=item;kind=spec['kind']
        name=kind if kind=='icache' else f"{kind}_{spec['tile_m']}x{spec['tile_n']}_s{spec['slots']}"
        binary=args.out/name;log=args.out/(name+'.ptxas.log')
        cpu=136+2*(index%4)
        command=['taskset','-c',f'{cpu}-{cpu+1}','nvcc','-rdc=true','-std=c++17','-O3',
                 '-gencode','arch=compute_90a,code=sm_90a','--expt-relaxed-constexpr',
                 '-Xptxas','-v','-Xptxas','--device-function-maxrregcount=224',
                 '-I'+str(ROOT/'include'),'-I'+str(ROOT/'third_party/cutlass/include'),
                 '-I'+str(args.out)]
        if kind!='icache':
            prefix='DNN4' if kind=='pipe' else 'F3_PROBE'
            command += [f'-D{prefix}_TM={spec["tile_m"]}',f'-D{prefix}_TN={spec["tile_n"]}',
                        f'-DDNN4_SLOTS={spec["slots"]}']
        command += [str(ROOT/f'scripts/dnn4_probe_{kind}.cu'),'-o',str(binary)]
        start=time.time()
        with log.open('w') as stream:
            done=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=1800,
                                env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        text=log.read_text()
        spills=list(map(int,re.findall(r'(\d+) bytes spill (?:stores|loads)',text)))
        if done.returncode or not spills or any(spills):
            raise RuntimeError('probe compile/spill failure: '+str(log))
        sass=args.out/(name+'.sass')
        with sass.open('w') as stream:
            subprocess.run(['cuobjdump','--dump-sass',str(binary)],stdout=stream,stderr=subprocess.STDOUT,check=True)
        insns=sum(bool(re.match(r'^\s*/\*[0-9a-fA-F]+\*/\s+(?:@!?[A-Z]+\d+\s+)?[A-Z]',line))
                  for line in sass.read_text().splitlines())
        result=dict(**spec,name=name,binary=str(binary),binary_sha256=sha(binary),
                    command=command,log=str(log),log_sha256=sha(log),
                    sass=str(sass),sass_sha256=sha(sass),all_functions_sass_bytes=16*insns,
                    build_seconds=time.time()-start)
        print(json.dumps(dict(built=name,seconds=result['build_seconds'])),flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        probes=list(pool.map(compile_one,enumerate(specs)))
    if inputs()!=frozen:raise RuntimeError('probe sources changed during build')
    save(args.out/'build.json',dict(evidence='verified',threads=288,inputs=frozen,probes=probes,
         skipped=skipped,icache_include_sha256=sha(args.out/'dnn4_icache_ops.inc'),
         nvcc_version=subprocess.check_output(['nvcc','--version'],text=True)))


def fixtures(build,kind):
    result=[]
    for p in build['probes']:
        if kind!='all' and kind!=p['kind']:continue
        if p['kind']=='icache':
            for kb in [8,32,128,512]:result.append((p,[str(kb)],'icache'))
        elif p['kind']=='resident':result.append((p,[],'resident'))
        else:
            if p['slots']!=7:
                for src in ['l2','dram']:result.append((p,[src,'132'],'pipe'))
            maximum=min(8,230400//(128*(p['tile_m']+p['tile_n'])))
            if (p['tile_m'],p['tile_n']) in [(128,128),(128,256)] and p['slots']==maximum:
                for n in [33,66,99,132]:
                    if not any(q[0]==p and q[1]==['l2',str(n)] for q in result):
                        result.append((p,['l2',str(n)],'bandwidth'))
                if not any(q[0]==p and q[1]==['dram','132'] for q in result):
                    result.append((p,['dram','132'],'bandwidth'))
    return result


def execute(args):
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD')!='1' or not os.environ.get('TILEMEGA_RUN_NONCE'):
        raise ValueError('calibration requires the scheduler and guard')
    if os.environ.get('TILEMEGA_DEVICE_INDEX')!='1':raise ValueError('Phase P calibration uses GPU1')
    sys.path.insert(0,str(ROOT/'scripts'))
    from dnn1_device import gpu
    build=json.loads(args.build.read_text())
    if inputs()!=build['inputs']:raise ValueError('probe source identity changed')
    args.out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for probe,arguments,family in fixtures(build,args.kind):
        binary=Path(probe['binary'])
        if sha(binary)!=probe['binary_sha256']:raise ValueError('probe binary changed')
        processes=[];name='_'.join([probe['name'],*arguments])
        for index in range(5):
            start=gpu('1');started=time.time()
            done=subprocess.run([str(binary),*arguments],capture_output=True,text=True,timeout=120)
            end=gpu('1');stem=f'{name}_p{index}'
            (args.out/(stem+'.stdout')).write_text(done.stdout)
            (args.out/(stem+'.stderr')).write_text(done.stderr)
            if done.returncode:raise RuntimeError('probe failed: '+stem+': '+done.stderr[-1000:])
            data=json.loads(done.stdout)
            metric='ns_per_k' if probe['kind']=='pipe' else ('ns' if probe['kind']=='resident' else 'hot_ns')
            if not math.isfinite(data[metric]) or data[metric]<=0:raise ValueError('invalid measured time')
            processes.append(dict(measurement=data,started=started,ended=time.time(),
                                  gpu=1,start_sm_clock_mhz=start['sm_clock_mhz'],
                                  end_sm_clock_mhz=end['sm_clock_mhz']))
        metrics=[p['measurement'] for p in processes]
        keys=['ns_per_k','per_cta_gbps','total_gbps'] if probe['kind']=='pipe' else (
             ['ns'] if probe['kind']=='resident' else ['hot_ns','cold_ns','l2_hot_ns','p_cold_us_per_kb','p_l2_us_per_kb'])
        row=dict(family=family,name=name,probe=probe,arguments=arguments,processes=processes,
                 median={key:statistics.median(m[key] for m in metrics) for key in keys},passes='5/5')
        rows.append(row)
        save(args.out/'result.json',dict(evidence='verified',status='partial',rows=rows))
        print(json.dumps(dict(name=name,**row['median'])),flush=True)
    if inputs()!=build['inputs']:raise ValueError('probe sources changed during measurement')
    save(args.out/'result.json',dict(evidence='verified',status='passed',rows=rows,
         build=str(args.build),build_sha256=sha(args.build),gpu=1,threads=288,
         qualification='provisional_container_pid_namespace'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['build','execute'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--build',type=Path)
    p.add_argument('--kind',choices=['all','pipe','resident','icache'],default='all')
    args=p.parse_args();args.out=args.out.resolve()
    if args.action=='build':build(args)
    else:
        if args.build is None:p.error('execute needs --build')
        args.build=args.build.resolve();execute(args)


if __name__=='__main__':main()
