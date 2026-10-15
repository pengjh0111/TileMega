#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""CPU-only full-model contracts for the explicit native forward seed tuple."""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time


def model_counts(source):
    match=re.search(r'constexpr ModelSpec kModel\s*=\s*\{kDims,\s*ScalarType::kBF16,\s*kBuffers,\s*(\d+)u,\s*kGemms,\s*(\d+)u,\s*kStages,\s*(\d+)u',source)
    if not match or min(map(int,match.groups()))<=0:raise ValueError('generated full-model descriptor is missing or empty')
    return tuple(map(int,match.groups()))


def check_matrix(manifest,source,expected,geometry):
    counts=model_counts(source)
    if counts!=expected or len(manifest['gemms'])!=counts[1]:raise ValueError('full-model buffer/GEMM/stage counts changed')
    table=re.search(r'constexpr\s+GemmRuntimeDesc\s+kRuntimeGemms0\[\]\s*=\s*\{(.*?)\n\};',source,re.S)
    rows=re.findall(r'\{\s*(\d+)u?,\s*(\d+)u?,\s*(\d+)u?,\s*(\d+)u?,\s*(\d+)u?,\s*(\d+)u?\s*\}',table[1]if table else '')
    if len(rows)!=counts[1]:raise ValueError('actual generated GEMM dispatch count differs')
    for i,(g,row)in enumerate(zip(manifest['gemms'],rows)):
        _,split,m,n,k,stages=map(int,row)
        if g['index']!=i or any(g[key]!=v for key,v in zip(('split_k','tile_m','tile_n','tile_k','stages'),(split,m,n,k,stages))):raise ValueError('manifest and generated GEMM geometry differ')
        if geometry and any(g[key]!=value for key,value in geometry.items()):raise ValueError('explicit global tuple did not reach every full-model GEMM')
    return counts


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cli',type=Path,required=True);p.add_argument('--reference-cli',type=Path,required=True);p.add_argument('--export',type=Path,required=True);p.add_argument('--target',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--reuse-reference',type=Path);p.add_argument('--timeout-s',type=int,default=1800);p.add_argument('--feature-contract',type=Path,default=Path(__file__).resolve().parents[1]/'python/tilemega/build/feature_contract.py');a=p.parse_args();out=a.out.absolute()
    spec=importlib.util.spec_from_file_location('r16_seed_feature_contract',a.feature_contract);feature=importlib.util.module_from_spec(spec);spec.loader.exec_module(feature)
    if out.exists():raise FileExistsError('new CPU contract receipt directory required')
    out.mkdir(parents=True);inp=out/'input/nafnet';inp.mkdir(parents=True);bridge=inp/'bridge.json';shutil.copy2(a.export/'bridge_b1.json',bridge)
    flags=['--emit','serving','--serving','forward','--frontend','dnn','--batch','1','--runtime-target',str(a.target.absolute()),'--pg','l2','--resource-probes','0','--seed-arch','native','--nonpaged-weight-layout','tiled','--deferred-ln','auto','--reuse','l2','--global-la','auto','--dwpw-fuse','auto'];steps=[];report=dict(started=time.time(),gpu_executed=False,passed=False,steps=steps,export_sha256=hashlib.sha256((a.export/'manifest.json').read_bytes()).hexdigest(),cli_sha256=hashlib.sha256(a.cli.read_bytes()).hexdigest(),reference_cli_sha256=hashlib.sha256(a.reference_cli.read_bytes()).hexdigest());lock=threading.Lock()
    def run(name,cli,extra,success=True):
        directory=out/name;directory.mkdir();dest=directory/'plan.cu';command=[str(cli.absolute()),'compile',str(bridge),str(dest),*flags,*extra];env={**os.environ,'CUDA_VISIBLE_DEVICES':''};env.pop('TILEMEGA_DEVICE_INDEX',None)
        start=time.time()
        try:
            with (directory/'log').open('w')as log:code=subprocess.run(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout_s).returncode
        except subprocess.TimeoutExpired:
            with lock:steps.append(dict(name=name,command=command,code='timeout',seconds=time.time()-start))
            raise
        with lock:
            steps.append(dict(name=name,command=command,code=code,seconds=time.time()-start));(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if success and code:raise ValueError('valid full-model forward seed failed: '+name)
        if not success:
            message=(directory/'log').read_text()
            if code==0 or not any(text in message for text in ('forward seed','forward-seed-tile','--seed-arch native requires')):raise ValueError('bad tuple was accepted or rejected for an unrelated reason: '+name)
        return dest
    try:
        if a.reuse_reference:
            directory=a.reuse_reference.absolute();prior=json.loads((directory.parent/'report.json').read_text());row=next(x for x in prior['steps']if x['name']==directory.name)
            if row['code']!=0 or row['command'][0]!=str(a.reference_cli.absolute()) or row['command'][4:]!=flags or Path(row['command'][2]).read_bytes()!=bridge.read_bytes():raise ValueError('previous reference does not match the immutable CLI/full model/flags')
            reference=directory/'plan.cu';report['reference_reused']=dict(directory=str(directory),source_report_sha256=hashlib.sha256((directory.parent/'report.json').read_bytes()).hexdigest(),cuda_sha256=hashlib.sha256(reference.read_bytes()).hexdigest())
        else:reference=run('reference-default',a.reference_cli,[])
        positive=[('default',[]),('explicit-default',['--forward-seed-tile','M128N64K16S3']),('mma',['--forward-seed-tile','M64N128K64S2']),('tma-wgmma',['--forward-seed-tile','M64N128K64S2','--forward-gemm-impl','wgmma','--conv-impl','tma_im2col'])]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
            futures={name:pool.submit(run,name,a.cli,extra)for name,extra in positive}
            generated={name:future.result()for name,future in futures.items()}
        default=generated['default'];explicit_default=generated['explicit-default']
        if reference.read_bytes()!=default.read_bytes() or reference.read_bytes()!=explicit_default.read_bytes():raise ValueError('default generated CUDA bytes changed')
        expected=model_counts(default.read_text());shape=dict(tile_m=64,tile_n=128,tile_k=64,stages=2,split_k=1)
        dispatch=[]
        for name,dest in generated.items():
            manifest=json.loads(Path(str(dest)+'.plan.json').read_text());source=dest.read_text();rows=feature.validate_gemm_dispatch(manifest,feature.build_defines(source,[]),source);dispatch.append(dict(name=name,gemms=len(rows),actual_wgmma=sum(row['impl']=='wgmma'for row in rows)))
        report.update(actual_dispatch=dispatch,feature_contract_sha256=hashlib.sha256(a.feature_contract.read_bytes()).hexdigest())
        for name in ('mma','tma-wgmma'):
            dest=generated[name];manifest=json.loads(Path(str(dest)+'.plan.json').read_text());check_matrix(manifest,dest.read_text(),expected,shape)
            if manifest.get('forward_seed_tile')!=shape or manifest.get('forward_seed_explicit')is not True or manifest['seed_target_arch']!='sm_90':raise ValueError('explicit tuple/native target identity is missing')
            if name=='tma-wgmma' and not any(g['impl']=='wgmma'for g in manifest['gemms']):raise ValueError('Hopper request selected no actual WGMMA GEMM')
        bad=[('format',['--forward-seed-tile','64x128x64x2']),('zero',['--forward-seed-tile','M0N128K64S2']),('negative',['--forward-seed-tile','M-64N128K64S2']),('stages',['--forward-seed-tile','M64N128K64S1']),('overflow',['--forward-seed-tile','M999999999999N128K64S2']),('legacy',['--forward-seed-tile','M64N128K64S2','--seed-arch','legacy']),('solve',['--forward-seed-tile','M64N128K64S2','--solve',str(a.target.absolute())]),('pages',['--forward-seed-tile','M64N128K64S2','--pg','pages']),('conv-m128',['--forward-seed-tile','M128N128K64S2','--forward-gemm-impl','wgmma','--conv-impl','tma_im2col']),('conv-k128',['--forward-seed-tile','M64N128K128S2','--forward-gemm-impl','wgmma','--conv-impl','tma_im2col']),('wgmma-m32',['--forward-seed-tile','M32N128K64S2','--forward-gemm-impl','wgmma'])]
        for name,extra in bad:run('reject-'+name,a.cli,extra,False)
        report.update(passed=True,finished=time.time(),default_cuda_byte_identical=True,full_model_counts=dict(zip(('buffers','gemms','stages'),expected)),valid_cases=5,rejected_cases=len(bad),scope='native seed candidates only; no solver-optimality, GPU-correctness, occupancy or performance claim')
    except Exception as error:
        report.update(error=str(error),finished=time.time())
        raise
    finally:(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k!='steps'}))

if __name__=='__main__':main()
