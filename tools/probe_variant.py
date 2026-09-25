#!/usr/bin/env python3
"""Compile a TaskBody wrapper once; report ptxas registers and actual sizeof storage.

The backend ABI receives dimensions and split at runtime. Identical generated
wrapper text therefore shares a physical compile across semantic classes and
split values; the solver still keeps a separate semantic resource-cache entry.
"""
import argparse, fcntl, hashlib, json, pathlib, re, shutil, subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1]

def serving_headers(source):
    """Hash only local headers reachable from this probe's translation unit."""
    pending=[name.decode() for name in re.findall(
        rb'^\s*#\s*include\s*[<"](tilemega/[^>"]+)[>"]',
        source.encode(),re.MULTILINE)]
    seen=set()
    while pending:
        name=pending.pop()
        path=ROOT/'include'/name
        if path in seen:continue
        if not path.is_file():raise FileNotFoundError(path)
        seen.add(path)
        body=path.read_bytes()
        pending.extend(item.decode() for item in re.findall(
            rb'^\s*#\s*include\s*[<"](tilemega/[^>"]+)[>"]',
            body,re.MULTILINE))
    return sorted(seen)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True)
    ap.add_argument('--arch',default='sm_89');ap.add_argument('--dtype',choices=['bf16','f32'],default='bf16')
    ap.add_argument('--tile',default='32,16,16,2');ap.add_argument('--nongemm',action='store_true')
    ap.add_argument('--serving',action='store_true')
    ap.add_argument('--head-dim',type=int,choices=(64,128),default=64)
    ap.add_argument('--qperkv',type=int,choices=(2,4),default=4)
    a=ap.parse_args();m,n,k,s=map(int,a.tile.split(','));threads=128 if a.dtype=='bf16' else 256
    if a.serving and a.dtype!='bf16':raise ValueError('serving resources require BF16')
    arch_id=int(a.arch.removeprefix('sm_'))*10
    pre=f'#define TILEMEGA_MODEL_BF16 {int(a.dtype=="bf16")}\n#define TILEMEGA_MIDPOINT_REFINE 0\n#define TILEMEGA_GEMM_TILE_M {m}\n#define TILEMEGA_GEMM_TILE_N {n}\n#define TILEMEGA_GEMM_TILE_K {k}\n#define TILEMEGA_GEMM_STAGES {s}\n'
    if a.serving:
        text=pre+'#include <tilemega/Target/ArchDispatch.h>\n#include <cstdio>\n'
        if a.nongemm:
            text+='#include <tilemega/Codegen/tasks/FusedAttentionTaskBody.h>\n'
            text+=f'using ProbeArch=tilemega::arch::ArchFromId<{arch_id}>::type;\nusing Body=tilemega::codegen::FusedAttentionTaskBody<ProbeArch,{a.head_dim},{a.qperkv},64,64,64,{str(a.head_dim==128).lower()}>;\n'
            text+='extern "C" __global__ __launch_bounds__(128) void probe_nongemm(tilemega::codegen::ServingAttentionOperands const* p) { extern __shared__ char bytes[]; Body::Run(*p,*reinterpret_cast<Body::SharedStorage*>(bytes),0,0,0,0); }\n'
            size='sizeof(Body::SharedStorage)'
        else:
            text+='#include <tilemega/Codegen/tasks/ServingGemmTaskBody.h>\n'
            text+=f'using ProbeArch=tilemega::arch::ArchFromId<{arch_id}>::type;\nusing Body=tilemega::codegen::ServingGemmTaskBody<ProbeArch,{m},{n},{k},{s}>;\n'
            text+='extern "C" __global__ __launch_bounds__(128) void probe_gemm(tilemega::codegen::ServingGemmOperands const* p) { extern __shared__ char bytes[]; Body::Run(*p,0,0,bytes); }\n'
            size='Body::kSharedBytes'
    else:
        text=pre+'#include <tilemega/Codegen/tasks/GemmStageTaskBody.h>\n#include <tilemega/Target/ArchDispatch.h>\n#include <cstdio>\n'
    if not a.serving and a.nongemm:
        headers=['RMSNorm','RoPE','KVAppend','Elementwise','Add','Embedding','QKNorm','AttentionChunk','GemmCombine']
        for body in headers:text+=f'#include <tilemega/Codegen/tasks/{body}TaskBody.h>\n'
        text+='using namespace tilemega::codegen;\nunion ProbeSmem { float rms[256]; float attention[TILEMEGA_ATTENTION_SCRATCH_EXTENT]; float pointwise[1]; };\n'
        for body in headers:
            cls='Attention' if body=='AttentionChunk' else body
            text+=f'extern "C" __global__ __launch_bounds__({threads}) void probe_{cls}(Params const* p,StageDesc const* stage,int task) {{ extern __shared__ char bytes[]; {cls}TaskBody<tilemega::arch::CurrentArch,ProbeSmem,{threads}>::RunTask(*p,*stage,*reinterpret_cast<ProbeSmem*>(bytes),task); }}\n'
        size='sizeof(ProbeSmem)'
    elif not a.serving:
        text+='using namespace tilemega::codegen;\n'
        text+=f'extern "C" __global__ __launch_bounds__({threads}) void probe_gemm(GemmInvocation const* g,int task) {{ extern __shared__ char bytes[]; GemmStageTaskBody<tilemega::arch::CurrentArch,GemmVariantSmem,{threads}>::RunTask<0>(*g,task,bytes,nullptr); }}\n'
        size='sizeof(GemmVariantSmem)'
    text+=f'int main() {{ std::printf("%zu {threads}\\n",{size}); }}\n'
    compiler='/usr/local/cuda/bin/nvcc'
    version=subprocess.check_output([compiler,'--version'],text=True)
    # Serving probes depend on a small, exact local include closure.  Hashing
    # every TaskBody made an unrelated RoPE or attention edit invalidate all
    # GEMM resources, turning one coordinate scan into serial recompilation.
    digest=hashlib.sha256((text+a.arch+version).encode())
    if a.serving:
        files=serving_headers(text)
        cutlass_revision=subprocess.check_output(
            ['git','-C',str(ROOT/'third_party/cutlass'),'rev-parse','HEAD'],
            text=True).strip()
        digest.update(cutlass_revision.encode())
    else:
        files=[p for root in (ROOT/'include/tilemega/Codegen/tasks',
                              ROOT/'include/tilemega/Backend',
                              ROOT/'include/tilemega/Target')
               for p in root.rglob('*') if p.is_file()]
        files.sort()
    for p in files:
        digest.update(str(p.relative_to(ROOT)).encode())
        digest.update(p.read_bytes())
    directory=a.cache/digest.hexdigest();directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'compile.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    result=directory/'resources.json';compiled=not result.exists()
    if compiled:
        free=shutil.disk_usage(directory).free//2**20;print(f'DISK NEED_MIB=1024 FREE_MIB={free}',flush=True)
        if free<1024:raise RuntimeError('insufficient disk for variant compilation')
        source=directory/'probe.cu';source.write_text(text)
        command=[compiler,'-std=c++17','-O2','-Xptxas=-v','-arch='+a.arch,'-DTILEMEGA_MIDPOINT_REFINE=0']
        for sub in ['include','third_party/cutlass/include','third_party/cutlass/tools/util/include','third_party/cutlass/test']:command+=['-I'+str(ROOT/sub)]
        command += [str(source),'-o',str(directory/'probe')]
        (directory/'command.json').write_text(json.dumps(command,indent=2)+'\n')
        p=subprocess.run(command,capture_output=True,text=True);(directory/'build.log').write_text(p.stdout+p.stderr)
        if p.returncode:raise RuntimeError('variant compile failed: '+str(directory/'build.log'))
        registers=[int(x) for x in re.findall(r'Used (\d+) registers',p.stderr)]
        if not registers:raise RuntimeError('missing ptxas registers')
        shared,actual_threads=map(int,subprocess.check_output([str(directory/'probe')],text=True).split())
        result.write_text(json.dumps(dict(registers=max(registers),shared_bytes=shared,threads=actual_threads,source_sha256=digest.hexdigest(),artifact=str(directory)),indent=2)+'\n')
    data=json.loads(result.read_text());data['compiled']=compiled;a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(data,indent=2)+'\n')
if __name__=='__main__':main()
