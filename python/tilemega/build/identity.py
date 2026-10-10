"""Bind generated serving artifacts and measured execution to immutable inputs."""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def snapshot(root, destination, compiler=None):
    root=Path(root).resolve()
    def git(*args):
        return subprocess.check_output(['git','-C',str(root),*args])
    prefixes=['include','lib','tools','python','CMakeLists.txt']
    diff=git('diff','--binary','HEAD','--',*prefixes)
    # git diff omits newly added files. Keep their bytes in the same patch so
    # HEAD + this patch reconstructs all locally changed compiler/runtime input.
    untracked=git('ls-files','--others','--exclude-standard','--',*prefixes).decode().splitlines()
    for p in sorted(untracked):
        if (root/p).is_file() and Path(p).suffix in {'.h','.hpp','.cuh','.cu','.cpp','.cc','.py','.cmake'}:
            part=subprocess.run(['git','diff','--no-index','--binary','--','/dev/null',p],
                                cwd=root,stdout=subprocess.PIPE,check=False)
            if part.returncode not in (0,1):raise ValueError('cannot archive untracked source')
            diff+=part.stdout
    paths=git('ls-files','-co','--exclude-standard','--',*prefixes).decode().splitlines()
    inputs={p:sha(root/p) for p in sorted(set(paths)) if (root/p).is_file() and
            (Path(p).suffix in {'.h','.hpp','.cuh','.cu','.cpp','.cc','.py','.cmake'} or p.endswith('CMakeLists.txt'))}
    patch=Path(str(destination)+'.patch');patch.write_bytes(diff)
    result={'head':git('rev-parse','HEAD').decode().strip(),
            'worktree_diff_sha256':hashlib.sha256(diff).hexdigest(),
            'source_files':inputs,'source_digest':digest(inputs),
            'patch':str(patch),'compiler':str(compiler) if compiler else None,
            'compiler_sha256':sha(compiler) if compiler else None}
    Path(destination).write_text(json.dumps(result,indent=2)+'\n')
    return result


def parse_resources(text):
    """Keep callee resources per entry; a spill in its hot callee counts too."""
    result={};entry=None;function=None
    for line in text.splitlines():
        m=re.search(r"Compiling entry function '([^']+)'",line)
        if m:
            entry=m[1];function=entry
            result.setdefault(entry,{'name':entry,'callees':{}})
        m=re.search(r"Function properties for\s+'?([^'\s]+)",line)
        if m:function=m[1]
        if not entry or not function:continue
        row=result[entry] if function==entry else result[entry]['callees'].setdefault(function,{'name':function})
        for pattern,key in [(r'Used (\d+) registers','registers'),
                            (r'(\d+) bytes stack frame','stack_bytes'),
                            (r'(\d+) bytes spill stores','spill_store_bytes'),
                            (r'(\d+) bytes spill loads','spill_load_bytes'),
                            (r'(\d+) bytes smem','static_smem_bytes')]:
            m=re.search(pattern,line)
            if m:row[key]=int(m[1])
    for row in result.values():
        row['spill']=any(x.get('spill_store_bytes',0) or x.get('spill_load_bytes',0)
                         for x in [row,*row['callees'].values()])
    return result


def bind_execution(identity, executor, loop=False, pdl=False):
    if executor not in ('L1','L2'):raise ValueError('execution identity needs an explicit executor')
    if identity.get('schema')=='tilemega.dm1.identity.v1':
        identity=dict(identity,plan=identity['execution'],
                      resources={k:dict(v,name=k) for k,v in identity['kernels'].items()})
    pg=identity['plan'].get('pg')
    if loop:
        if pg=='pages' and executor=='L2':needle='tilemega_loop_kernel'
        elif pg!='pages' and executor=='L1':needle='tilemega_l1_loop_kernel'
        else:raise ValueError('unsupported serving loop identity')
    else:needle='tilemega_l1_kernel' if executor=='L1' else 'tilemega_l2_kernel'
    matches=[v for k,v in identity['resources'].items() if needle in k]
    if len(matches)!=1:raise ValueError(f'expected one executed kernel {needle}, found {len(matches)}')
    result={'artifact_id':identity['artifact_id'],'executor':executor,'loop':bool(loop),
            'pdl':bool(pdl),'kernel':matches[0]['name'],'spill':matches[0]['spill'],
            'trace':identity['trace']}
    result['execution_id']=digest(result)
    return result


def verify(so):
    so=Path(so);path=Path(str(so)+'.identity.json')
    identity=json.loads(path.read_text())
    if identity.get('schema')=='tilemega.dm1.identity.v1':
        from .dm_identity import verify as verify_dm
        return verify_dm(so)
    payload={k:v for k,v in identity.items() if k!='artifact_id'}
    if digest(payload)!=identity['artifact_id']:raise ValueError('identity metadata digest mismatch')
    for suffix,key in [('', 'binary_sha256'),('.cu','generated_source_sha256'),
                       ('.plan.json','manifest_sha256'),('.ptxas.log','resources_sha256')]:
        if sha(str(so)+suffix)!=identity[key]:raise ValueError(f'artifact identity mismatch: {so}{suffix}')
    return identity


def optional_identity(so):
    """Old reference binaries remain loadable; R14 arms require an identity."""
    if Path(str(so)+'.identity.json').exists():return verify(so)
    manifest=Path(str(so)+'.plan.json')
    required=manifest.exists() and json.loads(manifest.read_text()).get('identity_schema')==1
    if required or os.environ.get('TILEMEGA_REQUIRE_IDENTITY')=='1':
        raise ValueError(f'missing serving artifact identity: {so}')
    return None


def generate(so, root, source_snapshot):
    so=Path(so).resolve();root=Path(root).resolve()
    source=json.loads(Path(source_snapshot).read_text())
    if any(not (root/p).is_file() or sha(root/p)!=value for p,value in source['source_files'].items()):
        raise ValueError('source changed during compilation; refuse artifact identity')
    command=Path(str(so)+'.build_command.txt').read_text().strip();options=shlex.split(command)
    manifest=json.loads(Path(str(so)+'.plan.json').read_text())
    generated=Path(str(so)+'.cu').read_text()
    defines=dict(re.findall(r'^#define\s+(\w+)\s+([^\n]+)',generated,re.M))
    for arg in options:
        if arg.startswith('-D'):
            name,_,value=arg[2:].partition('=');defines[name]=value or '1'
    nvcc=options[0]
    version=subprocess.check_output([nvcc,'--version'],text=True).strip()
    arch=next((x.split('=',1)[1] for x in options if x.startswith('-arch=')),None)
    if arch is None and '-arch' in options:arch=options[options.index('-arch')+1]
    resources_path=Path(str(so)+'.ptxas.log')
    resources=parse_resources(resources_path.read_text())
    paged=manifest.get('pg')=='pages';decode=manifest.get('phase')=='decode'
    library=ctypes.CDLL(str(so))
    shared=getattr(library,'tm_plan_shared_bytes',None)
    if shared is None:raise ValueError('new serving artifact lacks shared-memory identity export')
    shared.restype=ctypes.c_uint64;shared.argtypes=[]
    target=manifest.get('runtime_target')
    trace=any(v.strip() not in ('0','false') for k,v in defines.items()
              if k.startswith(('TILEMEGA_TRACE','TILEMEGA_PAGE_TRACE')))
    result={'schema':1,'source':source,'binary_path':str(so),'binary_sha256':sha(so),
            'generated_source_sha256':sha(str(so)+'.cu'),'manifest_sha256':sha(str(so)+'.plan.json'),
            'resources_sha256':sha(resources_path),'compiler_version':version,'options':options,
            'defines':defines,'arch':arch,'plan':manifest,'shared_memory_bytes':int(shared()),
            'resources':resources,'trace':trace,
            'target_sha256':sha(target) if target and Path(target).is_file() else None,
            'implementations':{'dn_vector_sums':bool(manifest.get('dn_vector_sums',False)),
                'swiglu_interleave_u':int(manifest.get('swiglu_interleave_u',16)),'gemms':[{'index':g['index'],'impl':g.get('impl','mma16'),
                'task_body':'PagedGemmTaskBody' if paged else ('ServingGemvTaskBody' if g.get('impl')=='gemv' else 'ServingGemmTaskBody'),
                'mma_reg_pipe':int(defines.get('TILEMEGA_MMA_REG_PIPE','0')),
                'ep_direct':int(defines.get('TILEMEGA_EP_DIRECT','0'))} for g in manifest['gemms']],
                'attention':{'impl':manifest.get('attention_impl','mma16'),
                'noinline':bool(manifest.get('attention_noinline',False)),
                'buffers':int(defines.get('TILEMEGA_ATTENTION_BUFFERS','2')),
                'task_body':('PagedAttentionTaskBody' if paged else 'IndependentAttentionTaskBody') if decode else 'FusedAttentionTaskBody',
                'page_policy':('Packed' if paged else 'WarpPrivate') if decode else None,
                'load':manifest.get('paged_attention_load','loader'),
                'frontier':bool(int(defines.get('TILEMEGA_ATTENTION_FRONTIER','0')))}}}
    result['artifact_id']=digest(result)
    Path(str(so)+'.identity.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--so',type=Path);p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--capture',action='store_true');p.add_argument('--compiler',type=Path)
    p.add_argument('--check-reuse',type=Path)
    a=p.parse_args()
    if a.capture:snapshot(a.root,a.snapshot,a.compiler)
    elif a.check_reuse:
        old=verify(a.check_reuse)['source'];new=json.loads(a.snapshot.read_text())
        if any(old[k]!=new[k] for k in ('source_digest','compiler_sha256')):
            raise ValueError('reused binary has different source or compiler inputs')
    else:generate(a.so,a.root,a.snapshot)

if __name__=='__main__':main()
