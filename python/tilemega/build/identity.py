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
    prefixes=['include','lib','tools','python','cmake','CMakeLists.txt']
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
    result={'root':str(root),'head':git('rev-parse','HEAD').decode().strip(),
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


def verify(so, expected_source=None):
    so=Path(so);path=Path(str(so)+'.identity.json')
    identity=json.loads(path.read_text())
    if identity.get('schema')!=1:raise ValueError('unsupported artifact identity schema; rebuild the plan')
    payload={k:v for k,v in identity.items() if k!='artifact_id'}
    if digest(payload)!=identity['artifact_id']:raise ValueError('identity metadata digest mismatch')
    if expected_source is not None and any(identity['source'].get(key)!=expected_source.get(key)
            for key in ('source_digest','compiler_sha256')):
        raise ValueError('artifact source/compiler differs from the requested build')
    for suffix,key in [('', 'binary_sha256'),('.cu','generated_source_sha256'),
                       ('.plan.json','manifest_sha256'),('.ptxas.log','resources_sha256')]:
        if sha(str(so)+suffix)!=identity[key]:raise ValueError(f'artifact identity mismatch: {so}{suffix}')
    if 'build_command_sha256' in identity and sha(str(so)+'.build_command.txt')!=identity['build_command_sha256']:
        raise ValueError('artifact compiler command changed')
    complete=identity.get('complete_macros')
    if complete is not None:
        if complete['compiler_command']!=identity['options']:
            raise ValueError('macro capture compiler command differs from the build')
        for unit in complete['translation_units']:
            if sha(unit['macros_file'])!=unit['macros_sha256']:
                raise ValueError('complete preprocessor macro table changed')
    return identity


def optional_identity(so):
    """Old reference binaries remain loadable; R14 arms require an identity."""
    if Path(str(so)+'.identity.json').exists():return verify(so)
    manifest=Path(str(so)+'.plan.json')
    required=manifest.exists() and json.loads(manifest.read_text()).get('identity_schema')==1
    if required or os.environ.get('TILEMEGA_REQUIRE_IDENTITY')=='1':
        raise ValueError(f'missing serving artifact identity: {so}')
    return None


def validate_snapshot_inputs(root, source):
    root=Path(root).resolve()
    if not source.get('source_files'):
        raise ValueError('build snapshot lacks source input hashes')
    if source.get('root') and Path(source['root']).resolve()!=root:
        raise ValueError('build snapshot belongs to a different source tree')
    if any(not (root/p).is_file() or sha(root/p)!=value for p,value in source['source_files'].items()):
        raise ValueError('source changed during compilation; refuse artifact identity')
    if source.get('compiler') and sha(source['compiler'])!=source['compiler_sha256']:
        raise ValueError('plan compiler changed after its before-build snapshot')


DM_FIELDS=('frontend','reuse','deferred_ln','dwpw_fuse','moe_dynamic','moe_opaque',
           'moe_gemv','shared_weight_layout_sha256','dm_pool_la','dm_moe_la',
           'moe_binding','moe_bm','memory_arena_bytes','dm_reduction_mask',
           'dm_reduction_proved_stages','dnn_structure','dnn_structure_search_sha256',
           'moe_structure','moe_structure_search_sha256','routing_profile','moe_profile_pricing')


def generate(so, root, source_snapshot):
    so=Path(so).resolve();root=Path(root).resolve()
    source=json.loads(Path(source_snapshot).read_text())
    validate_snapshot_inputs(root,source)
    command=Path(str(so)+'.build_command.txt').read_text().strip();options=shlex.split(command)
    manifest=json.loads(Path(str(so)+'.plan.json').read_text())
    generated=Path(str(so)+'.cu').read_text()
    from .feature_contract import build_defines, implementations, validate
    defines=build_defines(generated,options)
    validate(manifest,defines)
    nvcc=options[0]
    version=subprocess.check_output([nvcc,'--version'],text=True).strip()
    from .arch import compiler_arch
    arch=compiler_arch(options)
    resources_path=Path(str(so)+'.ptxas.log')
    resources=parse_resources(resources_path.read_text())
    if not resources or any('registers' not in row for row in resources.values()):
        raise ValueError('missing ptxas kernel resources')
    library=ctypes.CDLL(str(so))
    shared=getattr(library,'tm_plan_shared_bytes',None)
    if shared is None:raise ValueError('new serving artifact lacks shared-memory identity export')
    shared.restype=ctypes.c_uint64;shared.argtypes=[]
    target=manifest.get('runtime_target')
    trace=any(v.strip() not in ('0','false') for k,v in defines.items()
              if k.startswith(('TILEMEGA_TRACE','TILEMEGA_PAGE_TRACE')))
    result={'schema':1,'source':source,'binary_path':str(so),'binary_sha256':sha(so),
            'generated_source_sha256':sha(str(so)+'.cu'),'manifest_sha256':sha(str(so)+'.plan.json'),
            'build_command_sha256':sha(str(so)+'.build_command.txt'),
            'resources_sha256':sha(resources_path),'compiler_version':version,'options':options,
            'defines':defines,'arch':arch,'nvcc_arch':arch,
            'caps':manifest.get('target_spec',{}).get('caps',{}),'plan':manifest,'shared_memory_bytes':int(shared()),
            'resources':resources,'trace':trace,
            'target_sha256':sha(target) if target and Path(target).is_file() else None,
            'implementations':implementations(manifest,defines)}
    if defines.get('TILEMEGA_DM_SUPPORT','0').strip()=='1':
        from .compiler_macros import capture
        result['dm']={key:manifest[key] for key in DM_FIELDS if key in manifest}
        result['complete_macros']=capture(options,Path(str(so)+'.macro_capture'))
    validate_snapshot_inputs(root,source)
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
