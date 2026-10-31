#!/usr/bin/env python3
"""Build trace-only images from the exact frozen CP1/CP2 generated sources."""
import argparse
import concurrent.futures
import copy
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from tilemega.build.identity import generate, sha, snapshot, validate_snapshot_inputs, verify
from tilemega.mx5_resources import inspect
from tilemega.mx6_resources import profile


def record(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=sha(path))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/root/tilemega'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();root=a.root.resolve();out=a.out.resolve();out.mkdir(exist_ok=a.resume)
    trees={}
    for arm in('cp1','cp2'):
        source=root/f'runs/mx6/{arm}/builds/qwen3_30b_a3b_l24/B1/decode/plan.so.source.json'
        original=json.loads(source.read_text());tree=out/f'{arm}_source'
        if not tree.exists():
            subprocess.run(['git','worktree','add','--detach',str(tree),original['head']],cwd=root,check=True)
        patch=Path(str(source)+'.patch')
        if patch.stat().st_size and not a.resume:
            subprocess.run(['git','apply',str(patch)],cwd=tree,check=True)
        assert all(sha(tree/name)==digest for name,digest in original['source_files'].items())
        trees[arm]=tree

    def build(item):
        arm,model,batch=item;tree=trees[arm]
        parent=root/f'runs/mx6/{arm}/builds/{model}/B{batch}/decode/plan.so'
        identity=verify(parent);folder=out/arm/model/f'B{batch}';folder.mkdir(parents=True,exist_ok=a.resume)
        so=folder/'plan.so';source=Path(str(so)+'.cu')
        if (folder/'COMPLETE.json').exists():
            verify(so);return json.loads((folder/'COMPLETE.json').read_text())
        reused=so.exists()
        if not reused:source.write_bytes(Path(str(parent)+'.cu').read_bytes())
        assert source.read_bytes()==Path(str(parent)+'.cu').read_bytes()
        args=shlex.split(Path(str(parent)+'.build_command.txt').read_text())
        args=[arg.replace(str(root/'include'),str(tree/'include'))
              .replace(str(parent),str(so)).replace('TILEMEGA_TRACE_STAGE=0','TILEMEGA_TRACE_STAGE=1') for arg in args]
        assert '-DTILEMEGA_TRACE_STAGE=1'in args and '-DTILEMEGA_MX_DIAG=0'in args
        if reused:
            assert shlex.split(Path(str(so)+'.build_command.txt').read_text())==args
            captured=json.loads(Path(str(so)+'.source.json').read_text())
        else:
            Path(str(so)+'.build_command.txt').write_text(shlex.join(args)+'\n')
            Path(str(so)+'.runtime_flags.txt').write_text(' '.join(v for v in args if v.startswith('-D'))+'\n')
            Path(str(so)+'.mx5_pin.json').write_bytes(Path(str(parent)+'.mx5_pin.json').read_bytes())
            captured=snapshot(tree,str(so)+'.source.json',compiler=args[0])
        assert captured['source_digest']==identity['source']['source_digest']
        inputs=[record(v)for v in(source,str(so)+'.source.json',str(so)+'.build_command.txt',args[0])]
        if reused:
            receipt=json.loads(Path(str(so)+'.trace_compile_receipt.json').read_text())
            assert receipt['returncode']==0 and receipt['inputs']==inputs and receipt['argv']==args
        else:
            start=time.monotonic()
            with Path(str(so)+'.ptxas.log').open('x')as f:
                result=subprocess.run(args,cwd=tree,stdout=f,stderr=subprocess.STDOUT)
            receipt=dict(kind='mx6_archived_trace_compile',argv=args,returncode=result.returncode,
                         compile_s=time.monotonic()-start,parent=record(parent),inputs=inputs)
            Path(str(so)+'.trace_compile_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
            if result.returncode:raise RuntimeError('trace compilation failed: '+str(so))
        validate_snapshot_inputs(tree,captured)
        assert inputs==[record(v['path'])for v in inputs]
        res=inspect(so,str(so)+'.ptxas.log',512,kernel_name='mx_moe_kernel_v3')
        if res['violations']:raise RuntimeError('trace resource candidate rejected: '+str(so))
        auxiliary=Path(str(so)+'.attribution.so')
        if auxiliary.exists():
            data=profile(so,auxiliary,str(so)+'.ptxas.log',str(auxiliary)+'.ptxas.log',source,tree,'mx_moe_kernel_v3')
            data['reused_auxiliary_compile']=record(str(auxiliary)+'.compile.json')
            data['attribution_helper']=record(root/'python/tilemega/mx6_resources.py')
            attr=Path(str(so)+'.attribution.json')
            with attr.open('x')as f:json.dump(data,f,indent=2);f.write('\n')
            for key in('body_instruction_status','body_instructions','body_code_bytes','program_instruction_count'):
                res[key]=data[key]
            res['body_attribution']=record(attr)
            res['attribution_compile_s']=json.loads(Path(str(auxiliary)+'.compile.json').read_text())['compile_s']
            Path(str(so)+'.resources.json').write_text(json.dumps(res,indent=2)+'\n')
        else:
            subprocess.run([sys.executable,'-m','tilemega.mx6_resources','--so',str(so),
                '--log',str(so)+'.ptxas.log','--threads','512','--kernel','mx_moe_kernel_v3',
                '--out',str(so)+'.resources.json'],cwd=root,check=True)
            res=json.loads(Path(str(so)+'.resources.json').read_text())
        manifest=copy.deepcopy(identity['plan']);manifest['trace_stage']=True
        manifest['source_snapshot']=str(so)+'.source.json';manifest['resources']=res
        manifest['runtime_allocations']['trace']=dict(dtype='u64',elements='577 * stages_per_step * 104 * 3')
        for key in('body_instructions','body_code_bytes','body_attribution'):
            manifest[key]=res[key]
        manifest['program_code_bytes']=res['code_bytes'];manifest['program_instruction_count']=res['program_instruction_count']
        manifest['diagnostic_parent']=record(parent);manifest['diagnostic_variant']=arm+'_trace'
        Path(str(so)+'.plan.json').write_text(json.dumps(manifest,indent=2)+'\n')
        generate(so,tree,str(so)+'.source.json');verify(so)
        mir=subprocess.run([sys.executable,str(root/'experiments/mx5/mir_check.py'),
            str(so)+'.ptxas.log','--out',str(folder/'MIR.json')],cwd=root,capture_output=True,text=True)
        (folder/'MIR.log').write_text(mir.stdout+mir.stderr)
        if mir.returncode:raise RuntimeError('trace MIR check failed: '+str(so))
        row=dict(arm=arm,model=model,batch=batch,binary=record(so),parent=record(parent),
                 source_head=captured['head'],source_digest=captured['source_digest'],resources=res['kernel'])
        (folder/'COMPLETE.json').write_text(json.dumps(row,indent=2)+'\n')
        print('ARCHIVED_TRACE_BUILD_PASS',arm,model,batch,flush=True);return row

    items=[(arm,model,b)for arm in('cp1','cp2')for model in('qwen3_30b_a3b_l24','olmoe_1b_7b')for b in(1,16)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:rows=list(pool.map(build,items))
    (out/'COMPLETE.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
