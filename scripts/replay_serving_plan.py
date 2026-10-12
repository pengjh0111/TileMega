#!/usr/bin/env python3
"""Rebuild one frozen serving design at explicit source revisions for RF-1.

This is a build-only diagnostic. Measurements are separate guarded bench
steps, and intermediate bisect revisions must be selected from accepted data.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from tilemega.build.identity import snapshot, generate, sha
from tilemega.bench.gpu_guard import stop


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()


def nvcc_command(original,tree,source,binary):
    result=[];output=False;include=False
    for token in original:
        if output:result.append(str(binary));output=False;continue
        if include:
            result.append(str(tree/'include') if (Path(token)/'tilemega').is_dir() else token)
            include=False;continue
        if token=='-I':result.append(token);include=True;continue
        if token=='-o':result.append(token);output=True;continue
        if token.endswith('.cu') and not token.startswith('-'):token=str(source)
        elif token.startswith('-I') and token[2:].endswith('/include') and \
                (Path(token[2:])/'tilemega').exists():token='-I'+str(tree/'include')
        elif '/lib/' in token and token.endswith('.cpp'):
            suffix=token.split('/lib/',1)[1]
            if (tree/'lib'/suffix).is_file():token=str(tree/'lib'/suffix)
        result.append(token)
    if output or include or '-o' not in result or str(source) not in result:
        raise ValueError('original NVCC command lacks source or output')
    return result


def execute(command,tree,log,timeout):
    log.with_suffix('.command.json').write_text(json.dumps(command)+'\n')
    with log.open('w') as stream:
        proc=subprocess.Popen(command,cwd=tree,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=timeout)
        except BaseException:stop(proc);raise
    if code:raise RuntimeError(f'command failed ({code}): {log}')


def prepare(args):
    root=args.root.resolve();out=args.out.resolve()
    if out.exists():raise FileExistsError('replay inputs already exist')
    spec=json.loads(args.design.read_text())
    out.mkdir(parents=True)
    cases=[]
    for case in spec['cases']:
        dest=out/'inputs'/case['cell'];dest.mkdir(parents=True)
        paths={}
        for key in ('mlir','manifest','target','nvcc'):
            source=Path(case[key]);target=dest/source.name
            if target.exists():raise ValueError('ambiguous input filenames')
            shutil.copy2(source,target);paths[key]=dict(path=str(target),sha256=sha(target))
        cases.append(dict(cell=case['cell'],batch=case['batch'],inputs=paths,
                          overrides=case['overrides']))
    trees={}
    args.work.mkdir(parents=True,exist_ok=True)
    for revision in args.revision:
        head=git(root,'rev-parse',revision);tree=args.work.resolve()/head[:12]
        subprocess.run(['git','-C',str(root),'worktree','add','--detach','--no-checkout',str(tree),head],check=True)
        for command in (['sparse-checkout','init','--cone'],
                ['sparse-checkout','set','include','lib','tools','cmake','python','scripts','test','configs'],
                ['read-tree','-mu','HEAD']):
            subprocess.run(['git','-C',str(tree),*command],check=True)
        third=tree/'third_party'
        if third.exists():
            if any(third.iterdir()):raise ValueError('refuse to replace populated third_party')
            third.rmdir()
        third.symlink_to(root/'third_party',target_is_directory=True)
        # Historical fingerprints read these files even with tests disabled.
        for relative in spec.get('fingerprint_inputs',[]):
            target=tree/relative;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(subprocess.check_output(['git','-C',str(root),'show',head+':'+relative]))
        trees[head]=str(tree)
    shutil.copy2(__file__,out/'runner.py')
    (out/'inputs.json').write_text(json.dumps(dict(root=str(root),trees=trees,cases=cases,
        runner_sha256=sha(out/'runner.py'),mlir_dir=args.mlir_dir,nvcc_timeout_s=args.nvcc_timeout_s,
        sources_only=args.sources_only),indent=2)+'\n')


def build(args):
    out=args.out.resolve();record=json.loads((out/'inputs.json').read_text());root=Path(record['root'])
    if sha(Path(__file__).resolve())!=record['runner_sha256']:raise ValueError('replay driver changed')
    for case in record['cases']:
        for value in case['inputs'].values():
            if sha(value['path'])!=value['sha256']:raise ValueError('replay design changed')
    built=[]
    with Path(os.environ['TILEMEGA_GPU_LOCK']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for head,location in record['trees'].items():
            tree=Path(location);directory=out/head[:12];directory.mkdir(exist_ok=True)
            if git(tree,'rev-parse','HEAD')!=head or git(tree,'diff','--name-only','HEAD'):
                raise ValueError('replay source revision changed')
            compiler=tree/'build-r15/tools/tilemega'
            command=['cmake','-S',str(tree),'-B',str(tree/'build-r15'),'-G','Ninja',
                '-DCMAKE_BUILD_TYPE=Release','-DTILEMEGA_TARGET_ARCH=sm_89',
                '-DTILEMEGA_BUILD_TESTS=OFF','-DTILEMEGA_BUILD_VERIFY=OFF',
                '-DTILEMEGA_BUILD_EXPERIMENTAL=OFF','-DMLIR_DIR='+record['mlir_dir']]
            command += ['-DTILEMEGA_'+name+'_BUILD_DIR='+str(root/('build-'+name.lower()))
                        for name in ('ISL','POLYLIB','BARVINOK')]
            execute(command,tree,directory/'configure.log',600)
            execute(['cmake','--build',str(tree/'build-r15'),'--target','tilemega','-j','6'],
                    tree,directory/'compiler.log',10800)
            for case in record['cases']:
                folder=directory/case['cell'];folder.mkdir(exist_ok=True);so=folder/'plan.so'
                if so.exists():raise FileExistsError('refuse to overwrite replay binary')
                inputs={k:Path(v['path']) for k,v in case['inputs'].items()}
                manifest=json.loads(inputs['manifest'].read_text())
                options=dict(case['overrides'],pg=manifest['pg'],
                    page_bytes=(manifest.get('pages') or {}).get('page_bytes',16384),
                    weight_layout='tiled' if manifest['pg']=='pages' else 'row')
                if manifest['pg']=='pages':options['paged_la']=0
                command=[str(compiler),'compile',str(inputs['mlir']),str(so)+'.cu',
                    '--serving',manifest['phase'],'--emit','serving','--batch',str(case['batch']),
                    '--capacity',str(manifest['capacity']),'--past-range',
                    f"{manifest['past_lo']}:{manifest['past_hi']}",'--runtime-target',str(inputs['target'])]
                for key,value in options.items():command+=['--'+key.replace('_','-'),str(value)]
                snapshot(tree,folder/'source.json',compiler)
                execute(command,tree,folder/'generate.log',1800)
                if record.get('sources_only'):
                    built.append(dict(head=head,cell=case['cell'],source=str(so)+'.cu',
                                      cu_sha256=sha(str(so)+'.cu')))
                    (out/'builds.json').write_text(json.dumps(built,indent=2)+'\n')
                    print('GENERATED '+head[:12]+' '+case['cell'],flush=True)
                    continue
                nvcc=nvcc_command(shlex.split(inputs['nvcc'].read_text()),tree,Path(str(so)+'.cu'),so)
                execute(nvcc,tree,Path(str(so)+'.ptxas.log'),record['nvcc_timeout_s'])
                Path(str(so)+'.build_command.txt').write_text(shlex.join(nvcc)+'\n')
                generated_manifest=Path(str(so)+'.cu.plan.json')
                if not generated_manifest.is_file():raise ValueError('replay compiler did not emit its manifest')
                actual=json.loads(generated_manifest.read_text())
                fields=('index','tile_m','tile_n','tile_k','stages','split_k')
                geometry=lambda m:[tuple(g[k] for k in fields) for g in m['gemms']]
                if geometry(actual)!=geometry(manifest):raise ValueError('replay changed the fixed GEMM geometry')
                shutil.copy2(generated_manifest,str(so)+'.plan.json')
                identity=generate(so,tree,folder/'source.json')
                built.append(dict(head=head,cell=case['cell'],library=str(so),artifact_id=identity['artifact_id']))
                (out/'builds.json').write_text(json.dumps(built,indent=2)+'\n')
                print('BUILT '+head[:12]+' '+case['cell'],flush=True)


    if record.get('sources_only'):
        comparisons=compare_sources(built)
        (out/'source_comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
        if not comparisons['passed']:
            raise RuntimeError('default-workload generated CUDA differs; inspect source_comparison.json')


def compare_sources(rows):
    by_cell={}
    for row in rows:by_cell.setdefault(row['cell'],[]).append(row)
    if not by_cell or any(len(group)!=2 or len({r['head'] for r in group})!=2
                          for group in by_cell.values()):
        raise ValueError('source comparison needs exactly two revisions per cell')
    cases=[]
    for cell,group in sorted(by_cell.items()):
        for row in group:
            if sha(row['source'])!=row['cu_sha256']:raise ValueError('generated source changed')
        cases.append(dict(cell=cell,passed=Path(group[0]['source']).read_bytes()==
                          Path(group[1]['source']).read_bytes(),sources=group))
    return dict(passed=all(row['passed'] for row in cases),cases=cases,
                scope='fixed designs at default 64/1024; source only, no GPU or binary equivalence claim')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','build'))
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--work',type=Path)
    parser.add_argument('--design',type=Path)
    parser.add_argument('--revision',action='append')
    parser.add_argument('--mlir-dir',default='/root/toolchains/mlir-23a60f15/lib/cmake/mlir')
    parser.add_argument('--nvcc-timeout-s',type=int,default=10800)
    parser.add_argument('--sources-only',action='store_true',help='G-REG-W: compare generated CUDA at two revisions without nvcc')
    args=parser.parse_args()
    if args.sources_only and len(args.revision or [])!=2:
        parser.error('--sources-only preparation requires exactly two revisions')
    if args.action=='prepare' and not (args.design and args.work and args.revision):
        parser.error('prepare requires --design, --work and --revision')
    (prepare if args.action=='prepare' else build)(args)

if __name__=='__main__':main()
