#!/usr/bin/env python3
"""Freeze the upstream-model checker and reuse an identified native library."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--export',type=Path,required=True)
    parser.add_argument('--bridge',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--batch',type=int,default=2)
    parser.add_argument('--diagnostic',action='store_true')
    parser.add_argument('--input-tensors',type=Path)
    args=parser.parse_args()
    if args.input_tensors and not args.diagnostic:parser.error('supplied inputs require a diagnostic')
    repo=Path(__file__).resolve().parents[3];root=args.out.resolve()
    library=args.library.resolve();identity_path=Path(str(library)+'.identity.json')
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    identity=json.loads(identity_path.read_text())
    if identity['binary_sha256']!=sha(library):raise ValueError('library identity mismatch')
    root.mkdir(parents=True,exist_ok=False);(root/'queue').mkdir()
    shutil.copytree(repo/'python/tilemega',root/'python/tilemega',
        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(args.bridge,root/'bridge.json')
    if args.input_tensors:shutil.copy2(args.input_tensors,root/'inputs.safetensors')
    inputs={str(path):sha(path) for path in root.rglob('*') if path.is_file()}
    inputs[str(library)]=sha(library);inputs[str(identity_path)]=sha(identity_path)
    inputs[str(args.export.resolve()/'exported_program.pt2')]=sha(args.export/'exported_program.pt2')
    preparation=dict(source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        inputs=inputs,artifact_id=identity['artifact_id'],scope='Intermediate diagnostic only; not G-DNN' if args.diagnostic else 'Complete upstream DNN graph smoke; not G-DNN')
    (root/'preparation.json').write_text(json.dumps(preparation,indent=2)+'\n')
    command=['env','PYTHONPATH='+str(root/'python'),'/root/dm1_work/venv-gpu/bin/python',
        '-m','tilemega.dnn.check_generated','--library',str(library),'--export',str(args.export.resolve()),
        '--bridge',str(root/'bridge.json'),'--batch',str(args.batch)]
    if args.diagnostic:command+=['--diagnostics',str(root/'intermediates.json')]
    if args.input_tensors:command+=['--input-tensors',str(root/'inputs.safetensors')]
    steps=[]
    for name in (['check_00'] if args.diagnostic else
                 [f'check_{i:02}' for i in range(50)]+['memcheck','racecheck']):
        check=command+['--out',str(root/f'{name}.json')]
        sanitizer=not name.startswith('check_')
        if sanitizer:check=['/usr/local/cuda/bin/compute-sanitizer','--tool',name,
            '--target-processes','all','--error-exitcode','86']+check
        seconds=1800 if sanitizer else 600
        steps.append(dict(name=name,command=['flock','/root/r14_work/gpu.lock','timeout',str(seconds)]+check,
            cwd=str(repo),gpu=False,after=[] if name=='check_00' else ['check_00'],
            priority=1 if name=='check_00' else 2 if sanitizer else 10,timeout_s=seconds+600))
    (root/'queue/queue_all.json').write_text(json.dumps(steps,indent=2)+'\n')
    print(json.dumps(dict(root=str(root),steps=len(steps),artifact_id=identity['artifact_id'])),flush=True)


if __name__=='__main__':main()
