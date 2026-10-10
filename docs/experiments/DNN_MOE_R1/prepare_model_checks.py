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
    parser.add_argument('--export',type=Path)
    parser.add_argument('--bridge',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--batch',type=int,default=2)
    parser.add_argument('--diagnostic',action='store_true')
    parser.add_argument('--input-tensors',type=Path)
    parser.add_argument('--moe',action='store_true')
    parser.add_argument('--moe-checkpoint',type=Path)
    parser.add_argument('--moe-hidden',type=Path)
    args=parser.parse_args()
    if args.input_tensors and not args.diagnostic:parser.error('supplied inputs require a diagnostic')
    if args.moe and (args.export or args.diagnostic or args.input_tensors):
        parser.error('MoE checks require their own bridge and captured inputs')
    if not args.moe and not args.export:parser.error('DNN checks require --export')
    if bool(args.moe_checkpoint)!=bool(args.moe_hidden) or (args.moe_checkpoint and not args.moe):
        parser.error('real MoE checks require --moe, --moe-checkpoint and --moe-hidden')
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
    if args.export:
        inputs[str(args.export.resolve()/'exported_program.pt2')]=sha(args.export/'exported_program.pt2')
    if args.moe_hidden:inputs[str(args.moe_hidden.resolve())]=sha(args.moe_hidden)
    scope='MoE region numerical replay; no decoder or full-model gate' if args.moe else (
        'Intermediate diagnostic only; not G-DNN' if args.diagnostic else 'Complete upstream DNN graph smoke; not G-DNN')
    preparation=dict(source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        inputs=inputs,artifact_id=identity['artifact_id'],scope=scope)
    (root/'preparation.json').write_text(json.dumps(preparation,indent=2)+'\n')
    command=['env','PYTHONPATH='+str(root/'python'),'/root/dm1_work/venv-gpu/bin/python',
        '-m','tilemega.moe.check_generated' if args.moe else 'tilemega.dnn.check_generated',
        '--library',str(library),'--bridge',str(root/'bridge.json')]
    if args.export:command+=['--export',str(args.export.resolve()),'--batch',str(args.batch)]
    if args.moe_checkpoint:
        command+=['--checkpoint',str(args.moe_checkpoint.resolve()),'--hidden',str(args.moe_hidden.resolve())]
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
