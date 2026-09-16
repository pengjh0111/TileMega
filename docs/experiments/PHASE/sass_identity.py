#!/usr/bin/env python3
"""R5 full default SASS identity; final artifact commit must directly follow tested HEAD."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
BASE='a001b0ac53197551fe1e5e115528a9ea40d24783'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--provisional',action='store_true');a=ap.parse_args()
    out=HERE/('sass_provisional' if a.provisional else 'sass_identity');out.mkdir(exist_ok=True)
    (out/'bin').mkdir(exist_ok=True)
    if not a.provisional:
        subprocess.run(['git','diff','--exit-code','HEAD','--','include','lib','test','docs/experiments/PHASE','docs/experiments/JOINT','docs/experiments/TRACE_V2/analyze.py','docs/STATUS.md','docs/TODO.md','docs/FINDINGS.md'],cwd=REPO,check=True)
    free=shutil.disk_usage(out).free//2**20;print(f'DISK NEED_MIB=4096 FREE_MIB={free}',flush=True)
    if free<4096:raise RuntimeError('disk budget')
    manifest=dict(base=BASE,tested_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                  provisional=a.provisional,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),models={},commands=[],inputs={})
    files=subprocess.check_output(['git','ls-files','--','include','lib','test','docs/experiments/PHASE','docs/experiments/JOINT','docs/experiments/TRACE_V2/analyze.py','docs/STATUS.md','docs/TODO.md','docs/FINDINGS.md'],cwd=REPO,text=True).splitlines()
    for p in files:
        if '/sass_identity/' not in p and '/sass_provisional/' not in p:manifest['inputs'][p]=sha(REPO/p)
    with tempfile.TemporaryDirectory(prefix='tilemega-r5-sass-') as tmp:
        tmp=Path(tmp);archive=tmp/'base.tar'
        with archive.open('wb') as f:subprocess.run(['git','archive',BASE,'include'],cwd=REPO,stdout=f,check=True)
        subprocess.run(['tar','-xf',str(archive),'-C',str(tmp)],check=True)
        for model in ('gqa2','mha4'):
            source=REPO/f'docs/experiments/SEQSCAN/raw/src/{model}.cu'
            for arm in ('base','head'):
                cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-O2','-arch=sm_89','-lineinfo','-DTILEMEGA_EVENT_KAPPA=1',
                     '-I'+str((tmp if arm=='base' else REPO)/'include'),
                     *['-I'+str(REPO/p) for p in ('third_party/cutlass/include','third_party/cutlass/tools/util/include','third_party/cutlass/test')],
                     str(source),str(REPO/'build-portable/libtilemega.a'),'-L/usr/local/cuda/lib64','-lcudart','-o',str(out/'bin'/f'{model}_{arm}')]
                manifest['commands'].append(cmd)
                with (out/f'{model}_{arm}.build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
                with (out/f'{model}_{arm}.sass').open('wb') as f:subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(out/'bin'/f'{model}_{arm}')],stdout=f,check=True)
            base,head=out/f'{model}_base.sass',out/f'{model}_head.sass'
            if not all(n in head.read_text() for n in ('tilemega_l1_kernel','tilemega_l2_kernel')):raise ValueError('incomplete SASS')
            with (out/f'{model}.diff').open('w') as f:r=subprocess.run(['diff','-u',str(base),str(head)],stdout=f)
            manifest['models'][model]=dict(base_sha256=sha(base),head_sha256=sha(head),bytes=head.stat().st_size,identical=r.returncode==0)
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            if r.returncode:raise ValueError(f'SASS differs: {model}')
            print('SASS_IDENTITY',model,'PASS',flush=True)
    manifest['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
