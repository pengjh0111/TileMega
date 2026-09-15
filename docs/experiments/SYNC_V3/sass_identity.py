#!/usr/bin/env python3
"""H2: compare complete default SASS after the code/documentation commit.

The evidence commit records its parent, then verification checks that the
child changes evidence only. A commit cannot contain its own content hash.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import datetime

REPO = Path(__file__).resolve().parents[3]
BASE = 'ee905036d2ec0c9dc880df604097981552423e53'
OUT = Path(__file__).resolve().parent / 'sass_identity'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frozen = ['include', 'lib', 'test', 'docs/experiments/TRACE_V2/analyze.py',
              'docs/experiments/TRACE_V2/rebuild_r4.py',
              'docs/experiments/SYNC_V3/sass_identity.py', 'docs/experiments/SYNC_V3/verify.py']
    subprocess.run(['git','diff','--exit-code','HEAD','--',*frozen],cwd=REPO,check=True)
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    OUT.mkdir(exist_ok=True)
    (OUT/'bin').mkdir(exist_ok=True)
    manifest = dict(base=BASE, tested_head=head, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    commands=[], models={}, inputs={})
    for f in (REPO/'include').rglob('*'):
        if f.is_file():
            manifest['inputs'][str(f.relative_to(REPO))] = sha(f)
    manifest['libtilemega_sha256'] = sha(REPO/'build-portable/libtilemega.a')
    with tempfile.TemporaryDirectory(prefix='tilemega-r4-sass-') as tmp:
        temp = Path(tmp)
        archive = temp/'base.tar'
        with archive.open('wb') as f:
            subprocess.run(['git','archive',BASE,'include'],cwd=REPO,stdout=f,check=True)
        subprocess.run(['tar','-xf',str(archive),'-C',str(temp)],check=True)
        for model in ('gqa2','mha4'):
            source = REPO/f'docs/experiments/SEQSCAN/raw/src/{model}.cu'
            manifest['inputs'][str(source.relative_to(REPO))] = sha(source)
            for arm in ('base','head'):
                include = (temp if arm == 'base' else REPO)/'include'
                binary = OUT/'bin'/f'{model}_{arm}'
                cmd = ['/usr/local/cuda/bin/nvcc','-std=c++17','-O2','-arch=sm_89','-lineinfo',
                       '-DTILEMEGA_EVENT_KAPPA=1','-I'+str(include),
                       *['-I'+str(REPO/p) for p in ('third_party/cutlass/include',
                          'third_party/cutlass/tools/util/include','third_party/cutlass/test')],
                       str(source),str(REPO/'build-portable/libtilemega.a'),
                       '-L/usr/local/cuda/lib64','-lcudart','-o',str(binary)]
                manifest['commands'].append(cmd)
                with (OUT/f'{model}_{arm}.build.log').open('w') as log:
                    subprocess.run(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
                with (OUT/f'{model}_{arm}.sass').open('wb') as f:
                    subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(binary)],stdout=f,check=True)
            base, tested = OUT/f'{model}_base.sass', OUT/f'{model}_head.sass'
            for p in (base,tested):
                text = p.read_text()
                if 'tilemega_l1_kernel' not in text or 'tilemega_l2_kernel' not in text:
                    raise RuntimeError('incomplete SASS dump')
            with (OUT/f'{model}.diff').open('w') as f:
                diff = subprocess.run(['diff','-u',str(base),str(tested)],stdout=f)
            manifest['models'][model] = dict(base_sha256=sha(base),head_sha256=sha(tested),
                                             bytes=tested.stat().st_size,identical=diff.returncode == 0)
            print(f'SASS_IDENTITY {model} identical={int(diff.returncode == 0)} bytes={tested.stat().st_size}',flush=True)
            if diff.returncode:
                (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
                return 1
    manifest['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
