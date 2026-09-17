#!/usr/bin/env python3
"""R6 default SASS stamp. Run after the final source/document commit.

The next commit must contain ONLY JOINT2/sass_identity artifacts. Baseline
headers link the baseline host archive: R6 changes the TargetSpec host ABI.
--baseline-tree must be a bad8a0d9 checkout with build/libtilemega.a built using
this machine's CUDA/LLVM/ISL toolchain (WRITEBACK/baseline_host_build_retry.log).
No GPU kernels are launched. NEED_MIB=8192 is checked before compilation.
"""
import argparse,datetime,hashlib,json,shutil,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
BASE='bad8a0d9b17804b73afe00a6d545dcea72cc6cbb'
SCOPE=['include','lib','tools','test','TileMega_skeleton.md','docs/STATUS.md','docs/TODO.md','docs/FINDINGS.md',*[f'docs/experiments/{n}' for n in ('COSTMODEL','JOINT2','WRITEBACK','REBASE','SYMBOLIC','MODELS')]]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--baseline-tree',type=Path,default=Path('/tmp/tilemega-r6-baseline'));ap.add_argument('--provisional',action='store_true');a=ap.parse_args();out=HERE/('sass_provisional' if a.provisional else 'sass_identity');out.mkdir(exist_ok=True);(out/'bin').mkdir(exist_ok=True)
 if not a.provisional:subprocess.run(['git','diff','--exit-code','HEAD','--',*SCOPE],cwd=REPO,check=True)
 free=shutil.disk_usage(out).free//2**20;print(f'DISK NEED_MIB=8192 FREE_MIB={free}',flush=True)
 if free<8192:raise RuntimeError('disk budget')
 baseline=a.baseline_tree;lib=baseline/'build/libtilemega.a'
 if not lib.is_file():raise RuntimeError('build the matching baseline host library first')
 for p in subprocess.check_output(['git','ls-tree','-r','--name-only',BASE,'--','include','lib'],cwd=REPO,text=True).splitlines():
  expected=subprocess.check_output(['git','show',BASE+':'+p],cwd=REPO)
  if (baseline/p).read_bytes()!=expected:raise RuntimeError('baseline source mismatch '+p)
 manifest=dict(baseline=BASE,source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),provisional=a.provisional,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),models={},commands=[],inputs={},baseline_host_sha256=sha(lib),head_host_sha256=sha(REPO/'build-portable/libtilemega.a'))
 for p in subprocess.check_output(['git','ls-files','--',*SCOPE],cwd=REPO,text=True).splitlines():
  if '/sass_identity/' not in p and '/sass_provisional/' not in p:manifest['inputs'][p]=sha(REPO/p)
 # nvcc embeds the archive argument in its fatbin identifier. Keep that
 # argument identical while each compilation links its matching host ABI.
 link_archive=out/'bin/libtilemega.a'
 for model in ('gqa2','mha4'):
  source=REPO/f'docs/experiments/SEQSCAN/raw/src/{model}.cu'
  for arm in ('base','head'):
   link_archive.unlink(missing_ok=True)
   link_archive.symlink_to(lib if arm=='base' else REPO/'build-portable/libtilemega.a')
   cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-O2','-arch=sm_89','-lineinfo','-DTILEMEGA_EVENT_KAPPA=1','-I'+str((baseline if arm=='base' else REPO)/'include'),*['-I'+str(REPO/p) for p in ('third_party/cutlass/include','third_party/cutlass/tools/util/include','third_party/cutlass/test')],str(source),str(link_archive),'-L/usr/local/cuda/lib64','-lcudart','-o',str(out/'bin'/f'{model}_{arm}')]
   manifest['commands'].append(cmd)
   with (out/f'{model}_{arm}.build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
   with (out/f'{model}_{arm}.sass').open('wb') as f:subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(out/'bin'/f'{model}_{arm}')],stdout=f,check=True)
  before=out/f'{model}_base.sass';after=out/f'{model}_head.sass'
  if not all(n in after.read_text() for n in ('tilemega_l1_kernel','tilemega_l2_kernel')):raise RuntimeError('incomplete SASS')
  with (out/f'{model}.diff').open('w') as f:r=subprocess.run(['diff','-u',str(before),str(after)],stdout=f)
  if r.returncode:raise RuntimeError(model+' default SASS differs')
  manifest['models'][model]=dict(sha256=sha(after),bytes=after.stat().st_size,baseline_sha256=sha(before));print('SASS_IDENTITY',model,'PASS',flush=True)
 manifest['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
