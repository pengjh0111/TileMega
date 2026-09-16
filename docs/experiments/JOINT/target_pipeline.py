#!/usr/bin/env python3
"""sm_120 orchestration; local inputs, calibration, projection and measurement."""
import argparse,json,os,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2];PHASE=REPO/'docs/experiments/PHASE'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--phase-only',action='store_true');a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env.update(R5_INPUT_ROOT=str(out/'inputs'),PHASE_TICK_NS='32',JOINT_PHASE_ROOT=str(out/'phase'))
 def run(script,*args):subprocess.run(['python3',str(script),*[str(x) for x in args]],cwd=REPO,env=env,check=True)
 run(HERE/'build_tools.py','--out',out/'tools')
 run(HERE/'bootstrap.py','--out',out/'inputs')
 with (out/'target_calibration.log').open('w') as f:subprocess.run([str(REPO/'build-portable/tools/tilemega-calibrate'),'--dtype','bf16','--out',str(out/'target.json')],cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 for action in ('build','correctness','measure','dump'):run(PHASE/'run.py',action,'--raw',out/'phase','--arch','sm_120')
 with (out/'fork.txt').open('w') as f:subprocess.run(['python3',str(PHASE/'analyze.py'),str(out/'phase')],cwd=REPO,env=env,stdout=f,check=True)
 if a.phase_only:return
 fork=(out/'fork.txt').read_text().strip()
 if 'rule=1' in fork:
  (out/'conditional_degradation.txt').write_text(fork+'\nTarget requires E4, but R5 sm_89 rule3 did not implement it. Continue non-prefetch search; conditional E4 is unresolved, not passed.\n')
 run(HERE/'calibrate_target.py','--out',out/'calibration','--phase',out/'phase')
 pub=json.loads((out/'calibration/publication.json').read_text());env.update(JOINT_TARGET=str(out/'target.json'),JOINT_HOP_TSV=str(out/'calibration/hop_ns.tsv'),JOINT_PUBLICATION_NS=str(pub['publication_ns']),JOINT_WAIT_NS=str(pub['consumer_wait_ns']))
 # A numerical candidate failure stops that cell only. Other cells retain
 # their independent data; no inherited sm_89 Plan or numeric exclusion.
 failures=[]
 for m in ('gqa2','mha4','real'):
  for s in (4,128):
   try:
    for action in ('screen','project','select'):run(HERE/'search.py',action,'--raw',out/'joint','--models',m,'--seqs',s,'--driver',out/'tools/project','--rank-driver',out/'tools/rank')
    for action in ('build','pilot','freeze','measure','correctness','trace_build','trace'):run(HERE/'measure.py',action,'--raw',out/'joint','--arch','sm_120','--models',m,'--seqs',s)
    if m!='real':
     for action in ('prepare','run'):run(HERE/'seqscan.py',action,'--raw',out/'joint','--arch','sm_120','--models',m,'--seqs',s,'--jobs','1','--driver',out/'tools/project')
   except subprocess.CalledProcessError as e:failures.append(dict(model=m,seq=s,command=e.cmd,exit_code=e.returncode))
 (out/'cell_failures.json').write_text(json.dumps(failures,indent=2)+'\n')
 if failures:raise RuntimeError(f'{len(failures)} target cells require local numeric/build repair; evidence retained')
if __name__=='__main__':main()
