#!/usr/bin/env python3
"""R6 sm_120 orchestration: regenerate, recalibrate, solve, then measure.

--mode cost|joint|rebase|models; --out must be fresh.
--local-r5-root optionally reuses a R5 target run made on this exact GPU UUID.
Otherwise a complete local R5 control run is produced first. No checked-in
sm_89 materialized Plan is used as an input to target measurements.
"""
import argparse,csv,hashlib,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
def existing_failures(out):
 # New search candidates can be excluded. A failed existing control cannot.
 logs=[]
 for pattern in ('joint/*/pilot/control/*.log','joint/*/pilot/champion/*.log',
                 'joint/*/measure/control/*.log','joint/*/measure/champion/*.log',
                 'rebase/*/measure/*__full/*.log'):
  logs.extend(p for p in out.glob(pattern) if 'RESULT status=FAIL' in p.read_text())
 return sorted(map(str,logs))
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--mode',choices=['cost','joint','rebase','models'],required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--local-r5-root',type=Path);ap.add_argument('--capacity',type=int,default=18);a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
 uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()[0].strip()
 base=(a.local_r5_root or out/'local_r5').resolve();env=os.environ.copy();env.update(R5_INPUT_ROOT=str(base/'inputs'),PHASE_TICK_NS='32')
 def run(script,*args):
  try:subprocess.run([sys.executable,str(REPO/'docs/experiments'/script),*map(str,args)],cwd=REPO,env=env,check=True)
  except subprocess.CalledProcessError:
   failures=existing_failures(out)
   if failures:
    (out/'global_stop.json').write_text(json.dumps(dict(rule='R6 9.4 existing configuration correctness regression',logs=failures),indent=2)+'\n')
    raise RuntimeError('global correctness stop; see global_stop.json')
   raise
 run('JOINT2/build_tools.py','--out',out/'tools')
 env['R6_REBASE_DRIVER']=str(out/'tools/rebase_generate')
 if a.local_r5_root:
  witness=json.loads((base/'r6_local_device.json').read_text())
  if witness['uuid']!=uuid:raise RuntimeError('local baseline was made on a different GPU; regenerate it')
 else:
  run('JOINT/target_pipeline.py','--out',base)
  (base/'r6_local_device.json').write_text(json.dumps(dict(uuid=uuid,scope='locally regenerated R5 controls and calibration'),indent=2)+'\n')
 run('COSTMODEL/uniform_sources.py','--input-root',base/'inputs','--out',out/'calibration_sources')
 for action in ('build','correctness','dump'):run('COSTMODEL/kloop.py',action,'--raw',out/'kloop','--arch','sm_120','--champion-root',base/'joint','--calibration-sources',out/'calibration_sources/sources.json')
 run('COSTMODEL/analyze_kloop.py','--raw',out/'kloop')
 run('COSTMODEL/fit_body.py','--raw',out/'kloop','--target',base/'target.json','--out',out/'body_fit')
 observations=out/'body_fit/observations.tsv';geometries=sorted({tuple(int(r[k]) for k in ('tile_m','tile_n','tile_k','stages')) for r in csv.DictReader(observations.open(),delimiter='\t')})
 domain=out/'search_domain.json';domain.write_text(json.dumps(dict(scope='R6 shapes measured on this target',source=str(observations),source_sha256=hashlib.sha256(observations.read_bytes()).hexdigest(),geometries=[dict(zip(('tile_m','tile_n','tile_k','stages'),g)) for g in geometries]),indent=2)+'\n')
 target=json.loads((out/'body_fit/target.json').read_text());pub_path=base/'calibration/publication.json';pub=json.loads(pub_path.read_text())
 rates=target.setdefault('event_calibration_by_dtype',{}).setdefault('bf16',{})
 for field,key,unit in [('task_publication','publication_ns','ns/publishing_runtime_task'),('task_wait','consumer_wait_ns','ns/waiting_runtime_task')]:
  rates[field]=dict(ns=pub[key],reason='measured',unit=unit,source=str(pub_path),source_sha256=hashlib.sha256(pub_path.read_bytes()).hexdigest())
 target_path=out/'target.json';target_path.write_text(json.dumps(target,indent=2)+'\n')
 (out/'device.json').write_text(json.dumps(dict(uuid=uuid,base=str(base),mode=a.mode),indent=2)+'\n')
 manifest=out/'replay_manifest.tsv'
 with manifest.open('w') as f:
  w=csv.DictWriter(f,fieldnames=['dump','window','source'],delimiter='\t');w.writeheader()
  for m in ('gqa2','mha4','real'):
   for s in (4,128):
    c=base/'joint'/f'{m}_s{s}';choice=json.loads((c/'choice.json').read_text());arm=choice['arm']
    w.writerow(dict(dump=str(c/'trace'/arm/'dump'),window=1,source=choice['choice']['source']))
 with (out/'replay.log').open('w') as f:
  subprocess.run([str(out/'tools/replay'),'/',str(manifest),str(out/'replay.tsv'),str(pub['publication_ns']),str(pub['consumer_wait_ns']),str(pub['visibility_ns'])],cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 if a.mode=='cost':return
 if a.mode=='models':
  model=out/'llama_covered';run('MODELS/export_covered.py','--seq','4','--out',model)
  subprocess.run([str(REPO/'build-portable/tools/tilemega-compile'),str(model/'exported_program.pt2'),str(model/'auto.cu'),'--solve',str(target_path),'--seq','4','--past','3','--search-capacity',str(a.capacity),'--search-domain',str(domain),'--hop-curve',str(base/'calibration/hop_ns.tsv'),'--dump-cg',str(model/'auto.mlir')],cwd=REPO,env=env,check=True)
  run('MODELS/run_covered.py','--root',model,'--fixture',model/'fixture','--arch','sm_120')
  return
 common=['--raw',out/'joint','--target',target_path,'--hop',base/'calibration/hop_ns.tsv','--arch','sm_120','--champion-root',base/'joint','--capacity',a.capacity,'--search-domain',domain]
 failures=[];successful=[]
 for m in ('gqa2','mha4','real'):
  for seq in (4,128):
   try:
    for action in ('search','build','pilot','freeze','measure','correctness','trace_build','trace'):run('JOINT2/run.py',action,*common,'--models',m,'--seqs',seq)
    successful.append((m,seq))
    run('JOINT2/analyze.py','--raw',out/'joint','--out',out/'analysis','--models',m,'--seqs',seq)
   except subprocess.CalledProcessError as e:failures.append(dict(model=m,seq=seq,command=e.cmd,exit_code=e.returncode))
 (out/'cell_failures.json').write_text(json.dumps(failures,indent=2)+'\n')
 # Re-materialize the selected reference families at each required past value.
 for m,seq in successful:
  if m=='real':continue
  try:
   for past in (0,512):
    fixture=base/'inputs/fixture'/f'{m}_s{seq}_p{past}'
    if not fixture.exists():
     if m=='gqa2':run('E2E/prepare_e2e.py','--vh-raw',base/'inputs/export/gqa2','--out',fixture,'--seq',seq,'--past',past)
     else:run('P3_GENERALIZATION/prepare_fixture.py','--repo',REPO,'--program',base/'inputs/export/mha4/exported_program.pt2','--out',fixture,'--seq',seq,'--past',past)
   for action in ('generate','build','run'):
    run('JOINT2/seqscan.py',action,'--joint',out/'joint','--out',out/'seqscan','--models',m,'--seqs',seq,'--driver',out/'tools/seqscan','--target',target_path,'--hop',base/'calibration/hop_ns.tsv','--arch','sm_120','--input-root',base/'inputs')
  except subprocess.CalledProcessError as e:failures.append(dict(model=m,seq=seq,scope='selected_seqscan',command=e.cmd,exit_code=e.returncode))
 # The local CG is the only input to symbolic evaluation: this never imports
 # the 4090's worker/slot table. Templates which cannot prove the target domain
 # remain explicit failures, rather than replacing the measured winner.
 if (out/'tools/parametric').exists() and ('gqa2',4) in successful:
  c=out/'joint/gqa2_s4';choice=json.loads((c/'choice.json').read_text())['arm'];rank=choice[3:]
  try:
   with (out/'symbolic.log').open('w') as f:
    cg=c/f'auto.cu.top{rank}.mlir';symbolic=out/'symbolic';carried=symbolic/'parametric.mlir'
    for cmd in ([str(out/'tools/parametric'),str(cg),str(symbolic)],
                [str(out/'tools/symbolic_cg'),str(cg),str(symbolic),str(carried)],
                [str(out/'tools/cross_grid'),str(carried),str(target_path),str(base/'calibration/hop_ns.tsv'),str(symbolic/'cross_grid')]):
     subprocess.run(cmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  except subprocess.CalledProcessError as e:failures.append(dict(scope='symbolic',command=e.cmd,exit_code=e.returncode))
 else:(out/'symbolic_unavailable.txt').write_text('Build SYMBOLIC/r6_templates.cpp locally; no cross-grid proof claimed without it.\n')
 if a.mode=='rebase':
  env['R6_REBASE_TARGET']=str(target_path);env['R6_REBASE_HOP']=str(base/'calibration/hop_ns.tsv')
  for m,seq in successful:
   try:
    for action in ('generate','build','measure','trace_build','trace'):run('REBASE/run.py',action,'--joint',out/'joint','--out',out/'rebase','--models',m,'--seqs',seq,'--arch','sm_120')
    run('REBASE/analyze.py','--raw',out/'rebase','--out',out/'rebase_analysis','--models',m,'--seqs',seq)
   except subprocess.CalledProcessError as e:failures.append(dict(model=m,seq=seq,scope='rebase',command=e.cmd,exit_code=e.returncode))
 (out/'cell_failures.json').write_text(json.dumps(failures,indent=2)+'\n')
 if failures:raise RuntimeError('target cells failed; all independent cells were attempted; see cell_failures.json')
if __name__=='__main__':main()
