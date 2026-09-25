#!/usr/bin/env python3
"""Wait for each s4 winner, then run occupancy-checked ablations and 50 processes.

Runs independently of the matrix coordinator. Products are ablations/<cell>/
{template,eft,wide}, stages/llama_s4_S{2,3,4}, colocation/<model>. Each contains
raw commands, fresh process logs and SASS. Failed prerequisites are recorded;
they never turn into an implicit pass. No existing measurement is overwritten.
"""
import csv,json,pathlib,re,subprocess,sys,time
E=pathlib.Path(__file__).resolve().parent;ROOT=E.parents[2]

def run(args,log):
 log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('x') as stream:
  code=subprocess.call([sys.executable,*map(str,args)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
 if code:raise RuntimeError(f'command failed ({code}), see {log}')

def fixed(model,key,arm,pure=False,width='8',stages=None):
 out=(E/'stages'/f'llama_s4_S{stages}') if stages else E/'ablations'/f'{model}_s4'/arm
 if (out/'result.json').exists():return
 out.mkdir(parents=True,exist_ok=True)
 flags=['--width',width]+(['--pure'] if pure else [])+(['--stages',str(stages)] if stages else [])
 # The first source exists solely for the complete-kernel occupancy query.
 run([E/'run_fixed.py',model,'4',key,out/'probe',*flags],out/'materialize_probe.log')
 run([E/'query_source.py',out/'probe/selected.cu',out/'occupancy'],out/'query_runner.log')
 actual=json.loads((out/'occupancy/query.log').read_text())['resident']
 if actual<1:raise RuntimeError('no resident CTA: '+str(out))
 run([E/'run_fixed.py',model,'4',key,out/'materialized',*flags,'--actual',str(actual)],out/'materialize_final.log')
 fixture=json.loads((E/'floor'/f'{model}_s4.command.json').read_text())[5]
 run([E/'measure.py','--source',out/'materialized/selected.cu','--fixture',fixture,'--out',out],out/'measure_runner.log')
 (out/'result.json').write_text(json.dumps(dict(actual_limit=actual,key=key,arm=arm,stages=stages))+'\n')

def finish(model):
 out=E/'matrix'/f'{model}_s4';result=json.loads((out/'selected.cu.measurements.json').read_text())
 if not result['winner']:raise RuntimeError('no valid matrix winner '+model)
 candidate=result['winner']['candidate'];key=candidate['key']
 # Selection is checked independently from raw logs by verify.py.
 for name,pure,width in [('template',True,'8'),('eft',False,'8'),('wide',False,'W')]:
  fixed(model,key,name,pure,width)
 if model=='llama':
  for stages in (2,3,4):fixed(model,key,'stages',stages=stages)
 cg=pathlib.Path(candidate['cg']).read_text()
 omitted=re.search(r'tmexec.sync_omitted_event_waits = (\d+)',cg)
 if int(candidate['kappa'])==1 and omitted and int(omitted[1])>0:
  source=pathlib.Path(candidate['source'])
 else:
  # A kappa>1 winner makes no omission claim. Exercise the same geometry at
  # kappa=1 explicitly, retaining this distinction in the synchronization record.
  sync_key=re.sub(r'kappa=\d+','kappa=1',key)
  fixed(model,sync_key,'sync_template',pure=True)
  source=E/'ablations'/f'{model}_s4'/'sync_template/materialized/selected.cu'
  cg=source.with_suffix('.mlir').read_text()
  omitted=re.search(r'tmexec.sync_omitted_event_waits = (\d+)',cg)
  if not omitted or int(omitted[1])<=0:raise RuntimeError('no actual event-wait omission to validate')
 destination=E/'colocation'/model;destination.mkdir(parents=True,exist_ok=True)
 fixture=json.loads((E/'floor'/f'{model}_s4.command.json').read_text())[5]
 run([E/'measure.py','--source',source,'--fixture',fixture,'--out',destination,'--processes','50'],destination/'runner.log')
 (destination/'selection.json').write_text(json.dumps(dict(matrix_key=key,source=str(source),same_as_winner=str(source)==candidate['source'],actual_omitted_waits=int(omitted[1])))+'\n')

def main():
 pending={'llama','qwen3'};state={}
 while pending:
  progressed=False
  for model in sorted(pending):
   out=E/'matrix'/f'{model}_s4'
   if not (out/'selected.cu.measurements.json').exists():
    status=out/'solve.exit.json'
    if status.exists() and json.loads(status.read_text())['exit']:
     state[model]='blocked: matrix solve failed';pending.remove(model);progressed=True
    continue
   try:finish(model);state[model]='complete'
   except Exception as error:state[model]='failed: '+str(error)
   pending.remove(model);progressed=True
   print(json.dumps(state),flush=True)
  (E/'postmatrix_progress.json').write_text(json.dumps(dict(completed=state,pending=sorted(pending)),indent=2)+'\n')
  if pending and not progressed:time.sleep(30)
 if any(x!='complete' for x in state.values()):raise SystemExit(1)
if __name__=='__main__':main()
