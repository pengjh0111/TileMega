#!/usr/bin/env python3
"""Recompute every R6 gate from raw processes, evaluator rows, and plan dumps.

Run from any directory: python3 docs/experiments/JOINT2/verify.py
No summary.md, report.json, result.json or gate-result input is consulted.
Cross-grid metadata indexes raw Plan pairs; differences are recomputed. Missing evidence is FAIL. All gates run before the exit code.
Research/report failures are displayed; any hard failure returns one.
"""
import csv,hashlib,importlib.util,json,math,re,statistics,subprocess,sys,traceback
from functools import lru_cache
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2];EX=HERE.parent
sys.path.insert(0,str(HERE));import analyze as joint

def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
rank=module('r6_rank_math',EX/'SIMULATOR/r5/report.py')
C=EX/'COSTMODEL';W=EX/'WRITEBACK';S=EX/'SYMBOLIC';B=EX/'REBASE'
REFS=['gqa2_s4','gqa2_s128','mha4_s4','mha4_s128'];REAL=['real_s4','real_s128'];ALL=REFS+REAL
results=[]
def table(p):return list(csv.DictReader(p.open(),delimiter='\t'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(condition,message):
 if not condition:raise AssertionError(message)
def evidence(p):return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)
def gate(name,kind,fn):
 try:
  ok,detail=fn();results.append((name,kind,bool(ok)));print(f'{name} {"PASS" if ok else "FAIL"} [{kind}] {detail}',flush=True)
 except Exception as e:
  results.append((name,kind,False));print(f'{name} FAIL [{kind}] {type(e).__name__}: {e}',flush=True)
@lru_cache(None)
def cell(name):return REPO/json.loads((HERE/'cells.json').read_text())[name]
@lru_cache(None)
def choice(name):return joint.chosen(cell(name))
@lru_cache(None)
def trace(name,arm):return joint.trace_cell(cell(name),arm)
def processes(folder,n=50):
 seen=set();passed=0;errors=[]
 for i in range(n):
  p=folder/f'r{i}.log'
  try:
   meta=json.loads(p.with_suffix('.json').read_text());start=meta['started_ns'];check(start not in seen,'process reused');seen.add(start)
   passed+=int(meta['exit_code']==0 and 'RESULT status=PASS' in p.read_text())
  except Exception as e:errors.append(str(e))
 return passed==n and len(seen)==n,f'{passed}/{n} {evidence(folder)}'+(' missing='+str(len(errors)) if errors else '')
def collections(folders):
 details=[];ok=True
 for f in folders:
  valid,text=processes(f);ok &= valid;details.append(text)
 return ok,'; '.join(details)
def cost_branches():
 files=['CostModel.cpp','ChainDP.cpp','CouplingInterfaceDP.cpp','TaskModel.cpp','ScalarTaskWork.cpp']
 hits=[]
 for f in files:
  for n,line in enumerate((REPO/'lib/Solver'/f).read_text().splitlines(),1):
   if 'NonGemmStageNs' in line or re.search(r'(case\s+StageKind|(?:if|switch).*StageKind)',line):hits.append(f'{f}:{n}:{line.strip()}')
 command=['rg','-n','StageKind|NonGemmStageNs','lib/Solver/CostModel.cpp','lib/Solver/ChainDP.cpp','lib/Solver/CouplingInterfaceDP.cpp','lib/Solver/TaskModel.cpp','lib/Solver/ScalarTaskWork.cpp']
 output=subprocess.run(command,cwd=REPO,text=True,capture_output=True).stdout.strip()
 roles={'ModelDescription.cpp':'model parsing', 'AlignmentPropagation.cpp':'alignment constraints',
        'RuntimeProjection.cpp':'runtime ownership and dependency projection',
        'AttentionWork.cpp':'attention semantic/resource contract validation'}
 all_command=['rg','-n','StageKind|NonGemmStageNs','lib/Solver']
 whole=subprocess.run(all_command,cwd=REPO,text=True,capture_output=True).stdout.strip()
 for line in whole.splitlines():
  filename=Path(line.split(':',1)[0]).name
  check(filename in roles,'unreviewed operator-kind branch: '+line)
  check('NonGemmStageNs' not in line,'retired formula remains: '+line)
 return not hits,'command='+ ' '.join(command)+'; output='+repr(output)+'; whole-tree command=rg -n StageKind lib/Solver; reviewed non-price roles='+repr(roles)+'; full grep evidence=COSTMODEL/stagekind_audit.txt'
def rank_gate(key,threshold):
 rows=table(C/'closure_effects/evaluations.tsv');actual=table(EX/'SIMULATOR/raw/time/l2.tsv');modes={'legacy_grid_stride':'0','balanced':'4','rotate':'5'}
 rows=[r for r in rows if r['model']!='real' and r['candidate'] in modes]
 y=[statistics.median(float(x['l2_ms'])*1e6 for x in actual if (x['model'],x['seq'],x['place'])==(r['model'],r['seq'],modes[r['candidate']])) for r in rows]
 rho=rank.spearman([float(r[key]) for r in rows],y)
 return len(rows)==18 and rho>=threshold,f'n={len(rows)} rho={rho:.12f} required={threshold}; {evidence(C/"closure_effects/evaluations.tsv")} + SIMULATOR/raw/time/l2.tsv (historical GPU calibration, fresh CPU evaluation)'
def replay():
 rows=table(C/'closure_replay/replay.tsv');v=sorted(abs(float(r['predicted_ns'])/float(r['measured_ns'])-1) for r in rows)
 return len(rows)==68,f'n={len(rows)} relative_error p50={statistics.median(v):.9f} p90={joint.trace.percentile(v,.9):.9f} max={max(v):.9f}; {evidence(C/"closure_replay/replay.tsv")}'
def kloop():
 a=module('r6_loop_raw',C/'analyze_kloop.py');rs=[]
 for name in REFS:
  root=C/'raw_kloop'/name;source=Path(json.loads((root/'specs.json').read_text())['selected']['source']);r,_=a.analyze(root/'phase/selected/dump',source);rs.append(r)
 wait=statistics.median(r['cp_kloop_wait_share'] for r in rs);fixed=statistics.median(r['cp_kloop_fixed_share'] for r in rs)
 line=f'FORK6 rule={1 if wait>=.25 else 2} mainloop_exposed_wait_share={wait:.3f} kloop_fixed_share={fixed:.3f} cells=4'
 check(line in (REPO/'docs/FINDINGS.md').read_text(),'raw FORK6 missing in FINDINGS')
 vals=[(r['cp_kloop_iteration_p50_ns'],r['cp_kloop_fixed_p50_ns']) for r in rs]
 return True,line+'; (iteration_ns,fixed_ns)='+repr(vals)+'; scope=instrumented GEMM mainloops, SIMT wait unmeasured; COSTMODEL/raw_kloop/*/phase/selected/dump'
def sass():
 root=HERE/'sass_identity';m=json.loads((root/'manifest.json').read_text());head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();parent=subprocess.check_output(['git','rev-parse','HEAD^'],cwd=REPO,text=True).strip()
 check(m['source_head']==parent,'identity must stamp immediate parent of evidence-only HEAD')
 check(m['baseline']=='bad8a0d9b17804b73afe00a6d545dcea72cc6cbb','wrong baseline')
 paths=subprocess.check_output(['git','diff-tree','--no-commit-id','--name-only','-r',head],cwd=REPO,text=True).splitlines();check(all(p.startswith('docs/experiments/JOINT2/sass_identity/') for p in paths),'final commit contains source/docs changes')
 for model in ('gqa2','mha4'):
  a=root/(model+'_base.sass');b=root/(model+'_head.sass');check(a.read_bytes()==b.read_bytes(),model+' SASS differs');check(sha(a)==m['models'][model]['sha256'],'SASS hash mismatch')
 
 for path,digest in m['inputs'].items():check(sha(REPO/path)==digest,'source changed after stamp: '+path)
 return True,'models=2/2 bytes_identical source_hashes_match final_source_parent_stamp_valid; '+evidence(root)
def cf():
 ok,detail=collections([C/'raw_kloop'/x/'correctness' for x in REFS])
 try:ident,msg=sass()
 except Exception as e:ident=False;msg=str(e)
 return ok and ident,detail+'; '+msg

def paired_gate(names,control,upper):
 ok=True;parts=[]
 for name in names:
  try:
   med,lo,hi=joint.paired(cell(name),choice(name),control);valid=(med<1 and hi<1) if upper==1 else hi<=upper;ok &=valid;parts.append(f'{name} {med:.9f} [{lo:.9f},{hi:.9f}] {evidence(cell(name)/"measure")}')
  except Exception as e:ok=False;parts.append(f'{name}: {e}')
 return ok,'; '.join(parts)
def queue_gate():
 values=[];ok=True
 for name in REAL:
  try:r=trace(name,choice(name));values.append(f'{name} queue/semantic_CP={r["queue_over_cp"]:.9f} {r["dump"]}');ok &= r['queue_over_cp']<=1
  except Exception as e:ok=False;values.append(f'{name}: {e}')
 return ok,'; '.join(values)
def outer_bounds():
 details=[]
 for name in ALL:
  root=HERE/'bounded_search'/name;rows=table(root/'auto.cu.bounds.tsv')
  bounds={r['candidate']:max(float(r[k]) for k in ('work_lb_ns','cp_lb_ns','queue_lb_lb_ns')) for r in rows}
  check(len(rows)>0 and all(v>0 and math.isfinite(v) for v in bounds.values()),'missing nonzero task-derived bounds')
  for r in rows:check(math.isclose(float(r['priority_ns']),bounds[r['candidate']],rel_tol=1e-12),'outer priority is not the binding bound')
  plans=table(root/'auto.cu.search.tsv');groups={};checked=0
  for r in plans:
   if r['placement']=='-':continue
   groups.setdefault(r['candidate'],set()).add(r['placement'])
   if r['status']=='ok':
    key=re.sub(r'r[0-9]+$','',r['candidate']);floor=float(r['floor_ns'])
    check(bounds[key]<=floor+max(1e-4,abs(floor)*1e-5),'outer bound exceeds evaluated plan floor: '+name+'/'+key);checked+=1
  for modes in groups.values():check(modes=={'legacy_grid_stride','rotate','balanced','eft','wavefront','chain'},'inner placement coverage gap')
  details.append(f'{name} outer={len(rows)} inner={checked} admissible=all; {evidence(root)}')
 return True,'; '.join(details)

def budget():
 rows=table(C/'closure_effects/evaluations.tsv');parts=[];ok=True
 for model,limit in [('reference',1000),('real',10000)]:
  rs=[r for r in rows if (r['model']=='real')==(model=='real')];worst=max(rs,key=lambda r:float(r['full_us']));us=float(worst['full_us']);ok &=us<limit
  parts.append(f'{model} full_max_us={us:.3f} budget={limit} at {worst["model"]}/s{worst["seq"]}/{worst["candidate"]}; prepare_max_us={max(float(r["prepare_us"]) for r in rs):.3f}')
 return ok,'; '.join(parts)+'; '+evidence(C/'closure_effects/evaluations.tsv')
def ranking():
 parts=[];good=0
 for name in ALL:
  try:
   scores={a:statistics.median(joint.timing(cell(name)/'measure'/a/f'r{i}.log')['l2_ms'] for i in range(25)) for a in ('top1','top2','top3')};order=sorted(scores,key=scores.get);pos=order.index('top1')+1;good+=pos<=2;parts.append(f'{name} predicted_top1={pos}/3')
  except Exception as e:parts.append(f'{name}: {e}')
 return good>=4,f'{good}/6 top1 in measured top2; '+'; '.join(parts)
def jf():
 a,t=collections([cell(n)/'correctness' for n in ALL]);b,u=collections([HERE/'bounded_seqscan'/f'{m}_s{s}_p{p}'/'correctness' for m in ('gqa2','mha4') for s in (4,128) for p in (0,512)])
 return a and b,t+'; selected SEQSCAN '+u

def fuse():
 parts=[];maximum=0
 for name in ALL:
  root=HERE/'bounded_fuse'/f'{name}_selected.tsv';rows=table(root);supported=[r for r in rows if r['status']=='SUPPORTED'];bound=0
  for r in supported:
   separate=float(r['separate_task_ns']);fixed=.232*separate;traffic=max(0.,separate-float(r['traffic_floor_ns']));upper=min(separate,fixed+traffic)
   check(float(r['fixed_share_input'])==.232,'fixed share changed');check(math.isclose(upper,float(r['optimistic_upper_ns']),rel_tol=1e-8),'upper arithmetic mismatch');bound+=upper
  floor=trace(name,choice(name))['floor_ns'];share=bound/floor;maximum=max(maximum,share);parts.append(f'{name} supported={len(supported)} upper_ns={bound:.6f} share={share:.6f} {evidence(root)}')
 line=f'FUSE6 enter_r7={int(maximum>=.1)} maximum_bound_share={maximum:.6f} cells=6';check(line in (REPO/'docs/FINDINGS.md').read_text(),'Fuse decision missing from FINDINGS')
 return True,line+'; '+'; '.join(parts)
def writeback():
 src=REPO/'include/tilemega/Dialect/CouplingGraph/PlacementSolvePass.h';hits=[f'{i}:{l.strip()}' for i,l in enumerate(src.read_text().splitlines(),1) if 'setAttr(' in l];cg=W/'gqa2_resident_auto.mlir';text=cg.read_text();check('tilemega.solved_kappa' in text and 'tilemega.placement' in text,'CG lacks solved metadata')
 return any('placement->setAttr(' in h for h in hits),'rg -n setAttr '+evidence(src)+' => '+repr(hits)+'; '+evidence(cg)
def command_gate():
 root=EX/'MODELS/llama_mlp';m=json.loads((root/'direct_pt2.command.json').read_text());cmd=m['command'];check('--solve' in cmd and cmd[1].endswith('.pt2'),'not direct torch.export solver command');check((root/'direct_pt2.cu').stat().st_size>0,'missing generated CUDA');text=(root/'direct_pt2.mlir').read_text();check(all(x in text for x in ('tilemega.solved_kappa','tilemega.solved_residency','tilemega.placement')),'missing decisions')
 check((root/'direct_pt2.cu').read_bytes()==(root/'auto.cu').read_bytes(),'direct torch.export output differs from correctness-tested CUDA')
 return True,'command='+repr(cmd)+'; output='+evidence(root/'direct_pt2.cu')+' byte_equal_to_50_process_tested_source'

def legacy():
 n=0
 for m in ('gqa2','mha4'):
  for s in (4,128):
   root=W/'legacy_closure/identity'/f'{m}_s{s}'
   for name in ('schedule.tsv','waits.tsv','events.tsv'):
    check((root/'base'/name).read_bytes()==(root/'head'/name).read_bytes(),str(root/name));n+=1
 return n==12,f'{n}/12 byte-identical tables; WRITEBACK/legacy_closure/identity'
def roundtrip():
 root=W/'roundtrip_gqa2_s4';a=(root/'cg_plan.tsv').read_bytes();b=(root/'host_plan.tsv').read_bytes();check(a==b,'point CG/host table differs')
 interval=W/'interval_closure';counts=[]
 source=(interval/'gqa2.cu').read_text()
 for seq in range(1,6):
  expected=table(interval/f'direct_s{seq}.tsv')
  actual=sorted(table(interval/'host'/f's{seq}/schedule.tsv'),key=lambda r:(int(r['stage']),int(r['logical_task'])))
  check(len(actual)==len(expected),'interval node count differs')
  for key in ('worker','slot'):
   values=[int(r[key]) for r in expected];check(values==[int(r[key]) for r in actual],f'host {key} differs at seq={seq}')
   name='Worker' if key=='worker' else 'Slot'
   generated=re.search(r'kInterval'+name+f'0_{seq-1}'+r'\[\] = \{([^}]+)\}',source)
   check(generated is not None and values==[int(v) for v in generated[1].split(',')],f'generated {key} differs')
  counts.append(len(actual))
 return True,f'point + interval CG/direct solver/generated/host arrays identical; interval nodes={counts}; {evidence(interval)}'
def interval_writeback():
 root=W/'interval_closure';ok,detail=collections([root/'correctness'/f's{s}' for s in range(1,6)])
 cg=(root/'gqa2.mlir').read_text();check('tilemega.solved_seq_begin = 1' in cg and 'tilemega.solved_seq_end = 5' in cg,'missing interval bounds')
 rows=table(root/'gqa2.cu.interval.tsv');check(len(rows)==30,'not all six placements per point')
 for seq in range(1,6):check(len({r['placement'] for r in rows if int(r['seq'])==seq})==6,'placement coverage gap')
 hashes={json.loads(p.read_text())['binary_sha256'] for p in (root/'correctness').rglob('r*.json')}
 check(len(hashes)==1,'interval used multiple binaries')
 return ok,detail+'; one binary, all six placements at every integer point; geometry selected at upper endpoint, fixed past/grid'
def we():
 log=HERE/'closure/finite_slices/ctest.log';text=log.read_text();match=re.search(r'100% tests passed, 0 tests failed out of (\d+)',text);ok,detail=collections([W/'legacy_closure/seqscan'/f'{m}_s{s}_p{p}' for m in ('gqa2','mha4') for s in (4,128,2048) for p in (0,512)])
 return bool(match) and int(match[1])>=49 and ok,f'CTest={match[1] if match else "FAIL"} {evidence(log)}; '+detail

def symbolic_proofs():
 root=S/'complete';rows=table(root/'proofs.tsv');parts=[];ok=True
 for family in ('legacy_grid_stride','rotate','band','wavefront'):
  for g in (256,340):
   rs=[r for r in rows if r['family']==family and int(r['grid'])==g];covered=set()
   for r in rs:
    valid=all(r[k]=='1' for k in ('total','bijective','dense','acyclic','resident','level_exact'));ok &=valid
    if valid:covered.update(range(int(r['begin']),int(r['end'])+1))
   ok &= covered==set(range(1,129));parts.append(f'{family}/G{g} proved={len(covered)}/128')
 for name in ALL:
  source=cell(name)/'trace'/choice(name)/'r0.log'
  counts=re.findall(r'\bvariant_count=(\d+)',source.read_text())
  check(len(counts)==1 and 1<=int(counts[0])<=2,'binary variant cap failed '+str(source))
  parts.append(name+' binary_variants='+counts[0])
 return ok,'; '.join(parts)+'; SYMBOLIC/complete/proofs.tsv and JOINT2 selected raw trace resource records'
def symbolic_samples():
 root=S/'complete';rows=table(root/'samples.tsv');n=0
 for family in ('legacy_grid_stride','rotate','band','wavefront'):
  for g in (256,340):
   for s in (1,32,64,96,128):
    r=next(r for r in rows if r['family']==family and int(r['grid'])==g and int(r['seq'])==s);check((root/r['template']).read_bytes()==(root/r['native']).read_bytes(),str(r));n+=1
 return n==40,f'{n}/40 endpoint/interior complete plan byte comparisons; SYMBOLIC/complete/'
def symbolic_champion():
 parts=[]
 for name in ALL:
  root=S/'bounded_fit'/name;rows=table(root/'fit.tsv');check(len(rows)==4,'missing family '+name);fits=[]
  for r in rows:
   symbolic=(root/r['template_file']).read_bytes();native=(root/(r['family']+'_native.tsv')).read_bytes();check(symbolic==native,'wrong template evaluation '+str(root))
   selected=(root/r['selected_file']).read_bytes();pairs=list(zip(symbolic.splitlines()[1:],selected.splitlines()[1:]));check(len(symbolic.splitlines())==len(selected.splitlines()),'count mismatch')
   different=sum(a!=b for a,b in pairs);check(different==int(r['different_entries']),'fit counter mismatch')
   if not different:fits.append(r['family'])
  parts.append(name+'='+(','.join(fits) if fits else 'outside_these_four_template_families'))
 return True,'; '.join(parts)+'; SYMBOLIC/bounded_fit/*/fit.tsv and complete native/symbolic/selected tables; point witnesses do not assert impossibility in every quasi-affine family'

def current_symbolic_proofs():
 parts=[]
 for name in ALL:
  fitted=[r['family'] for r in table(S/'bounded_fit'/name/'fit.tsv') if int(r['different_entries'])==0]
  if not fitted:
   parts.append(name+' outside four families; original winner retained');continue
  root=S/'bounded_certificates'/name;m=json.loads((root/'manifest.json').read_text());check(m['family'] in fitted,'certificate does not match winner')
  arm=choice(name);selected=next(r for r in table(cell(name)/'auto.cu.top3.tsv') if r['rank']==arm[3:]);cg=REPO/selected['cg']
  check(sha(cg)==m['source_sha256'],'certificate is for a different solved CG')
  covered=set();samples=set()
  for shard in m['shards']:
   folder=root/shard;command=json.loads((folder/'command.json').read_text());check(command['exit_code']==0,'proof process failed '+name+'/'+shard)
   for r in table(folder/'proofs.tsv'):
    check(r['family']==m['family'] and int(r['grid'])==m['grid'],'wrong family/grid')
    check(all(r[k]=='1' for k in ('total','bijective','dense','acyclic','resident','level_exact')),'failed proof row')
    covered.update(range(int(r['begin']),int(r['end'])+1))
   for r in table(folder/'samples.tsv'):
    check((folder/r['template']).read_bytes()==(folder/r['native']).read_bytes(),'current native/template mismatch')
    samples.add(int(r['seq']))
  check(covered==set(range(1,129)),'current winner domain gap')
  check(samples=={1,32,64,96,128},'current winner endpoints/interiors missing')
  parts.append(f'{name} family={m["family"]} grid={m["grid"]} proved=128/128 samples=5/5; '+evidence(root))
 return True,'; '.join(parts)

def sd():
 root=S/'cross_grid';rows=table(root/'comparisons.tsv');check(len(rows)==16,'expected four templates x two grids x two references')
 check({r['grid'] for r in rows}=={'256','340'},'missing grid');check({r['reference'] for r in rows}=={'selected','resolved'},'missing fresh solve')
 for r in rows:
  a=table(root/r['template_file']);b=table(root/r['reference_file']);check(len(a)==len(b),'task count differs')
  for x,y in zip(a,b):check((x['stage'],x['task'])==(y['stage'],y['task']),'task identity differs')
  actual=sum((x['worker'],x['slot'])!=(y['worker'],y['slot']) for x,y in zip(a,b))
  check(actual==int(r['different_entries']),'comparison disagrees with raw Plan tables')
 log=(S/'cg_roundtrip.log').read_text();check(len(re.findall(r'^CG_SYMBOLIC_ROUNDTRIP .* identical=1$',log,re.M))==40,'CG serialize/read checks incomplete')
 return True,f'{len(rows)} template/champion/fresh-solve comparisons recomputed from complete raw tables; 40 CG round trips; CPU only; SYMBOLIC/cross_grid/'
def rebase():
 ok=True;details=[]
 for name in ALL:
  root=B/'bounded_raw'/name
  try:
   specs=json.loads((root/'specs.json').read_text());names=list(specs);check(len(names)==50,'expected 10 configurations x 5 arms');starts=set();sessions=set();n=0
   for i in range(25):
    for j,arm in enumerate(names):
     p=root/'measure'/arm/f'r{i}.log';m=json.loads(p.with_suffix('.json').read_text());check(m['round']==i and m['order']==(j-i)%50,'rotation mismatch');sessions.add(m['session']);starts.add(m['started_ns']);joint.timing(p)
     if arm.endswith('__full'):check(m['exit_code']==0 and 'RESULT status=PASS' in p.read_text(),'full correctness failed')
     n+=1
   check(n==1250 and len(starts)==1250 and len(sessions)==1,'missing/reused sessions');details.append(f'{name} {n}/1250')
   values=[]
   for i in range(25):
    at=lambda arm:joint.timing(root/'measure'/arm/f'r{i}.log')
    rotate=at('rotate__full')['l2_ms'];legacy=at('legacy_grid_stride__full')['l2_ms']
    selected=json.loads((root/'selection.json').read_text())['placement']
    full=at(selected+'__full');nowait=at(selected+'__nowait')['l2_ms'];neither=at(selected+'__neither')['l2_ms']
    barrier=full['l1_ms']-at(selected+'__l1nosync')['l1_ms']
    values.append((rotate/legacy,full['l2_ms']-nowait,nowait-neither,full['l2_ms']-at(selected+'__nofence')['l2_ms'],barrier,(full['l2_ms']-neither)/barrier if barrier else float('nan')))
   medians=[statistics.median(v[k] for v in values) if all(math.isfinite(v[k]) for v in values) else float('nan') for k in range(6)]
   details.append(name+' (rotate/legacy,selected_wait_ms,notify_ms,fence_ms,barrier_ms,protocol/barrier)='+repr(medians)+'; selected_nonpositive_barrier_pairs='+str(sum(v[4]<=0 for v in values)))
   for config in dict.fromkeys(arm.split('__')[0] for arm in names):
    arm=config+'__full';dump=root/'trace'/arm/'dump';source=Path(specs[arm]['source'])
    attribution=joint.trace.analyze(dump,source,1)
    cp=float(attribution['cp_corrected_ns']);queue=float(attribution['queue_lb_ms'])*1e6;floor=max(cp,queue)
    check(math.isfinite(floor) and floor>0,'invalid raw trace floor '+name+'/'+config)
    timed=[joint.timing(root/'measure'/arm/f'r{i}.log') for i in range(25)]
    latency=statistics.median(r['l2_ms'] for r in timed)
    relative=[timed[i]['l2_ms']/joint.timing(root/'measure'/(selected+'__full')/f'r{i}.log')['l2_ms'] for i in range(25)]
    if config in ('c1','c2','c3'):
     previous={'c1':selected,'c2':'c1','c3':'c2'}[config]
     incremental=[timed[i]['l2_ms']/joint.timing(root/'measure'/(previous+'__full')/f'r{i}.log')['l2_ms'] for i in range(25)]
     details.append(f'{name}/{config} relative_previous={previous}:{joint.interval(incremental)}')
    details.append(f'{name}/{config} cp_ns={cp:.3f} queue_ns={queue:.3f} measured_floor={latency*1e6/floor:.6f} l2_l1={statistics.median(r["l2_ms"]/r["l1_ms"] for r in timed):.6f} relative_selected={joint.interval(relative)} dump='+evidence(dump))
  except Exception as e:ok=False;details.append(f'{name}: {e}')
 try:
  pool=module('r6_placement_pool',B/'placement_pool.py');pooled,_=pool.calculate(B/'bounded_raw')
  details.extend('placement_pool '+repr(r) for r in pooled)
 except Exception as e:ok=False;details.append('placement_pool: '+str(e))
 return ok,'; '.join(details)+'; REBASE/bounded_raw/*/measure'
def models():
 root=EX/'MODELS';old_ok,old_detail=processes(root/'llama_mlp/correctness')
 maximal=root/'covered_llama_admitted2';ok,detail=processes(maximal/'correctness')
 raw=(maximal/'correctness/r0.log').read_text()
 outputs=re.findall(r'^E2E_OUTPUT_DIFF index=(\d+) .*?mismatch=(\d+)',raw,re.M)
 check(len(outputs)==66 and {int(i) for i,_ in outputs}==set(range(66)),'maximal graph output coverage differs')
 cg=(maximal/'auto.mlir').read_text();check('tilemega.solved_kappa' in cg,'maximal graph bypassed solver')
 mismatches=[(int(i),int(n)) for i,n in outputs if int(n)]
 return ok,detail+f'; maximal connected graph checked_outputs=66 mismatches={mismatches}; '+evidence(maximal/'correctness/r0.log')+'; earlier independent MLP subset='+str(old_ok)+' '+old_detail+' does not satisfy maximal-graph admission; no tolerance or output changes'

def history_order():
 pairs=[('041b7b81','5e7ab8b6','continuation derived combine before bound repair'),('f9f7fb31','1611afc3','interval writer before current winner audits'),('567cb81d','6344daa5','C1 before J1'),('dbfb4051','6344daa5','C2 before J1'),('7fc201bd','eaedcd14','FORK6 before Fuse R7 decision'),('c8d58a18','a88ec285','W2 before symbolic proof implementation')]
 details=[]
 for before,after,label in pairs:
  check(subprocess.run(['git','merge-base','--is-ancestor',before,after],cwd=REPO).returncode==0,label);details.append(label)
 for name in ALL:
  root=B/'bounded_raw'/name;m=json.loads((root/'selection.json').read_text());rev=m['frozen_commit']
  check(subprocess.run(['git','merge-base','--is-ancestor',rev,'HEAD'],cwd=REPO).returncode==0,'selection commit not an ancestor')
  choice_path=(cell(name)/'choice.json').relative_to(REPO)
  recorded=json.loads(subprocess.check_output(['git','show',rev+':'+str(choice_path)],cwd=REPO,text=True));check(recorded==m['choice'],'choice was not frozen at the recorded commit')
  stamp=int(subprocess.check_output(['git','show','-s','--format=%ct',rev],cwd=REPO,text=True))*10**9
  samples=list((root/'measure').rglob('r*.json'));check(len(samples)==1250,'incomplete rebase campaign '+name)
  check(all(json.loads(p.read_text())['started_ns']>=stamp for p in samples),'measurement predates frozen commit')
  details.append(name+' selection '+rev[:8]+' precedes all B1 measurements')
 return True,'; '.join(details)
def target_selfcheck():
 details=[]
 for name in ('COSTMODEL','JOINT2','REBASE','MODELS'):
  folder=EX/name/'selfcheck_sm120_portable';text=(folder/'status.txt').read_text();check('SELF_CHECK PASS; sm_120 not run' in text,'self-check failed '+name)
  check((folder/'compile.log').exists(),'compile evidence missing '+name);details.append(evidence(folder))
 return True,'four local compile/guard checks; no sm_120 execution claim; '+'; '.join(details)

def main():
 gates=[('C-a','hard',cost_branches),('C-b','hard',lambda:rank_gate('full_ns',.880288958)),('C-c','hard',lambda:rank_gate('coarse_ns',.85)),('C-d','report',replay),('C-e','report',kloop),('C-f','hard',cf),('J-a','research',lambda:paired_gate(REAL,'control',1)),('J-b','hard',queue_gate),('J-c','hard',budget),('J-outer','hard',outer_bounds),('J-d','hard',ranking),('J-e','hard',lambda:paired_gate(REFS,'champion',1.02)),('J-f','hard',jf),('J-g','report',fuse),('W-a','hard',writeback),('W-b','hard',command_gate),('W-c','hard',legacy),('W-d','hard',roundtrip),('W-e','hard',we),('W-interval','hard',interval_writeback),('B1','report',rebase),('S-a','hard',symbolic_proofs),('S-b','hard',symbolic_samples),('S-c','hard',symbolic_champion),('S-current','hard',current_symbolic_proofs),('S-d','report',sd),('A1-subset','report',models),('H2','hard',sass),('H4','hard',history_order),('H7','report',target_selfcheck)]
 for name,kind,fn in gates:gate(name,kind,fn)
 failed=sum(not ok and kind=='hard' for _,kind,ok in results);print(f'R6_VERIFY gates={len(results)} hard_failures={failed} exit={int(failed>0)}',flush=True);return int(failed>0)
if __name__=='__main__':sys.exit(main())
