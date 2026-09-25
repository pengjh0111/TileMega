#!/usr/bin/env python3
"""R9b acceptance from source, commands, raw measurements and prediction rows.

Missing evidence fails its gate. All checks run before the final nonzero exit.
No summary.md, summary.tsv, status.json or precomputed gate verdict is read.
"""
import csv,hashlib,json,math,pathlib,re,statistics,subprocess,sys,struct
E=pathlib.Path(__file__).resolve().parent;ROOT=E.parents[2]
BASE=json.loads((E/'baseline.json').read_text())['baseline']
results={}
def read(p):return (ROOT/p).read_text()
def rows(p):
 with p.open() as f:return list(csv.DictReader(f,delimiter='\t'))
def check(name,fn):
 try:
  ok,detail=fn();state='PASS' if ok else 'FAIL'
 except Exception as ex:ok=False;state='FAIL';detail=f'{type(ex).__name__}: {ex}'
 results[name]=ok;print(f'{name} {state} {detail}',flush=True)
def require(p,*need,absent=()):
 text=read(p);evidence=[]
 for pattern in need:
  m=re.search(pattern,text)
  if not m:return False,f'{p}: missing /{pattern}/'
  line=text.count('\n',0,m.start())+1;evidence.append(f'{p}:{line}: {text.splitlines()[line-1].strip()}')
 for pattern in absent:
  m=re.search(pattern,text)
  if m:return False,f'{p}:{text.count(chr(10),0,m.start())+1}: prohibited /{pattern}/'
 return True,'; '.join(evidence) or f'{p}: prohibited tokens absent'
def body(text,signature):
 start=text.index('{',text.index(signature));depth=1;i=start+1
 # Source checks use balanced braces, ignoring quoted literals and comments.
 pattern=re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]',re.S)
 for m in pattern.finditer(text,i):
  if m[0]=='{':depth+=1
  elif m[0]=='}':
   depth-=1
   if not depth:return text[start:m.end()]
 raise ValueError('unbalanced function '+signature)
def allof(*checks):
 return all(x[0] for x in checks),'; '.join(x[1] for x in checks)
def outer():
 p='lib/Solver/SkeletonSearch.cpp';s=read(p);b=body(s,'CoordinateDescent(')+body(s,'SkeletonCandidate Evaluate(')
 return 'EvaluateFlow(' in b and not re.search(r'ScheduleBySkeleton|SimulateExecution',b),f'{p}:{s[:s.index("SkeletonCandidate Evaluate(")].count(chr(10))+1}: EvaluateFlow in Evaluate; CoordinateDescent checked; EvaluateFlow={b.count("EvaluateFlow(")}, forbidden={bool(re.search("ScheduleBySkeleton|SimulateExecution",b))}'
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True)
def immutable():
 paths=['include/tilemega/Codegen/tasks/*TaskBody*.h','include/tilemega/Codegen/tasks/EventSync.cuh','include/tilemega/Codegen/tasks/ClusterSync.cuh','include/tilemega/Codegen/tasks/ModelRuntime.h','include/tilemega/Runtime','lib/Solver/RuntimeProjection.cpp','lib/Solver/EftPlacement.cpp','include/tilemega/Solver/EftPlacement.h','include/tilemega/Solver/ChainPlacement.h','include/tilemega/Solver/JointPlacement.h','include/tilemega/Dialect/CouplingGraph/PlacementSolvePass.h']
 diff=git('diff','--name-only',BASE,'--',*paths)
 p='lib/Solver/ExecutionSimulator.cpp';old=git('show',BASE+':'+p);new=read(p)
 # Only the guarded dispatch and its include may differ in the existing evaluator.
 delta=git('diff','--unified=0',BASE,'--',p)
 edits=[s for s in delta.splitlines() if s[:1] in '+-' and not s.startswith(('+++','---'))]
 sim_ok=all(s.startswith('+') and ('FluidExecutionSimulator.h' in s or 'if(options.dram_fluid)return SimulateFluidExecution' in s) for s in edits)
 cost_path='lib/Solver/CostModel.cpp';cost=read(cost_path)
 guard=body(cost,'if(options_.regime_a && dtype_==ScalarType::kBF16)')
 cost_rest=cost.replace('  if(options_.regime_a && dtype_==ScalarType::kBF16) '+guard+'\n','',1)
 cost_ok=cost_rest==git('show',BASE+':'+cost_path)
 model_path='lib/Solver/TaskModel.cpp';model=read(model_path)
 current=body(model,'PriceTaskInstances(');original=body(git('show',BASE+':'+model_path),'PriceTaskInstances(')
 current=current.replace('  bool regime_a=cost.options().regime_a && model.dtype==ScalarType::kBF16;\n','')
 current=current.replace('collective && !(regime_a && cost.options().physical_traffic) ?', 'collective ?')
 current=current.replace('std::tuple<double,double,long,double,double>', 'std::tuple<double,double,long>')
 current=current.replace('reduction[i],\n        regime_a?traffic[i].no_producer_read_bytes:0.0,regime_a?traffic[i].external_write_bytes:0.0)', 'reduction[i])')
 model_ok=current==original
 return not diff and sim_ok and cost_ok and model_ok,f'git diff {BASE}: protected={diff.strip() or "none"}; simulator guarded additions={sim_ok}; CostModel removing BF16 regime_a branch equals baseline={cost_ok}; PriceTaskInstances with regime_a=false equals baseline body={model_ok}; new traffic metadata audited by G-5/G-6'

def no_schedule():
 paths=set(git('diff','--name-only',BASE).splitlines())|set(git('ls-files','--others','--exclude-standard','include','lib','tools').splitlines())
 bad=[];scanned=0
 for p in paths:
  file=ROOT/p
  if not p.startswith(('include/','lib/','tools/')) or p=='tools/tilemega-affine-probe.cpp' or not file.is_file():continue
  if file.suffix not in ('.h','.cpp','.cuh','.py','.td'):continue
  scanned+=1
  for i,line in enumerate(file.read_text().splitlines(),1):
   if re.search(r'\bisl_schedule_\w+\s*\(',line):bad.append(f'{p}:{i}:{line.strip()}')
 return not bad,f'{scanned} changed implementation files; '+'; '.join(bad)
def command_audit():
 paths=sorted(set(E.rglob('*command.json'))|set(E.rglob('build_command.txt')));bad=[]
 for p in paths:
  if re.search(r'MIDPOINT_REFINE(?:=|\s+)1\b',p.read_text()):bad.append(str(p.relative_to(ROOT)))
 return bool(paths) and not bad,f'{len(paths)} raw command files (including occupancy build_command.txt); refine1={bad}; examples={[str(p.relative_to(ROOT)) for p in paths[:3]]}'
def sass():
 found=sorted(E.rglob('sass.log'));bad=[];kernels=0
 for p in found:
  current=None
  for line in p.read_text().splitlines():
   m=re.search(r'Function\s*:\s*(\S+)',line)
   if m:
    current=m[1] if any(t in m[1] for t in ('tilemega_l1_kernel','tilemega_l2_kernel')) else None
    kernels+=bool(current)
   if current and re.search(r'\b(?:DADD|DMUL|DFMA|DSETP)\b|\bF64\b|\b(?:F2D|D2F|I2D|D2I)\b',line):bad.append((str(p.relative_to(ROOT)),line.strip()))
 return kernels>0 and not bad,f'kernel_functions={kernels} FP64={len(bad)} first={bad[:2]}'
def variant_check():
 checked=0;bad=[];proof=[]
 commands=list(E.rglob('build.command.json'))+list(E.rglob('original_build.command.json'))
 for cmd in commands:
  # Rejected preflight binaries have no timings and are retained separately.
  if not any('E2E_TIME ' in p.read_text() for p in cmd.parent.glob('process_*.log')):continue
  data=json.loads(cmd.read_text());args=data.get('command',data) if isinstance(data,dict) else data
  cu=next((pathlib.Path(x) for x in args if isinstance(x,str) and x.endswith('.cu')),None)
  if not cu or not cu.exists():bad.append(str(cmd)+': source missing');continue
  text=cu.read_text();m=re.search(r'#define TILEMEGA_GEMM_VARIANT_COUNT (\d+)',text)
  if not m:bad.append(str(cu)+': missing count');continue
  cg=cu.with_suffix('.mlir')
  if 'early_corrected' in cu.parts:cg=E/'early'/cg.name
  if not cg.exists():bad.append(str(cg)+': solved CG missing');continue
  runtime=re.search(r'tilemega.gemm_runtime = \[(.*?)\]',cg.read_text(),re.S)
  if not runtime:bad.append(str(cg)+': no GEMM runtime configurations');continue
  shapes=set()
  for item in re.findall(r'\{(.*?)\}',runtime[1]):
   vals=dict(re.findall(r'(tile_m|tile_n|tile_k|stages) = (\d+)',item))
   shapes.add(tuple(int(vals[k]) for k in ('tile_m','tile_n','tile_k','stages')))
  compiled=set()
  for prefix in re.findall(r'#define (TILEMEGA_GEMM_(?:V\d+_)?)TILE_M ',text):
   compiled.add(tuple(int(re.search(r'#define '+prefix+suffix+r' (\d+)',text)[1]) for suffix in ('TILE_M','TILE_N','TILE_K','STAGES')))
  checked+=1;proof.append(f'{cu}:{text[:m.start()].count(chr(10))+1}: {m[0]}; CG distinct={len(shapes)} emitted={len(compiled)}')
  if int(m[1])!=len(shapes) or compiled!=shapes:bad.append(f'{cu}: count={m[1]} distinct_CG={len(shapes)} emitted_match={compiled==shapes}')
 return checked>0 and not bad,f'measured sources={checked}; CG versus emitted shapes: {bad}; '+ '; '.join(proof)
check('K-1',outer)
check('K-2',lambda:require('lib/Solver/SkeletonSearch.cpp',absent=(r'EvaluateIsolated',r'\bfork\s*\(')))
check('K-3',lambda:allof(require('lib/Solver/PiecePricing.cpp',r'SemanticSignature',r'kernel_shared_bytes',absent=(r'ownership\.Points\(',)),require('lib/Solver/FlowPreparation.cpp',r'geometry_keys',absent=(r'ownership\.Points\(',r'problem\.geometry\s*\)\s*key'))))
check('K-4',lambda:require('lib/Solver/PlanSkeleton.cpp',r'space.width=all_workers \? grid:std::min\(grid,k_base\)',absent=(r'k_base\s*\*\s*loads',)))
check('K-5',lambda:require('lib/Solver/PlanSkeleton.cpp',r'colocation->reverse.Query',r'OracleKind::Unique',r'UniqueMapText'))
check('K-6',lambda:require('lib/Analysis/DramFloor.cpp',r'isl_map_range',r'isl_set_card',r'compute_ns',r'no_producer'))
check('K-7',lambda:allof(require('lib/Solver/TaskPriceParts.cpp',r'options_\.regime_a',r'curve.HitFraction',r'no_producer_read_bytes',r'produced_read_bytes',r'kPhysicalTensor'),require('include/tilemega/Solver/CostModel.h',r'no_producer_read_bytes',r'produced_read_bytes')))
check('K-8',lambda:allof(require('lib/Solver/TaskPriceParts.cpp',r'CostModel::PriceParts',r'fit.latency_scale',r'fit.stage_rate_bytes_per_ns',r'traits.stages-1'),require('lib/Solver/CostModel.cpp',r'IsolatedNs\(PriceParts'),require(str((E/'unit/prices.log').relative_to(ROOT)),r'stage_monotonic=PASS')))
check('K-9',lambda:allof(require('lib/Solver/StageFlowModel.cpp',r'CoarsenRelease',r'EventCalibrationFor',r'HopCurve',r'DramFluidServer'),require('lib/Solver/DramFluid.cpp',r'cap',r'rate')))
check('K-10',lambda:allof(*(require(p,absent=(r'tilemega\.g["\s]',r'AlignedEdges',r'Bind\("T[mn]"')) for p in ['lib/Solver/SkeletonSearch.cpp','lib/Solver/FlowPreparation.cpp','lib/Solver/StageFlowModel.cpp'])))
check('K-11',no_schedule)
check('K-12',lambda:allof(command_audit(),sass()))
check('K-13',immutable)
check('K-14',variant_check)
check('K-15',lambda:allof(require('lib/Solver/TaskPriceParts.cpp',r'options_\.regime_a',r'physical_fixed',r'fixed_physical'),(all(json.loads((E/'fit/target.json').read_text())['calibration_by_dtype'][d]['task_body'].get(k)==v for d,cal in json.loads(read('docs/experiments/COSTMODEL/event_fit/target.json'))['calibration_by_dtype'].items() for k,v in cal.get('task_body',{}).items()),'old task_body fields preserved')))
check('K-16',lambda:allof(require('lib/Solver/StageFlowModel.cpp',r'all_external_miss && !options.no_external',r'T >= T_dram assertion'),require('lib/Solver/FluidExecutionSimulator.cpp',r'all_external_miss && !options.no_external_dram',r'T >= T_dram assertion')))
# Declared K-12 conflict is not hidden: its individual FAIL and G-3 FAIL remain.
check('G-1',lambda:(all(results.get(f'K-{i}',False) for i in range(1,17) if i!=12) and command_audit()[0] and '## SV-9(e) / G-3: static FP64' in (E/'deviations.md').read_text(), 'K-12 FP64 conflict declared in deviations.md; all other structural checks required'))
def parse_measure(p):
 s=p.read_text();h=re.search(r'E2E_HASH l05=(\w+) l1=(\w+) l2=(\w+)',s)
 good=h is not None and len(set(h.groups()))==1 and all(re.search(k+r'=0\b',s) for k in ('l1_vs_l05_mismatch','l2_vs_l1_mismatch'))
 t=re.search(r'E2E_TIME l05_ms=([\d.]+) l1_ms=([\d.]+).*? l2_ms=([\d.]+)',s)
 return good,tuple(map(float,t.groups())) if t else None
def selected_measurement(out):
 candidates=[]
 for r in rows(out/'selected.cu.top3.tsv'):
  directory=pathlib.Path(r['source']+'.measurement');logs=sorted(directory.glob('process_*.log'))
  values=[parse_measure(p) for p in logs]
  if len(values)==10 and all(good and t for good,t in values):candidates.append((statistics.median(t[2] for good,t in values),directory))
 expected=rows(out/'selected.cu.top3.tsv')
 if len(expected)!=3 or len(candidates)!=3:raise ValueError(f'{out}: top-3 incomplete ({len(candidates)}/{len(expected)}); cannot select winner')
 return min(candidates)[1]
def correctness():
 issues=[];count=0
 expected=[E/'controls'/f'{m}_s{seq}' for m in ('llama','qwen3') for seq in (1,4,16,64)]
 for family,models,seqs in [('matrix',('llama','qwen3'),(1,4,16,64)),('reference',('gqa2','mha4'),(4,128))]:
  for model in models:
   for seq in seqs:
    out=E/family/f'{model}_s{seq}'
    try:expected.append(selected_measurement(out))
    except Exception as error:issues.append(str(error))
 for directory in expected:
  logs=list(directory.glob('process_*.log'));passed=sum(parse_measure(p)[0] for p in logs);count+=passed
  if len(logs)!=10 or passed!=10:issues.append(f'{directory.relative_to(E)}={passed}/{len(logs)}')
 # Every actually measured binary is subject to the gate, including top-3
 # runners-up and ablations. A failed candidate cannot vanish in selection.
 all_logs={p.parent for p in E.rglob('process_*.log') if 'E2E_TIME ' in p.read_text()}
 for directory in sorted(all_logs):
  logs=list(directory.glob('process_*.log'));passed=sum(parse_measure(p)[0] and parse_measure(p)[1] is not None for p in logs)
  required=50 if 'colocation' in directory.parts else 10
  if len(logs)<required or passed!=len(logs):issues.append(f'{directory.relative_to(E)}={passed}/{len(logs)} required={required}')
 for model in ('llama','qwen3'):
  directory=E/'colocation'/model;logs=list(directory.glob('process_*.log'));passed=sum(parse_measure(p)[0] for p in logs)
  if len(logs)<50 or passed!=len(logs):issues.append(f'colocation/{model}={passed}/{len(logs)}')
  commands=list(directory.glob('build.command.json'))
  if commands:
   data=json.loads(commands[0].read_text());args=data.get('command',data) if isinstance(data,dict) else data
   cu=next(pathlib.Path(x) for x in args if isinstance(x,str) and x.endswith('.cu'))
   cg=cu.with_suffix('.mlir').read_text();omitted=re.search(r'tmexec.sync_omitted_event_waits = (\d+)',cg)
   if not omitted or int(omitted[1])<=0:issues.append(f'colocation/{model}: binary CG has no actual wait omission proof')
 return not issues,f'required samples passed={count}; measured binaries={len(all_logs)}; '+', '.join(issues)
check('G-2',correctness);check('G-3',sass)
def floor_check():
 details=[];ok=True
 for model,name in [('llama','llama_config_public_copy.json'),('qwen3','qwen_config.json')]:
  source=E.parent/'MODELS/sources'/name;config=json.loads(source.read_text())
  h=config['hidden_size'];inter=config['intermediate_size'];layers=config['num_hidden_layers'];heads=config['num_attention_heads'];kv=config['num_key_value_heads'];dim=config.get('head_dim',h//heads);vocab=config['vocab_size']
  # BF16 projection/norm weights, LM head, final norm, shared RoPE inv_freq.
  dense=2*(layers*(2*h*heads*dim+2*h*kv*dim+3*h*inter+2*h+(2*dim if model=='qwen3' else 0))+h*vocab+h)+dim
  for seq in (1,4,16,64):
   log=E/'floor'/f'{model}_s{seq}.log';values=dict(re.findall(r'(\w+)=([\d.eE+-]+)',log.read_text()))
   command=json.loads((E/'floor'/f'{model}_s{seq}.command.json').read_text());fixture=pathlib.Path(command[5])
   cg=(ROOT/command[1]).read_text();bits=re.search(r'token_id_bits = (32|64)',cg);bits=int(bits[1]) if bits else 64
   raw=(fixture/'input_input_ids.bin').read_bytes()[:seq*(bits//8)]
   unique=len(set(struct.unpack('<'+('q' if bits==64 else 'i')*seq,raw)))
   expected=dense+2*h*unique;weight=sum(float(r['read_bytes']) for r in rows(E/'floor'/f'{model}_s{seq}.tsv') if r['source']=='weight')
   err=abs(weight/expected-1);ok &= err<=.005 and weight==float(values['weight_bytes'])
   details.append(f'{model}_s{seq}: config_bytes={expected} CG_bytes={weight} rel={err:.9g} T_dram={values["dram_ns"]} T_compute={values["compute_ns"]} source={source.relative_to(ROOT)}')
 return ok,'; '.join(details)
def measured_floor_check():
 cal=json.loads((E/'fit/target.json').read_text())['calibration_by_dtype']['bf16']['pipelines']
 allowance=cal['l2_knee_bytes']/cal['dram_gbps'];bad=[];checked=0
 for path in E.rglob('process_*.log'):
  good,times=parse_measure(path)
  if not times:continue
  # Every timed directory has its model encoded either in its path or raw
  # build command. Use source CG where available, otherwise the audited cell.
  command=next(iter(path.parent.glob('*build.command.json')),None)
  bound=None
  if command:
   data=json.loads(command.read_text());args=data.get('command',data) if isinstance(data,dict) else data
   source=next((pathlib.Path(x) for x in args if isinstance(x,str) and x.endswith('.cu')),None)
   if source and source.with_suffix('.mlir').exists():
    match=re.search(r'floor_value_ns = ([0-9.eE+-]+)',source.with_suffix('.mlir').read_text())
    if match:bound=float(match[1])
  if bound is None:
   match=re.search(r'(llama|qwen3)_s(1|4|16|64)(?:\D|$)',str(path))
   if match:
    raw=E/'floor'/f'{match[1]}_s{match[2]}.log';bound=float(re.search(r'floor_ns=([0-9.eE+-]+)',raw.read_text())[1])
  if bound is None:continue # Reference floor is checked from its generated CG.
  checked+=1
  if times[2]*1e6<bound-allowance:bad.append(str(path.relative_to(E)))
 return checked>0 and not bad,f'raw timed anchored samples={checked}; allowance_ns={allowance}; below_floor={bad}'
check('G-4',lambda:allof(floor_check(),measured_floor_check()))
def ranks(a):
 ids=sorted(range(len(a)),key=a.__getitem__);r=[0.]*len(a);i=0
 while i<len(a):
  j=i+1
  while j<len(a) and a[ids[j]]==a[ids[i]]:j+=1
  for k in ids[i:j]:r[k]=(i+j-1)/2
  i=j
 return r
def rho(a,b):return statistics.correlation(ranks(a),ranks(b))
def replay():
 ok=True;details=[]
 for dtype in ('bf16','f32'):
  baseline='bits_baseline_'+dtype
  for model in ('gqa2','mha4'):
   base=rows(E/'replay'/baseline/f'predictions_{model}.tsv')
   for arm in [f'bits_off_{dtype}']+(['bits_on_f32'] if dtype=='f32' else []):
    new=rows(E/'replay'/arm/f'predictions_{model}.tsv');equal=len(base)==len(new) and all(a==b and all(k in a for k in ('total_ns_bits','gemm_ns_bits','combine_ns_bits','other_ns_bits','barrier_ns_bits','task_ns_sum_bits')) for a,b in zip(base,new));ok &= equal;details.append(f'{model}/{arm} n={len(new)} bit_text_equal={equal}')
 for model,limit in [('gqa2',.9039),('mha4',.8911)]:
  data=rows(E/'replay/selected_bf16'/f'predictions_{model}.tsv');r=rho([float(x['measured_ms']) for x in data],[float(x['model_ms']) for x in data]);ok &= r>=limit;details.append(f'{model} selected rho={r:.9g} required={limit}')
 return ok,'; '.join(details)
check('G-5',replay)
def pieces():
 details=[];ok=True
 for model in ('llama','qwen3'):
  directory=E/('piece_validation_boundary_identity' if model=='llama' else 'piece_validation')/model
  p=directory/'price_checks.tsv';data=rows(p);configs={r['configuration'] for r in data}
  status=json.loads((p.parent/'exit.json').read_text())
  by_config={c:{int(r['stage']) for r in data if r['configuration']==c} for c in configs}
  good=len(configs)==3 and status['exit']==0 and all(float(r['relative_error'])<=1e-9 and r['bit_exact']=='1' for r in data)
  good &= all(v and v==set(range(max(v)+1)) for v in by_config.values())
  if model=='llama':
   previous=rows(E/'piece_validation'/model/'price_checks.tsv');lookup={(r['configuration'],r['space']):r for r in data}
   good &= all((r['configuration'],r['space']) in lookup and all(r[k]==lookup[r['configuration'],r['space']][k] for k in ('tasks','pieces','piece_ns','tile_ns','relative_error','bit_exact')) for r in previous)
   details.append(f'old exhaustive prefix agrees={len(previous)} spaces; each space fully summed; bit identity sampled at first/middle/last price pieces')
  ok &= good;details.append(f'{p.relative_to(E)} configs={len(configs)} spaces={len(data)} max_rel={max(float(r["relative_error"]) for r in data)} completed={status["exit"]==0}')
 return ok,'; '.join(details)
check('G-6',pieces)
def budgets():
 maxima=[];missing=[];sim=[];solves=[]
 for m in ('llama','qwen3'):
  for seq in (1,4,16,64):
   cell=f'{m}_s{seq}';p=E/'flow_final'/cell/'run.log'
   match=re.search(r'FLOW .* max_ms=([\d.eE+-]+)',p.read_text()) if p.exists() else None
   if match:maxima.append(float(match[1]))
   else:missing.append('flow/'+cell)
   out=E/'matrix'/cell
   if not (out/'solve.exit.json').exists():missing.append('solve/'+cell);continue
   record=json.loads((out/'solve.exit.json').read_text())
   phases=rows(out/'selected.cu.timing.tsv')
   compile_ms=sum(float(x['total_ms']) for x in phases if x['phase']=='megakernel_compile')
   solves.append(record['wall_seconds']-compile_ms/1000)
   if record['exit']!=0:missing.append('solve_exit/'+cell)
 for m in ('llama','qwen3'):
  for p in (E/'validation'/m).glob('sample_*.timing.tsv'):
   sim.extend(float(r['total_ms']) for r in rows(p) if r['phase']=='simulate')
 ok=not missing and bool(sim) and max(maxima)<=10 and max(sim)<=5000 and max(solves)<=1800
 return ok,f'flow_max_ms={max(maxima,default=float("nan"))} pure_fluid_max_ms={max(sim,default=float("nan"))} solve_excluding_final_compile_max_s={max(solves,default=float("nan"))} missing={missing}'
check('G-7',budgets)
def concordance():
 ok=True;detail=[]
 for m in ('llama','qwen3'):
  directory=E/'validation'/m;r=rows(directory/'samples.tsv');complete=(directory/'exit.json').exists() and json.loads((directory/'exit.json').read_text())['exit']==0;a=[float(x['flow_ns']) for x in r];b=[float(x['fluid_ns']) for x in r];score=rho(a,b);ok &= complete and len(r)>=100 and score>=.85;detail.append(f'{m} n={len(r)} completed={complete} rho={score} ratio_p50={statistics.median(x/y for x,y in zip(a,b))}')
 return ok,'; '.join(detail)
check('G-8',concordance)
check('A-runtime-release',lambda:require(str((E/'unit/runtime_release.log').relative_to(ROOT)),r'RUNTIME_RELEASE checks=2064 mismatches=0(?:\s|$)'))
def performance(absolute=False):
 wins=0;ratios=[];details=[]
 for m in ('llama','qwen3'):
  for s in (1,4,16,64):
   def median(directory):
    logs=sorted(directory.glob('process_*.log'))
    if len(logs)!=10:raise ValueError(f'{directory}: requires 10 logs, found {len(logs)}')
    values=[parse_measure(p) for p in logs]
    if not all(good and t for good,t in values):raise ValueError('invalid timing/correctness sample')
    return statistics.median(t[2] for _,t in values)
   control=median(E/'controls'/f'{m}_s{s}');selected=median(selected_measurement(E/'matrix'/f'{m}_s{s}'));wins+=selected<=control
   f=float(re.search(r'floor_ns=([\d.eE+-]+)',(E/'floor'/f'{m}_s{s}.log').read_text())[1])/1e6
   if s<=16:ratios.append(selected/f)
   details.append(f'{m}_s{s}: L2={selected} control={control} L2/floor={selected/f}')
 gm=statistics.geometric_mean(ratios)
 return (gm<=2 if absolute else wins>=6),f'wins={wins}/8 geomean_seq_le16={gm}; '+'; '.join(details)
check('G-9',performance);check('G-10',lambda:performance(True))
def early():
 detail=[];ok=True
 for seq in (1,4):
  p=E/'early_resident'/f'llama_s{seq}.measurement';logs=sorted(p.glob('process_*.log'))
  parsed=[parse_measure(x) for x in logs];valid=[t for good,t in parsed if good and t]
  good=len(logs)==10 and len(valid)==10;ok &= good
  control=[parse_measure(x)[1] for x in (E/'controls'/f'llama_s{seq}').glob('process_*.log')]
  ratio=statistics.median(t[2] for t in valid)/statistics.median(t[2] for t in control) if valid else float('nan')
  detail.append(f'{p.relative_to(E)} internal={len(valid)}/{len(logs)} L2/control={ratio}')
 return ok,'; '.join(detail)
def decomposition():
 detail=[];ok=True
 for model in ('llama','qwen3'):
  for seq in (1,4,16,64):
   cell=f'{model}_s{seq}';out=E/'matrix'/cell
   try:
    table=rows(out/'selected.cu.top3.tsv');choice=selected_measurement(out)
    source=next(r for r in table if pathlib.Path(r['source']+'.measurement')==choice)
    key=source['key'];found=[]
    for path in out.glob('*.flow.tsv'):
     for r in rows(path):
      if r['key']==key:found.append((path,r))
    if not found:raise ValueError('missing raw counterfactual solves '+cell)
    for path,r in found:
     x={k:float(v) for k,v in r.items() if k!='key'}
     sync=x['T']-x['T_s'];fixed=x['T_s']-x['T_sf'];contention=x['T_sf']-max(x['T_floor'],x['T_sf_infinite']);chain=max(0.,x['T_sf_infinite']-x['T_floor'])
     error=abs(sync+fixed+contention+chain-(x['T']-x['T_floor']))
     ok &= error<=1e-6 and abs(x['pg_upper_bound']-(x['T']-max(x['T_floor'],x['T_np0'])))<=1e-6
     links=rows(path.with_name(path.name.replace('.flow.tsv','.flow_chain.tsv')))
     protocol=sum(float(r[k]) for r in links for k in ('wait_ns','publication_ns','hop_ns'))
     attention=sum(float(r['end_ns'])-float(r['start_ns']) for r in links if r['category']=='attention')/x['T']
     detail.append(f'{path.relative_to(E)} closure_ns={error} sync={sync} fixed={fixed} contention={contention} chain={chain} PG={x["pg_upper_bound"]} chain_protocol_ns={protocol} attention_fraction={attention} bubble_ns={(x["T"]-x["T_floor"])/x["chain_depth"]}')
   except Exception as error:ok=False;detail.append(cell+': '+str(error))
 import importlib.util
 spec=importlib.util.spec_from_file_location('r9b_raw_trace',E.parent/'TRACE_V2/analyze.py');trace=importlib.util.module_from_spec(spec);spec.loader.exec_module(trace)
 for model in ('llama','qwen3'):
  for seq in (1,64):
   dump=E/'trace'/f'{model}_s{seq}'/'dump'
   try:
    reconstructed,spaces,links=trace.analyze_task_space_trace(dump)
    wall=sum(r['wall_ns'] for r in links)
    closed=wall==reconstructed['cp_reconstructed_with_publish_ns'] and all(min(r[k] for k in ('publish_ns','wait_hop_ns','barrier_ns','idle_ns','task_fixed_plus_mainloop_ns'))>=0 for r in links)
    ok &= closed;detail.append(f'{dump.relative_to(E)} spaces={len(spaces)} links={len(links)} wall_ns={wall} partition_closed={closed}')
   except Exception as error:ok=False;detail.append(str(dump.relative_to(E))+': '+str(error))
 return ok,'; '.join(detail)
check('G-11',decomposition)
def theta_grid():
 details=[];ok=True
 for model in ('llama','qwen3'):
  imports=rows(E/'theta'/f'{model}.import.tsv');count=sum(int(r['count']) for r in imports if r['phase']=='import')
  ok &= count==1;details.append(f'{model} import_count={count}')
  previous=None
  for seq in (1,2,4,8,16,32,64):
   directory=E/'theta'/f'{model}_s{seq}'
   try:
    completed=rows(directory/'completed.tsv');assert len(completed)==1
    p=directory/'selected.cu.search.tsv';valid=[]
    for line in p.read_text().splitlines():
     fields=line.split('\t')
     if fields[0]=='EVALUATE' and len(fields)>=7 and not fields[6] and math.isfinite(float(fields[3])):valid.append((float(fields[3]),fields[2]))
    if not valid:raise ValueError('no valid candidate')
    best=min(valid);done=completed[0]
    ok &= abs(best[0]-float(done['flow_ns']))<=1e-6 and int(done['seq'])==seq
    bound=rows(directory/'selected.cu.floor.tsv')[0];ratio=best[0]/float(bound['floor_ns'])
    changes=previous is not None and previous[1]!=done['key']
    details.append(f'{model}_s{seq} complete T={best[0]} T/floor={ratio} config={done["key"]} seed_seq={done["seed_seq"]} changed_from_previous={changes}')
    previous=(seq,done['key'])
   except Exception as error:ok=False;details.append(f'{model}_s{seq}: {error}')
 return ok,'; '.join(details)
check('G-12',theta_grid)
def ablations():
 expected=[E/'ablations'/f'{m}_s4'/arm for m in ('llama','qwen3') for arm in ('template','eft','wide')]
 expected += [E/'stages'/f'llama_s4_S{s}' for s in (2,3,4)]
 detail=[];ok=True
 for out in expected:
  logs=sorted(out.glob('process_*.log'));data=[parse_measure(p) for p in logs];valid=[t for good,t in data if good and t]
  ok &= len(valid)==10 and len(logs)==10;detail.append(f'{out.relative_to(E)} n={len(valid)} L2={statistics.median(t[2] for t in valid) if valid else float("nan")}')
 return ok,'; '.join(detail)
check('G-13',ablations);check('G-14',early)
def fits():
 # Reconstruct every prediction from raw phase observations and target fields.
 C=E.parent/'COSTMODEL';data=rows(C/'body_fit/observations.tsv')
 old=json.loads((C/'event_fit/target.json').read_text());new=json.loads((E/'fit/target.json').read_text())
 cal=new['calibration_by_dtype']['bf16']['pipelines'];fit=old['calibration_by_dtype']['bf16']['task_body'];updated=new['calibration_by_dtype']['bf16']['task_body'];sms=new['resources']['num_sms']
 errors={};specs={};metadata={}
 for r in data:
  tm,tn,tk,stage=[int(r[x]) for x in ('tile_m','tile_n','tile_k','stages')]
  dump=pathlib.Path(r['dump']);identity=(r['cohort'],r['cell'])
  if identity not in specs:specs[identity]=json.loads((C/identity[0]/identity[1]/'specs.json').read_text())
  if dump not in metadata:metadata[dump]={x['key']:x['value'] for x in rows(dump/'meta.tsv')}
  m=int(metadata[dump]['seq']);occupancy=float(specs[identity][r['arm']].get('residency',1))
  assert m<tm or m%tm==0,'nonuniform M tails need per-coordinate reconstruction'
  physical_writes=min(m,tm)*tn;physical_stage_bytes=2*tk*(min(m,tm)+tn)
  external=tn/(min(m,tm)+tn) if r['cell'].startswith('real_') else 0
  latency=(1-external)*cal['l2_latency_ns']+external*cal['dram_latency_ns']
  body=fit['loop_body'][0]+fit['loop_body'][1]*tm*tn*tk
  lanes=max(occupancy*physical_stage_bytes/(cal['l2_gbps']/sms),occupancy*2*tm*tn*tk/(cal['tc_bf16_gflops']/sms))
  operand=float(r['operand_elements']);observed_loop=float(r['loop_body_ns'])+float(r['loop_wait_ns'])
  old_loop=max(body,lanes)+fit['loop_wait'][0]+fit['loop_wait'][1]*operand
  new_loop=max(body,lanes,(updated['latency_scale']*latency+occupancy*physical_stage_bytes/updated['stage_rate_bytes_per_ns'])/(stage-1))
  old_fixed=sum(a*b for a,b in zip(fit['fixed'],[1,tm*tn,operand*stage]))
  new_fixed=sum(a*b for a,b in zip(updated['fixed_physical'],[1,physical_writes,operand*stage]))
  for name,observed,pred in [('old_loop',observed_loop,old_loop),('new_loop',observed_loop,new_loop),('old_fixed',float(r['fixed_ns']),old_fixed),('new_fixed',float(r['fixed_ns']),new_fixed)]:
   errors.setdefault((stage,name),[]).append(abs(pred/observed-1))
 detail=[f'S{stage} {name} n={len(values)} p50={statistics.median(values):.8g} mean={statistics.mean(values):.8g}' for (stage,name),values in sorted(errors.items())]
 return len(data)==490,'raw COSTMODEL/body_fit/observations.tsv + raw specs/meta + target parameters; '+'; '.join(detail)
check('G-15',fits)
def simulator_identity():
 directory=E/'simulator_identity'
 baseline=(directory/'baseline.tsv').read_bytes();current=(directory/'current.tsv').read_bytes()
 summaries=[r for r in baseline.decode().splitlines() if r.split('\t')[3]=='summary']
 commands=json.loads((directory/'commands.json').read_text())
 inputs=json.loads((directory/'inputs.json').read_text())
 intact=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in inputs.items())
 return baseline==current and len(summaries)==24 and intact and all(c['exit']==0 for c in commands),f'{directory.relative_to(ROOT)}: four cells x three placements x two sync settings; summaries={len(summaries)} rows={len(baseline.splitlines())} bitwise_equal={baseline==current} inputs_intact={intact} sha256={hashlib.sha256(current).hexdigest()}; historical trace durations are fixed replay inputs, not GPU timing claims'
check('A-simulator-legacy-identity',simulator_identity)
failed=[k for k,v in results.items() if not v]
print('R9B_VERIFY failures='+','.join(failed))
sys.exit(bool(failed))
