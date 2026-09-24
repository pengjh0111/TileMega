#!/usr/bin/env python3
"""R9 raw-evidence and source-contract audit. Runs every check before exiting.

No summary, findings, measurement.json, or precomputed gate conclusion is read.
Optional --evidence selects a copied SOLVER_V2 tree. Build/solve command metadata,
compiler dumps, resource probe logs, and each process log remain the authority.
"""
import argparse,csv,hashlib,json,pathlib,re,statistics,subprocess,sys
from plan_statistics import interleaving,attention_path
ROOT=pathlib.Path(__file__).resolve().parents[3]
AP=argparse.ArgumentParser(description=__doc__);AP.add_argument('--evidence',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parent);ARGS=AP.parse_args();HERE=ARGS.evidence
BASE=json.loads((HERE/'baseline.json').read_text())['baseline']
results={}
def check(name,good,evidence):
    results[name]=bool(good);print(f'{name} {"PASS" if good else "FAIL"} {evidence}')
def read(path):
    p=pathlib.Path(path);return p.read_text(errors='replace') if p.exists() else ''
def source(path):return read(ROOT/path)
def rows(path):
    p=pathlib.Path(path)
    return list(csv.DictReader(p.open(),delimiter='\t')) if p.exists() else []
def hits(path,pattern):
    return [f'{path}:{i}:{line.strip()}' for i,line in enumerate(source(path).splitlines(),1) if re.search(pattern,line)]
def body(text,name):
    start=text.find(name+'(')
    if start<0:return ''
    begin=text.find('{',start);level=0
    for pos in range(begin,len(text)):
        level+=(text[pos]=='{')-(text[pos]=='}')
        if level==0:return text[begin:pos+1]
    return ''
def safe(name,fn):
    try:fn()
    except Exception as exc:check(name,False,f'audit error {type(exc).__name__}: {exc}')
SEARCH='lib/Solver/SkeletonSearch.cpp';PLACE='lib/Solver/SkeletonPlacement.cpp';ORACLE='lib/Analysis/SymbolicOracle.cpp';CACHE='lib/Analysis/CouplingCache.cpp'
new=source(SEARCH);placement=source(PLACE);oracle=source(ORACLE);outer=body(new,'CoordinateDescent')
changed=subprocess.check_output(['git','diff','--name-only',BASE,'--'],cwd=ROOT,text=True).splitlines()
real_cells=[(model,seq) for model in ('llama','qwen3') for seq in (1,4,16,64)]
arms=['legacy','skeleton-k4','skeleton-k8','skeleton-k16','skeleton-kW']
def arm_dir(model,seq,arm):
    return HERE/('legacy_r8_domain' if arm=='legacy' else 'matrix')/f'{model}_s{seq}'/('' if arm=='legacy' else arm)
def timing(directory):return {r['phase']:r for r in rows(directory/'selected.cu.timing.tsv')}
completed=[(m,s,a,arm_dir(m,s,a)) for m,s in real_cells for a in arms if (arm_dir(m,s,a)/'selected.cu.timing.tsv').exists()]
sk=[entry for entry in completed if entry[2]!='legacy']
check('C-1','import.gemms.assign(model.gemms.size(' not in new and 'ClassGranularity' in new,hits(SEARCH,'ClassGranularity'))
check('C-2',len(sk)==32 and all(int(timing(d)['import']['count'])==1 for *_,d in sk),f'completed skeleton cells={len(sk)}/32; import counts='+str([(m,s,a,timing(d).get('import')) for m,s,a,d in sk]))
forbidden=[]
for path in [PLACE,'include/tilemega/Solver/SkeletonPlacement.h',ORACLE,'include/tilemega/Analysis/SymbolicOracle.h']:
    forbidden+=hits(path,r'VisitFiniteRelation|\.successors')
check('C-3',not forbidden,forbidden or 'no flat adjacency in placement/Oracle source or headers')
check('C-4',not re.search(r'\bw\s*<\s*grid',body(placement,'ScheduleBySkeleton')) and 'for(int w:candidates)' in placement,hits(PLACE,r'for\(int w:candidates'))
check('C-5','CouplingCache cache' in new and 'cache.Derive' in new and bool(sk) and all(int(timing(d).get('cache_hit',{}).get('count',0))>0 for *_,d in sk),hits(SEARCH,r'CouplingCache|cache.Derive')+[(m,s,a,timing(d).get('cache_hit')) for m,s,a,d in sk])
check('C-6','Serialize()' in source(CACHE) and 'SemanticSignature' in source(CACHE) and 'element_reads' in source(CACHE),hits(CACHE,r'Serialize\(|SemanticSignature|element_reads'))
check('C-7',all(p in oracle for p in ['isl_map_is_bijective','isl_map_is_injective','isl_map_is_single_valued']),hits(ORACLE,r'isl_map_is_(bijective|injective|single_valued)'))
check('C-8','isl_pw_multi_aff_from_map' in oracle,hits(ORACLE,'isl_pw_multi_aff_from_map'))
check('C-9',bool(re.search(r'else if\(isl_set_is_box.*?isl_set_dim_min.*?isl_set_dim_max',oracle,re.S)),hits(ORACLE,r'isl_set_is_box|isl_set_dim_min|isl_set_dim_max'))
check('C-10',not re.search(r'lexmin|lexmax',oracle,re.I),'Oracle has no lexicographic range construction')
check('C-11','std::tie(est,negative_rank,stage_order,tile)' in placement and 'std::priority_queue<Ready>' in placement and not re.search(r'sort\((?:tasks|nodes|all|order)\.',placement),hits(PLACE,r'tie\(est|priority_queue|sort\('))
def phase_order():
    invalid=[]
    for m,s,a,d in sk:
        probed=set()
        for row in rows(d/'selected.cu.phases.tsv'):
            if row['phase']=='resource_probe':probed.add(row['candidate'])
            elif row['phase']=='skeleton' and row['candidate'] not in probed:invalid.append((str(d),row))
    check('C-12',len(sk)==32 and not invalid,f'cells={len(sk)}/32 invalid={invalid}')
safe('C-12',phase_order)
check('C-13',len(sk)==32 and all(int(timing(d).get('megakernel_compile',{}).get('count',99))<=5 for *_,d in sk),[(m,s,a,timing(d).get('megakernel_compile')) for m,s,a,d in sk])
new_files=[SEARCH,PLACE,ORACLE,'lib/Solver/PlanSkeleton.cpp','include/tilemega/Solver/OperatorClasses.h']
check('C-14',all(not re.search(r'isl_schedule_\w+\s*\(',source(p)) for p in new_files),'no ISL scheduler calls in skeleton search chain')
check('C-15','SkeletonOp' in source('include/tilemega/Dialect/CouplingGraph/ExecOps.td') and bool(sk) and all('tmexec.skeleton' in read(d/'selected.mlir') for *_,d in sk),hits('include/tilemega/Dialect/CouplingGraph/ExecOps.td','SkeletonOp')+[str(d/'selected.mlir') for *_,d in sk])
control=['include/tilemega/Solver/JointPlacement.h','lib/Solver/EftPlacement.cpp','lib/Solver/ChainPlacement.cpp','lib/Solver/BalancedPlacement.cpp','lib/Solver/WavefrontPlacement.cpp','lib/Solver/PlanMaterialize.cpp']
check('C-16',all(p not in changed for p in control) and 'SolvePlacementCatalog' in source(control[0]) and 'ScheduleByEarliestFinish' in source(control[1]),f'unchanged control sources={control}; changed intersection={set(control)&set(changed)}')
variant_counts=[]
for *_,d in sk:
    for p in d.glob('selected.cu.top*.cu'):
        match=re.search(r'#define TILEMEGA_GEMM_VARIANT_COUNT (\d+)',read(p))
        if match:variant_counts.append((str(p),int(match[1])))
check('C-17',any(n>1 for p,n in variant_counts),variant_counts)
check('C-18',bool(outer) and 'SimulateExecution' not in outer and 'SimulateExecution' not in new,hits(SEARCH,'CoordinateDescent')+hits('lib/Solver/SkeletonFinalize.cpp','SimulateExecution'))
commands=list(HERE.rglob('*command*.json'))+list(HERE.rglob('build_command.txt'));bad_commands=[]
missing_zero=[]
for path in commands:
    text=read(path)
    if re.search(r'TILEMEGA_MIDPOINT_REFINE(?:=|\s+)1\b',text):bad_commands.append(str(path))
    if 'nvcc' in text and 'TILEMEGA_MIDPOINT_REFINE=0' not in text:missing_zero.append(str(path))
check('C-19',bool(commands) and not bad_commands and not missing_zero,f'command files={len(commands)} forbidden={bad_commands} nvcc_missing_explicit_zero={missing_zero}')
body_changes=[p for p in changed if re.search(r'TaskBody.*\.h$',p)]
check('C-20',not body_changes,f'git diff {BASE} --name-only: TaskBody changes={body_changes}')
check('G-1',all(results.get(f'C-{i}',False) for i in range(1,21)),'all twenty code and dynamic structure checks')

def parse_process(path):
    text=read(path);h=re.search(r'E2E_HASH l05=(\w+) l1=(\w+) l2=(\w+)',text)
    t=re.search(r'E2E_TIME l05_ms=([\d.]+) l1_ms=([\d.]+).*? l2_ms=([\d.]+)',text)
    good=bool(h and len(set(h.groups()))==1 and 'l1_vs_l05_mismatch=0' in text and 'l2_vs_l1_mismatch=0' in text)
    return good,tuple(map(float,t.groups())) if t else None

def measure_dir(path):
    logs=sorted(path.glob('process_*.log'));samples=[];good=len(logs)==10
    for p in logs:
        ok,t=parse_process(p);good=good and ok and t is not None
        if t:samples.append(t)
    return good,tuple(map(statistics.median,zip(*samples))) if len(samples)==10 else None
measurements={};failures=[]
for m,s in real_cells:
    for arm in arms:
        d=arm_dir(m,s,arm)
        directories=[d/'selected.cu.measurement'] if arm=='legacy' else sorted(d.glob('selected.cu.top*.cu.measurement'))
        observations=[(p,*measure_dir(p)) for p in directories]
        valid=[(p,t) for p,ok,t in observations if ok]
        if not valid or len(directories)!=(1 if arm=='legacy' else 3):failures.append((m,s,arm,[(str(p),ok) for p,ok,t in observations]))
        if valid:measurements[m,s,arm]=min(valid,key=lambda x:x[1][2])
check('G-2',not failures,f'internal 10/10 and three timed shortlist candidates per skeleton arm; missing/failed={failures}')
reference=[]
for m in ('gqa2','mha4'):
    for s in (4,128):
        d=HERE/'reference'/f'{m}_s{s}';samples=sorted(d.glob('selected.cu.top*.cu.measurement'))
        good=bool(samples) and all(measure_dir(p)[0] for p in samples);reference.append((m,s,good,str(d)))
check('G-3',all(x[2] for x in reference),reference)
cache_log=read(HERE/'cache_test.log');check('G-4',all(p in cache_log for p in ['CACHE_EQ split=1','CACHE_EQ split=2']) and 'COLLISION' in cache_log and 'PASS' in cache_log,str(HERE/'cache_test.log')+' '+cache_log.strip())
oracle_log=read(HERE/'oracle_test.log');check('G-5','ORACLE_SET_EQUAL' in oracle_log and 'PASS' in oracle_log and all(k in oracle_log for k in ['Unique','Rectangular','General']),str(HERE/'oracle_test.log')+' '+oracle_log.strip())
check('G-6',results['C-19'] and bool(commands),'build command audit C-19; default header='+str(hits('include/tilemega/Codegen/tasks/ModelRuntime.h','define TILEMEGA_MIDPOINT_REFINE')))
resource_bad=[]
for m,s,a,d in sk:
    table=rows(d/'selected.cu.resources.tsv');log=read(d/'solve.log');actuals=list(map(int,re.findall(r'RESOURCE_QUERY.*resident=(\d+)',log)))
    if len(table)!=5 or len(actuals)!=5:resource_bad.append((str(d),'need five queries and five rows'))
    for i,row in enumerate(table):
        est=int(row['estimated']);actual=int(row['actual']);res=int(row['residency'])
        if actual<1 or res>actual or (actual!=est and int(row['re_solved'])!=1) or i>=len(actuals) or actuals[i]!=actual:resource_bad.append((str(d),row))
check('G-7',len(sk)==32 and not resource_bad,f'completed={len(sk)}/32 raw probe violations={resource_bad}')
wins=[]
for m,s in real_cells:
    base=measurements.get((m,s,'legacy'));new_arms=[(a,measurements[m,s,a]) for a in arms[1:] if (m,s,a) in measurements]
    if base and len(new_arms)==4:
        arm,best=min(new_arms,key=lambda x:x[1][1][2]);ratio=best[1][2]/base[1][2]
        wins.append((m,s,arm,ratio,ratio<=1))
check('G-8',len(wins)==8 and sum(w[4] for w in wins)>=6,f'wins={sum(w[4] for w in wins)}/8; {wins}')
check('G-9',len(completed)==40,f'phase ledgers={len(completed)}/40; '+str([(m,s,a,timing(d).get('total')) for m,s,a,d in completed]))
ratios=[]
for m,s in real_cells:
    wide=measurements.get((m,s,'skeleton-kW'))
    if wide:
        for k in (4,8,16):
            narrow=measurements.get((m,s,f'skeleton-k{k}'))
            if narrow:ratios.append((m,s,k,narrow[1][2]/wide[1][2]))
check('G-10',len(ratios)==24,f'narrow/wide ratios={ratios}')
def selected_dump(model,seq,arm):
    directory=arm_dir(model,seq,arm)
    measured=measurements[model,seq,arm][0]
    source_name=measured.name.removesuffix('.measurement')
    shortlist=rows(directory/'selected.cu.top3.tsv')
    key=next(r['key'] for r in shortlist if pathlib.Path(r['source']).name==source_name)
    metrics=next(p for p in directory.glob('selected.cu.final*.metrics.tsv') if rows(p)[0]['key']==key)
    prefix=str(metrics).removesuffix('.metrics.tsv')
    return prefix,rows(metrics)[0]

def queue_report():
    output=[];bad=[]
    for model,seq in real_cells:
        legacy=arm_dir(model,seq,'legacy')/'eft_queue.tsv'
        if not legacy.exists():bad.append(str(legacy));continue
        control=interleaving(legacy)
        for arm in arms[1:]:
            if (model,seq,arm) not in measurements:bad.append((model,seq,arm));continue
            prefix,metric=selected_dump(model,seq,arm)
            actual=interleaving(prefix+'.tasks.tsv')
            for field in ('transitions','adjacent_slots','placed'):
                measured_field='tasks' if field=='placed' else field
                if actual[measured_field]!=int(metric[field]):bad.append((prefix,field))
            if abs(actual['interleaving']-float(metric['interleaving']))>1e-10:bad.append((prefix,'interleaving'))
            output.append((model,seq,arm,actual,control))
    check('G-11',len(output)==32 and not bad,f'selected queue transitions versus control EFT={output}; missing/inconsistent={bad}')
safe('G-11',queue_report)

def attention_report():
    output=[];bad=[]
    for model,seq in real_cells:
        if seq not in (16,64):continue
        control=measurements.get((model,seq,'legacy'))
        for arm in arms[1:]:
            value=measurements.get((model,seq,arm))
            if not control or not value:bad.append((model,seq,arm));continue
            prefix,metric=selected_dump(model,seq,arm)
            path=attention_path(prefix+'.tasks.tsv',prefix+'.edges.tsv')
            if abs(path['cp_ns']-float(metric['cp_ns']))>max(1e-6,1e-8*path['cp_ns']):bad.append((prefix,'CP recurrence disagreement'))
            _,times=value;_,base=control
            output.append((model,seq,arm,path,dict(l2_l1=times[2]/times[1],l2_l05=times[2]/times[0],legacy_l2_l1=base[2]/base[1],legacy_l2_l05=base[2]/base[0])))
    check('G-12',len(output)==16 and not bad,f'simulated dependency CP and raw timing ratios={output}; missing/inconsistent={bad}')
safe('G-12',attention_report)
def class_report():
    output=[];bad=[]
    for model,seq,arm,directory in sk:
        if (model,seq,arm) not in measurements:bad.append((model,seq,arm,'not measured'));continue
        prefix,metric=selected_dump(model,seq,arm);classes=rows(prefix+'.classes.tsv')
        measured=measurements[model,seq,arm][0];source_path=pathlib.Path(str(measured).removesuffix('.measurement'))
        text=read(source_path)
        def macro(name):
            match=re.search(r'^#define '+name+r' (\d+)$',text,re.M)
            if not match:raise ValueError('missing macro '+name)
            return int(match[1])
        count=macro('TILEMEGA_GEMM_VARIANT_COUNT')
        variants=[]
        for variant in range(count):
            stem='TILEMEGA_GEMM_' if variant==0 else f'TILEMEGA_GEMM_V{variant}_'
            variants.append(tuple(macro(stem+key) for key in ('TILE_M','TILE_N','TILE_K','STAGES')))
        table=re.search(r'constexpr GemmRuntimeDesc kRuntimeGemms0\[\] = \{(.*?)\n\};',text,re.S)
        if not table:raise ValueError('missing runtime variant table')
        records=[tuple(map(int,re.findall(r'(\d+)u',line))) for line in table[1].splitlines() if '{' in line]
        if len(records)!=len(classes):bad.append((prefix,'GEMM count differs'))
        for row in classes:
            g=int(row['gemm']);v,split,m,n,k,stages=records[g]
            expected=tuple(int(row[x]) for x in ('tile_m','tile_n','tile_k','stages'))
            if v>=count or variants[v]!=expected or (m,n,k,stages)!=expected or split!=int(row['split_k']):bad.append((prefix,g,'variant mismatch'))
        if len(set(variants))!=count or len(set(tuple(r[2:]) for r in records))!=count:bad.append((prefix,'unused/duplicate variant'))
        compact={}
        for row in classes:
            compact.setdefault(row['class'],{key:row[key] for key in ('tile_m','tile_n','tile_k','stages','split_k','seed_m','seed_n','seed_k','seed_stages','seed_split')})
        output.append((model,seq,arm,count,metric['residency'],metric['grid'],compact))
    check('G-13',len(output)==32 and not bad and results['G-7'],f'measured winner classes and codegen mapping={output}; violations={bad}')
safe('G-13',class_report)
hard=['G-1','G-2','G-3','G-4','G-5','G-6','G-7']
print('VERIFY hard_failed='+','.join(k for k in hard if not results[k])+' research_pass='+str(int(results['G-8'])))
sys.exit(0 if all(results[k] for k in hard) else 1)
