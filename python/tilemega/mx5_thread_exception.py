"""Candidate-bound CPU audit only. Never enables legacy 512 execution or runs CUDA."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import ctypes
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from . import mx5_plan_resources as legacy
from . import mx5_plan_occupancy as occupancy
from .mx5_plan_receipt import load_native
from .mx5_select import check_file, record, validate_preview
from .build.identity import digest
from .mx5_resources import inspect

CLASSES=legacy.CLASSES
STAGES=legacy.STAGES
SPEC_DIFFERENCE='observed_576_budget_168_not_ap_total_227'
ROOT=Path(legacy.__file__).resolve().parents[2]


class Action(str,Enum):
    BLOCKED='blocked_evidence'
    USE_576='use_576_solver_path'
    NEEDS_COSTS='needs_stage_measurements'
    NEEDS_D7='needs_d7_review'
    NEXT_WIDTH='next_width_diagnostic'
    READY='review_ready_bound_descendant'
    COST_BOUND='descendant_costs_bound'


@dataclass(frozen=True)
class Outcome:
    action: Action
    reason: str
    frozen: legacy.FrozenJSON

    @property
    def data(self):return self.frozen.data

    def as_dict(self):
        return dict(kind='mx5_thread_exception_audit',schema_version=1,action=self.action.value,
                    reason=self.reason,details=self.data,execution_authorized=False,default_enabled=False)


def _out(action,reason,**details):
    return Outcome(action,reason,legacy.FrozenJSON.freeze(dict(execution_authorized=False,final=False,
                    spec_difference=SPEC_DIFFERENCE,validator_source=record(Path(__file__)),**details)))


def _need(condition,message):
    if not condition:raise ValueError(message)


def _int(value,name,minimum=0):
    _need(type(value) is int and value>=minimum,'invalid integer '+name);return value


def _load(ref):return json.loads(check_file(ref).read_text())


@dataclass(frozen=True)
class Origin:
    frozen: legacy.FrozenJSON

    @classmethod
    def capture(cls,preview_ref):
        path=check_file(preview_ref);preview=validate_preview(path)
        _need(preview['request']['threads']==512,'this bounded exception requires a native qualified512 origin')
        count=None
        for candidate in preview['candidates']:
            _need(candidate['pin']['mx_decode_v2']['threads']==512,'origin candidates must keep their native512 thread scope')
            _need(len(candidate['ranks'])==6 and all(type(x) is int and x>=0 for x in candidate['ranks']),'invalid native ranks')
            stages=candidate['predicted_stage_ns'];ns=candidate['predicted_tpot_ns']
            for field in STAGES:legacy._positive(stages[field],field)
            legacy._positive(ns,'origin prediction')
            layers=(ns-stages['head'])/sum(stages[x] for x in STAGES if x!='head')
            _need(round(layers)>=1 and abs(layers-round(layers))<1e-9,'invalid origin layer prediction')
            _need(count is None or count==round(layers),'origin layer counts differ');count=round(layers)
        return cls(legacy.FrozenJSON.freeze(dict(preview_ref=preview_ref,preview=preview,layers=count)))

    @property
    def data(self):return self.frozen.data
    @property
    def sha256(self):return self.frozen.sha256
    def verify(self):_need(Origin.capture(self.data['preview_ref']).frozen==self.frozen,'native origin changed')
    def candidate(self,index):
        _need(type(index) is int and index in range(4),'invalid original candidate ID')
        return self.data['preview']['candidates'][index]


def _configuration(source,origin,candidate_id,pin):
    request=origin.data['preview']['request'];candidate=origin.candidate(candidate_id);c=pin['mx_decode_v2']
    constants=dict(kLayers=origin.data['layers'],kThreads=c['threads'],kGrid=104,kBatch=request['batch'],
                   kCapacity=request['capacity'],kPastLo=request['past_lo'],kPastHi=request['past_hi'])
    for name,expected in constants.items():
        values=re.findall(r'\b'+name+r'\s*=\s*(\d+)',source)
        _need(values==[str(expected)],'actual generated constant differs: '+name)
    def alias(name,expected):
        found=re.findall(r'\busing\s+'+name+r'\s*=\s*([^;]+);',source)
        _need(len(found)==1 and re.sub(r'\s+','',found[0])==expected,'actual generated alias differs: '+name)
    for name,emitted in zip(CLASSES,('Qkv','O','GateUp','Down','Head')):
        item=c[name];key=candidate['evidence'][name][0]['key']
        args='%d,%d,%d'%(item['tn'],item['wn'],item['d'])
        if name=='head':body='MxHead<'+args+','+str(item['prefetch'])+'>'
        else:
            body='MxGemm<'+args+',MxOpV2::'+('kStore','kResidual','kSwiGLU')[key['op']]+','+('true' if key['norm'] else 'false')+',MxPart::'+item['part']+','+str(item['prefetch'])+'>'
        alias(emitted,body);alias(emitted+'Shape','MxStaticShapeV2<%d,%d,kBatch,kThreads>'%(key['n'],key['k']))
        if name=='down':alias('DownLast',body)
    key=candidate['evidence']['attention'][0]['key']
    alias('Attn','MxAttentionV2<%d,%d,%d,%s,%d,kCapacity>'%(key['dim'],key['q'],c['b_min'],'true' if key['qk_norm'] else 'false',c['attn_prefetch']))
    alias('AttnShape','MxStaticAttentionShapeV2<kBatch,%d,kThreads>'%key['hkv'])


def _geometry(origin_pin,pin,*,minimum=False):
    expected=json.loads(json.dumps(origin_pin));actual=json.loads(json.dumps(pin));e=expected['mx_decode_v2'];a=actual['mx_decode_v2']
    for name in CLASSES:
        for field in ('tn','wn','d','part'):_need(a[name][field]==e[name][field],'original geometry changed: '+name+'.'+field)
        _int(a[name]['prefetch'],'Pw');_need(a[name]['prefetch']>=a[name]['d'],'Pw below D')
        if minimum:
            _need(a[name]['prefetch']==a[name]['d'],'minimum576 has a nonfloor Pw')
            _need(a[name].get('r_next',-1)==-1,'minimum floor probe supports honest unset R_next only; positive576 references need an explicit adapter')
        else:_need(a[name].get('r_next')==e[name]['r_next'],'512 descendant changed qualified R_next')
    _need(a['b_min']==e['b_min'],'b_min changed')
    _need(a['threads']==(576 if minimum else 512),'wrong artifact thread role')
    _need(a['attn_prefetch']==(0 if minimum else e['attn_prefetch']),'attention PF changed')
    # Other pin metadata must remain exact; the floor probe has only a diagnostic purpose change.
    for name in CLASSES:
        e[name]['prefetch']=a[name]['prefetch']
        if minimum:
            e[name].pop('r_next',None);a[name].pop('r_next',None)
    if minimum:e.update(threads=576,attn_prefetch=0,purpose='resource_probe')
    _need(legacy.FrozenJSON.freeze(actual)==legacy.FrozenJSON.freeze(expected),'non-width pin metadata changed')


def _mir(log_ref,kernel):
    path=check_file(log_ref);text=path.read_text(errors='replace')
    headers=list(re.finditer(r'^# Machine code for function (\S+):',text,re.M))
    events=list(re.finditer(r'^# (Machine code for function (\S+):|End machine code for function (\S+)\.$)',text,re.M))
    _need(len(events)==len(re.findall(r'^# (?:Machine code|End machine code) for function\b',text,re.M)),
          'malformed/truncated MIR function boundary marker')
    active=None;complete=[]
    for event in events:
        if event[2] is not None:
            _need(active is None,'incomplete MIR function before another header')
            active=event
        else:
            _need(active is not None and active[2]==event[3],'missing/mismatched MIR function header/end')
            complete.append((active[2],active.end(),event.start()));active=None
    _need(headers and active is None and len(complete)==len(headers),'truncated/incomplete MIR function coverage')
    matches=[x for x in complete if x[0]==kernel]
    _need(len(matches)==1,'missing/duplicate exact kernel MIR')
    body=text[matches[0][1]:matches[0][2]]
    blocks=re.findall(r'^bb\.(\d+)',body,re.M)
    _need(blocks and len(blocks)==len(set(blocks)),'missing/duplicate exact kernel MIR blocks')
    successors=[v for line in body.splitlines() if line.strip().startswith('successors:')
                for v in re.findall(r'%bb\.(\d+)',line)]
    _need(set(successors)<=set(blocks),'incomplete exact kernel MIR CFG successor coverage')
    with tempfile.TemporaryDirectory(prefix='mx5_exception_mir_') as directory:
        output=Path(directory)/'report.json'
        result=subprocess.run([sys.executable,str(ROOT/'experiments/mx5/mir_check.py'),str(path),'--out',str(output)],capture_output=True,text=True)
        _need(result.returncode in (0,1) and output.is_file(),'trusted MIR parser failed: '+result.stderr[-300:])
        report=json.loads(output.read_text())
    _need(isinstance(report,dict) and isinstance(report.get('functions'),list)
          and [x.get('function') for x in report['functions']]==[x[0] for x in complete],
          'trusted MIR parser has incomplete function coverage')
    functions=[x for x in report['functions'] if x['function']==kernel]
    _need(len(functions)==1,'MIR CFG report does not identify the exact kernel')
    return dict(addrspace0=len(re.findall(r'\baddrspace\s*(?:\(\s*)?0\b',body)),generic=len(re.findall(r'\b(?:LDX|STX)_\w+',body)),calls=len(re.findall(r'\b(?:CALL\w*|S_CALL\w*|TAILCALL\w*)',body)),
                ring4160=sum(bool(x['wait_all']) for x in functions[0]['rings']),ring_generic=sum(x['generic_loads'] for x in functions[0]['rings']),
                private=functions[0]['private_accesses'],rings=len(functions[0]['rings']),parser=record(ROOT/'experiments/mx5/mir_check.py'))


def _normal_options(receipt,files,snapshot,watchdog):
    argv=receipt['argv'];generated=str(check_file(files['generated_source']));binary=str(check_file(files['binary']))
    library=[x for x in argv if x.startswith('-L')];_need(len(library)==1 and Path(library[0][2:]).is_absolute(),'invalid native CUDA library option')
    expected=[argv[0],'-std=c++17','-O3','-Xcompiler','-resource-usage','-fno-strict-aliasing',
              '-DTILEMEGA_SYNC_V3=1','-DTILEMEGA_MX_STATIC_DECODE_V2=1','-DTILEMEGA_WATCHDOG='+str(watchdog),
              '-DTILEMEGA_TRACE_STAGE=0','-DTILEMEGA_MX_DIAG=0','-mllvm','-print-after=stack-frame-layout',
              '-shared','-Xcompiler=-fPIC','-x','cu','-I'+str(Path(snapshot['root'])/'include'),generated,
              library[0],'-lcudart','-o',binary]
    _need(argv==expected,'native command differs from the normal role policy; overrides are diagnostic only')
    return [('<source>' if x==generated else '<output>' if x==binary else '-DTILEMEGA_WATCHDOG=<role>' if x.startswith('-DTILEMEGA_WATCHDOG=') else x) for x in argv]


def _query(query_ref,files,resource_ref,threads,report):
    row=_load(query_ref);spec=row['bound_inputs']
    _need(row.get('kind')=='mx5_plan_occupancy' and row.get('schema_version')==1,'wrong zero-query schema')
    for field in ('query_status','attributes_returncode','set_dynamic_shared_returncode','occupancy_returncode'):
        _need(_int(row.get(field),field)==0,'actual runtime query failed: '+field)
    _need(row.get('query_only') is True and row.get('no_launch') is True and _int(row.get('iterations'),'iterations')==0
          and _int(row.get('launches'),'launches')==0,'query is not a genuine zero-launch attribute observation')
    _need(row.get('execution_authorized') is False and row.get('grants_512_exception') is False,'query cannot grant execution')
    _need(row['threads']==threads and row['binary_sha256']==files['binary']['sha256'],'query thread/binary binding mismatch')
    _need(_load(row['inputs_manifest'])==spec,'query input manifest changed')
    expected=dict(binary=files['binary'],generated_source=files['generated_source'],log=files['log'],source_snapshot=files['source_snapshot'],resources=resource_ref)
    for name,ref in expected.items():_need(spec[name]==ref,'query input differs: '+name)
    _need(spec['context_kind']=='compile_receipt' and spec['context']['sha256']==files['compile_receipt']['sha256'],'query lacks same native compile context')
    validated=occupancy.validate_inputs(spec,command=occupancy.run_text)
    _need(row.get('registered_host_stub')==validated['stub'] and row.get('dynamic_shared_bytes')==validated['expected_shared_bytes'],'query stub/shared-byte observation differs from actual ELF host code')
    _need(check_file(row['helper_source'])==Path(occupancy.__file__).resolve(),'query helper is not the trusted implementation')
    _need(check_file(row['runtime_library'])==occupancy.RUNTIME.resolve(),'query runtime identity mismatch')
    attrs=row['attributes'];raw=bytes.fromhex(row['attributes_raw_hex'])
    _need(len(raw)==ctypes.sizeof(occupancy.Attributes),'wrong runtime attribute ABI')
    decoded=occupancy.Attributes.from_buffer_copy(raw)
    _need(attrs=={n:getattr(decoded,n) for n,_ in occupancy.Attributes._fields_},'raw runtime attributes disagree')
    _need(attrs['maxThreadsPerBlock']==threads and attrs['numRegs']==report['kernel']['registers']
          and attrs['localSizeBytes']==report['kernel']['stack_bytes'] and attrs['sharedSizeBytes']==report['kernel']['static_smem_bytes'],'runtime attributes disagree with ELF/MIR')
    _int(row['resident_ctas'],'resident_ctas');_need(row['resident_ctas']==row['occupancy_raw_value'],'raw occupancy value differs')
    _need(row['dynamic_shared_bytes']<=65536 and row['dynamic_shared_bytes']>0,'invalid actual shared-byte query')
    guard=row['guard'];gpu=guard['physical_gpu'];env=guard['environment'];owner=guard['owner']
    _need(gpu in ('4','5','6','7') and all(env.get(k)==gpu for k in ('TILEMEGA_DEVICE_INDEX','CUDA_VISIBLE_DEVICES','MACA_VISIBLE_DEVICES'))
          and env.get('TILEMEGA_GPU_LOCK_HELD')=='1' and guard['lock_path']==env.get('TILEMEGA_GPU_LOCK')=='/tmp/tilemega_mx_gpu'+gpu+'.lock','invalid saved guarded-query environment')
    lock=owner['proc_locks_line'].split()
    _need(len(lock)>=8 and lock[1:4]==['FLOCK','ADVISORY','WRITE'] and int(lock[4])==_int(owner['pid'],'guard PID',2) and 'tilemega.bench.gpu_guard' in owner['command'],'missing saved guard owner proof')
    return dict(ref=query_ref,resident_ctas=row['resident_ctas'],attributes=attrs,runtime=row['runtime_library'],helper=row['helper_source'])


def _artifact(ref,origin,candidate_id,*,minimum=False,watchdog=0,require_query=False):
    bundle=_load(ref)
    _need(set(bundle)=={'kind','schema_version','origin_sha256','candidate_id','ordinal','native_receipt','resources','query','stage_sources'},'unexpected artifact bundle fields; origin predictions cannot be overridden')
    _need(bundle.get('kind')=='mx5_thread_exception_artifact' and bundle.get('schema_version')==1,'wrong artifact bundle schema')
    _need(bundle['origin_sha256']==origin.sha256 and _int(bundle['candidate_id'],'artifact candidate ID')==candidate_id,'artifact origin/candidate mismatch')
    _int(bundle['ordinal'],'artifact ordinal')
    if minimum:_need(bundle['ordinal']==0,'minimum576 is an original-geometry diagnostic, not a retry ordinal')
    _,receipt=load_native(check_file(bundle['native_receipt']));files=dict(receipt['files'],compile_receipt=bundle['native_receipt'])
    _need(receipt['returncode']==0 and files.get('binary') is not None,'device compilation failed; no resource retry/exception evidence')
    pin=_load(files['pin']);resolved=_load(files['resolved_pin'])['mx_decode_v2'];config=pin['mx_decode_v2']
    _geometry(origin.candidate(candidate_id)['pin'],pin,minimum=minimum)
    for name in CLASSES:
        for field in ('tn','wn','d','part','prefetch','r_next'):
            _need(resolved[name].get(field)==config[name].get(field,-1),'resolved pin changed: '+name+'.'+field)
        key=origin.candidate(candidate_id)['evidence'][name][0]['key']
        for field,value in (('n',key['n']),('k',key['k']),('b',origin.data['preview']['request']['batch'])):
            _need(resolved[name].get(field)==value,'resolved shape changed: '+name+'.'+field)
        op='argmax' if name=='head' else ('kStore','kResidual','kSwiGLU')[key['op']]
        _need(resolved[name].get('op')==op and resolved[name].get('norm')==bool(key.get('norm',1)),'resolved operator changed')
    for field in ('threads','b_min','attn_prefetch'):_need(resolved.get(field)==config[field],'resolved '+field+' changed')
    snapshot=_load(files['source_snapshot'])
    _need(snapshot['source_digest']==digest(snapshot['source_files']),'source snapshot digest mismatch')
    legacy._snapshot_inputs(snapshot)
    _need(files['source_patch']['sha256']==snapshot['worktree_diff_sha256'],'source patch mismatch')
    _need(files['producer']['sha256']==snapshot['compiler_sha256']==origin.data['preview']['producer']['sha256'],'producer differs from native origin')
    _need(Path(snapshot['compiler']).resolve()==check_file(files['producer']),'snapshot producer path differs')
    options=_normal_options(receipt,files,snapshot,watchdog)
    _configuration(check_file(files['generated_source']).read_text(),origin,candidate_id,pin)
    checked=set()
    for rows in origin.candidate(candidate_id)['evidence'].values():
        for row in rows:
            source=row['provenance']['source_sha256']
            if source not in checked:
                legacy._verify_archive(row,bundle['stage_sources'],snapshot['source_files']);checked.add(source)
    report=inspect(check_file(files['binary']),check_file(files['log']),config['threads'])
    _need(report==_load(bundle['resources']),'resource report differs from actual ELF/same-compile MIR')
    kernel=report['kernel']
    _need(kernel['name']==occupancy.KERNEL and set(report['kernel_instructions'])=={occupancy.KERNEL}
          and len(report['device_images'])==1,'unsupported kernel/image identity')
    for name in ('registers','scalar_registers','stack_bytes','static_smem_bytes'):_int(kernel[name],name,1 if name=='registers' else 0)
    mir=_mir(files['log'],kernel['name'])
    query=_query(bundle['query'],files,bundle['resources'],config['threads'],report) if bundle.get('query') else None
    _need(query is not None or not require_query,'missing actual zero-launch query')
    family=digest(dict(source=snapshot['source_digest'],patch=files['source_patch']['sha256'],producer=files['producer']['sha256'],compiler=files['compiler']['sha256'],options=options))
    return dict(ref=ref,ordinal=bundle['ordinal'],pin=pin,files=files,resources=kernel,mir=mir,query=query,
                family=family,source_files=snapshot['source_files'],source_digest=snapshot['source_digest'])


def _failure(artifact):
    r=artifact['resources']
    _need(r['static_smem_bytes']==0,'static shared-memory failure is not a width-retry trigger')
    return r['registers']>255 or r['stack_bytes']>0


def _strict_final(artifact):
    r=artifact['resources'];m=artifact['mir']
    _need(not _failure(artifact),'final512 still fails MT/frame gates')
    _need(type(m.get('rings')) is int and m['rings']>0,'complete dense final512 lacks actual natural MMA-loop evidence')
    _need(not r.get('callees') and not any(m[k] for k in ('addrspace0','generic','calls','ring4160','ring_generic')),'final512 fails MIR generic/call/ring gates')
    _need(artifact['query'] is not None and artifact['query']['resident_ctas']==1,'final512 lacks actual resident1 query')


def _next(origin,candidate_id,parent,reduction_step):
    _need(_failure(parent),'parent is not an actual register/stack failure')
    _need(1<=reduction_step<=3,'three whole-plan reductions exhausted')
    pin=json.loads(json.dumps(parent['pin']));c=pin['mx_decode_v2'];reducible=[name for name in CLASSES if c[name]['prefetch']>c[name]['d']]
    _need(reducible,'all Pw equal D and 512 still fails; D7/defect, no replacement candidate')
    name=min(reducible,key=lambda x:(-c[x]['prefetch'],CLASSES.index(x)))
    before=c[name]['prefetch'];after=max(c[name]['d'],before//2);c[name]['prefetch']=after
    authorization=dict(kind='plan_resource_retry',schema_version=1,origin_sha256=origin.sha256,candidate_id=candidate_id,
        parent_attempt_sha256=parent['ref']['sha256'],parent_binary_sha256=parent['files']['binary']['sha256'],
        program_source_epoch=parent['family'],reduction_step=reduction_step,shared_class=name,from_pw=before,to_pw=after,
        previous_pin_sha256=digest(parent['pin']),next_pin_sha256=digest(pin),affected_edges=legacy.affected_edges(name,origin.data['layers']),
        thread_exception_policy=SPEC_DIFFERENCE,threads=512)
    return pin,authorization


def _lineage(origin,candidate_id,refs):
    _need(1<=len(refs)<=4,'require original512 plus at most three width children')
    artifacts=[];authorizations=[]
    for index,ref in enumerate(refs):
        artifact=_artifact(ref,origin,candidate_id)
        _need(artifact['ordinal']==index,'noncontiguous actual attempt ordinals')
        if index==0:_need(artifact['pin']==origin.candidate(candidate_id)['pin'],'lineage does not start at the immutable native pin')
        else:
            previous=artifacts[-1];pin,authorization=_next(origin,candidate_id,previous,index)
            _need(artifact['family']==previous['family'],'retry source/compiler/options epoch changed')
            _need(artifact['pin']==pin,'child is not the deterministic width-only largest-Pw halving')
            authorizations.append(authorization)
        artifacts.append(artifact)
    return artifacts,authorizations


def next_width_diagnostic(origin,candidate_id,lineage_refs):
    try:
        origin.verify();artifacts,auth=_lineage(origin,candidate_id,lineage_refs)
        pin,authorization=_next(origin,candidate_id,artifacts[-1],len(artifacts))
        return _out(Action.NEXT_WIDTH,'exact CPU resource diagnostic only; new costs/execution not authorized',next_pin=pin,authorization=authorization)
    except Exception as error:return _out(Action.BLOCKED,str(error))



def _fresh(entry,row):
    if entry.get('evidence_mode')=='ordinary_completed_original_batch_v1':
        from .mx5_ordinary_campaign import verify
        verify(entry,row)
        return
    _need(entry.get('evidence_mode') in (None,'retained_process_pids_v1'),'unknown variant fresh-process evidence mode')
    processes=entry.get('fresh_processes');_need(isinstance(processes,list) and len(processes)==3,'variant lacks actual three-process PID/command evidence')
    pids=set();rounds=set()
    for ref in processes:
        process=_load(ref);number=_int(process.get('round'),'process round',1);pid=_int(process.get('pid'),'fresh process PID',1)
        _need(number in (1,2,3) and number not in rounds and pid not in pids,'variant process rounds/PIDs are not three distinct fresh processes')
        _need(_int(process.get('returncode'),'process returncode')==0,'variant process failed')
        rounds.add(number);pids.add(pid);argv=process.get('argv',process.get('command'))
        _need(isinstance(argv,list) and all(isinstance(x,str) for x in argv),'variant process command missing')
        sample=next(x for x in row['rounds'] if x['round']==number)
        for flag,expected in (('--out',str(check_file(sample['input']))),('--round',str(number))):
            _need(argv.count(flag)==1 and argv.index(flag)+1<len(argv),'variant process command lacks '+flag)
            actual=argv[argv.index(flag)+1]
            _need((str(Path(actual).resolve()) if flag=='--out' else actual)==expected,'variant process does not bind raw round output')
        _need(argv.count('--keys')==1 and argv.index('--keys')+1<len(argv),'variant process lacks actual keys')
        keys_path=check_file(entry['keys'])
        _need(Path(argv[argv.index('--keys')+1]).resolve()==keys_path,'variant process measured another keys file')
        keys=json.loads(keys_path.read_text())
        _need(argv.count('--iterations')==1 and argv.index('--iterations')+1<len(argv)
              and argv[argv.index('--iterations')+1]==str(sample['measurement']['iterations']),'variant process iteration protocol differs from its raw row')
        qualifier=legacy._load_qualifier(entry['qualification_tool'])
        _need(sum(qualifier.canonical(dict(key=x),row['kind'])==row['key'] for x in keys)==1,'variant measured keys do not contain exact changed width')
    guard=entry.get('guard',{});result=_load(guard['result']);journal=check_file(guard['journal']).read_text().splitlines()
    _need(type(result.get('code')) is int and result['code']==0,'variant guard did not complete successfully')
    observations=[json.loads(line) for line in journal if line.strip()]
    _need(observations and observations[-1].get('phase')=='after' and type(observations[-1].get('exit_code')) is int and observations[-1]['exit_code']==0,'variant guard lacks successful final observation')


def _costs(origin,candidate_id,artifacts,authorizations,catalog_refs):
    original=origin.candidate(candidate_id);stages=dict(original['predicted_stage_ns']);final=artifacts[-1]
    catalogs=[legacy.VariantCatalog.load(check_file(ref)) for ref in catalog_refs]
    latest={x['shared_class']:x for x in authorizations};bound={};missing=[]
    for name,authorization in latest.items():
        base=original['evidence'][name][0];key=dict(base['key'],prefetch=final['pin']['mx_decode_v2'][name]['prefetch'])
        request=dict(kind=base['kind'],key=key);matches=[]
        for catalog in catalogs:
            row=catalog.lookup(request,authorization,final['source_files'])
            if row is not None:
                document=catalog.frozen.data['document']
                entry=next(x for x in document['rows'] if x['row_sha256']==row.data['row_sha256'] and x['authorization']==authorization)
                _fresh(dict(entry,qualification_tool=document['qualification_tool']),row.data['row'])
                matches.append(row.data)
        _need(len(matches)<=1,'ambiguous exact authorized cost binding')
        if not matches:
            missing.append(dict(shared_class=name,**request,authorization=authorization,required_fresh_processes=3,
                                missing_kind='authorized_exact_width_cost_binding'));continue
        row=matches[0]['row']
        _need(row['provenance']['source_sha256']==base['provenance']['source_sha256'] and row['r_next']==base['r_next'],'variant source/R_next differs from origin')
        stages[name]=row['median_ns'];bound[name]=matches[0]
    if missing:return dict(missing=missing,prediction=None)
    total=stages['head']+origin.data['layers']*sum(stages[x] for x in STAGES if x!='head')
    return dict(missing=[],prediction=dict(original_candidate_id=candidate_id,original_ranks=original['ranks'],origin_predicted_tpot_ns=original['predicted_tpot_ns'],
       effective_predicted_tpot_ns=total,effective_stage_ns=stages,effective_variant_rows=bound,selection_anchor=0,
       ranking_claim='verified descendant of original512 top4; no repaired global-ranking claim'))


def bind_descendant_costs(origin,candidate_id,lineage_refs,catalog_refs=()):
    try:
        origin.verify();artifacts,auth=_lineage(origin,candidate_id,lineage_refs);result=_costs(origin,candidate_id,artifacts,auth,catalog_refs)
        return _out(Action.NEEDS_COSTS if result['missing'] else Action.COST_BOUND,'cost binding only; no resource or thread exception approval',**result)
    except Exception as error:return _out(Action.BLOCKED,str(error))


def _d7_history(receipt_ref,origin,candidate_id):
    _,value=load_native(check_file(receipt_ref));files=dict(value['files'],compile_receipt=receipt_ref)
    _need(value['returncode']==0 and files.get('binary'),'D7 history lacks an actual successful compile')
    pin=_load(files['pin']);c=pin['mx_decode_v2'];base=origin.candidate(candidate_id)['pin']['mx_decode_v2']
    _need(c['threads']==576,'D7 before/after must inspect actual576 attempts')
    for name in CLASSES:
        for field in ('tn','wn','d','part'):_need(c[name][field]==base[name][field],'D7 history belongs to another geometry')
    _need(c['b_min']==base['b_min'],'D7 history changes attention partition')
    _configuration(check_file(files['generated_source']).read_text(),origin,candidate_id,pin)
    snapshot=_load(files['source_snapshot']);_need(snapshot['source_digest']==digest(snapshot['source_files']),'D7 source inventory digest differs')
    _need(snapshot['worktree_diff_sha256']==files['source_patch']['sha256'],'D7 patch identity differs')
    _need(snapshot['compiler_sha256']==files['producer']['sha256'] and Path(snapshot['compiler']).resolve()==check_file(files['producer']),'D7 snapshot producer identity differs')
    report=inspect(check_file(files['binary']),check_file(files['log']),576)
    _need(report['kernel']['name']==occupancy.KERNEL,'D7 report has another kernel')
    return dict(files=files,receipt=value,pin=pin,snapshot=snapshot,resources=report['kernel'],mir=_mir(files['log'],occupancy.KERNEL))


def _d7(ref,origin,candidate_id,minimum,final):
    doc=_load(ref)
    _need(doc.get('kind')=='mx5_candidate_d7_review' and doc.get('schema_version')==1,'D7 requires a substantive candidate-bound review, not a confirmed boolean')
    for field,value in dict(origin_sha256=origin.sha256,candidate_id=candidate_id,family_sha256=minimum['family'],
            spec_difference=SPEC_DIFFERENCE,minimum576_binary_sha256=minimum['files']['binary']['sha256'],
            final512_binary_sha256=final['files']['binary']['sha256']).items():
        _need(doc.get(field)==value,'D7 scope/binary binding differs: '+field)
    _need(doc.get('assessment')=='normal_floor_576_still_spills_at_observed_168'
          and doc.get('not_claimed')=='required_registers_gt227_or_global_576_impossible','D7 must state the narrow observation and exclude the false lower-bound claim')
    text=check_file(doc['report']).read_text()
    _need(all(token in text for token in (SPEC_DIFFERENCE,origin.sha256,minimum['files']['binary']['sha256'],final['files']['binary']['sha256'])),'D7 report text does not bind this candidate evidence')
    steps=doc.get('steps');_need(isinstance(steps,list) and steps,'D7 lacks before/after analysis evidence')
    for step in steps:
        _need(step.get('kind') in ('source_repair','budget_diagnostic') and isinstance(step.get('analysis'),str) and step['analysis'].strip(),'D7 step lacks a concrete analysis')
        before=_d7_history(step['before_native'],origin,candidate_id);after=_d7_history(step['after_native'],origin,candidate_id)
        _need(before['files']['binary']['sha256']!=after['files']['binary']['sha256'],'D7 before/after cannot be the same artifact')
        _need(after['snapshot']['source_digest']==minimum['source_digest']
              and after['files']['producer']['sha256']==minimum['files']['producer']['sha256']
              and after['files']['compiler']['sha256']==minimum['files']['compiler']['sha256'],'D7 after evidence is not the current source/compiler family')
        leaves=step.get('source_evidence');_need(isinstance(leaves,list) and leaves,'D7 lacks actual before/after source bytes')
        for leaf in leaves:
            path=Path(leaf['source_path']);_need(not path.is_absolute() and '..' not in path.parts,'invalid D7 source path')
            for role,node in (('before',before),('after',after)):
                check_file(leaf[role]);_need(node['snapshot']['source_files'].get(str(path))==leaf[role]['sha256'],'D7 source bytes differ from actual compile snapshot')
        if step['kind']=='source_repair':
            _need(any(x['before']['sha256']!=x['after']['sha256'] for x in leaves),'D7 source repair has no changed source evidence')
            _normal_options(after['receipt'],after['files'],after['snapshot'],0)
        else:
            _need(before['pin']==after['pin']==minimum['pin'] and before['snapshot']['source_digest']==after['snapshot']['source_digest']
                  and before['files']['producer']['sha256']==after['files']['producer']['sha256']
                  and before['files']['compiler']['sha256']==after['files']['compiler']['sha256'],'D7 budget experiment changes more than the controlled compile option')
            _normal_options(before['receipt'],before['files'],before['snapshot'],0)
            argv=after['receipt']['argv'];pairs=[i for i,x in enumerate(argv) if re.fullmatch(r'--max-mtreg-number=\d+',x)]
            _need(len(pairs)==1 and pairs[0]>0 and argv[pairs[0]-1]=='-mllvm','D7 budget diagnostic lacks its explicit controlled override')
            normalized=dict(after['receipt'],argv=argv[:pairs[0]-1]+argv[pairs[0]+1:])
            _normal_options(normalized,after['files'],after['snapshot'],0)
            resource_ref=step['after_resources'];_need(_load(resource_ref)==inspect(check_file(after['files']['binary']),check_file(after['files']['log']),576),'D7 diagnostic resources differ')
            _query(step['after_query'],after['files'],resource_ref,576,_load(resource_ref))
    return dict(ref=ref,report=doc['report'],steps=len(steps),assessment=doc['assessment'])


def audit_thread_exception(origin,candidate_id,minimum576,lineage512,*,catalogs=(),d7=None,watchdog512=None):
    request=dict(origin_preview=origin.data['preview_ref'],candidate_id=candidate_id,minimum576=minimum576,
                 lineage512=list(lineage512),catalogs=list(catalogs),d7=d7,watchdog512=watchdog512)
    try:
        origin.verify();candidate=origin.candidate(candidate_id)
        minimum=_artifact(minimum576,origin,candidate_id,minimum=True,require_query=True)
        m=minimum['resources'];q=minimum['query']
        facts=dict(original_candidate_id=candidate_id,original_ranks=candidate['ranks'],original_pin=candidate['pin'],
                   original_prediction_ns=candidate['predicted_tpot_ns'],minimum576=dict(binary=minimum['files']['binary'],resources=m,query=q),
                   nominal_dt_threads=576,origin_table_threads=512,selection_anchor=0)
        if m['registers']<=227 and m['stack_bytes']==0 and m['static_smem_bytes']==0 and q['resident_ctas']==1:
            return _out(Action.USE_576,'minimum576 fits; this512 exception is rejected',request=request,facts=facts)
        _need(m['registers']==168 and m['stack_bytes']>0 and m['static_smem_bytes']==0,'minimum576 does not match the narrow observed168 floor-stack branch')
        artifacts,authorizations=_lineage(origin,candidate_id,lineage512);final=artifacts[-1]
        _need(all(x['family']==minimum['family'] for x in artifacts),'minimum576/512 source/compiler/options family differs')
        if _failure(final):
            reducible=any(c['prefetch']>c['d'] for c in final['pin']['mx_decode_v2'].values() if isinstance(c,dict) and 'prefetch' in c)
            reason='all Pw equal D and512 still fails: D7/defect, no candidate replacement' if not reducible else 'final512 still fails; only a verified bounded diagnostic may follow'
            return _out(Action.BLOCKED,reason,request=request,facts=facts,actual_final_resources=final['resources'])
        _strict_final(final)
        if watchdog512 is not None:
            wd=_artifact(watchdog512,origin,candidate_id,watchdog=1,require_query=True)
            _need(wd['pin']==final['pin'] and wd['family']==final['family'] and wd['ordinal']==final['ordinal'],'WD1 is not the exact final descendant/family')
            _strict_final(wd)
        costs=_costs(origin,candidate_id,artifacts,authorizations,catalogs)
        facts.update(final512=dict(binary=final['files']['binary'],pin=final['pin'],resources=final['resources'],query=final['query'],mir=final['mir']),
                     family_sha256=minimum['family'],actual_width_reductions=authorizations)
        if costs['missing']:
            return _out(Action.NEEDS_COSTS,'exact source-qualified changed-Pw costs are not bound; no effective prediction',request=request,facts=facts,**costs)
        if d7 is None:
            return _out(Action.NEEDS_D7,'candidate-bound substantive D7 report is missing; no exception issued',request=request,facts=facts,**costs)
        review=_d7(d7,origin,candidate_id,minimum,final)
        return _out(Action.READY,'all audit bindings verified; execution remains unauthorized in this API version',request=request,facts=facts,d7_review=review,**costs)
    except Exception as error:
        return _out(Action.BLOCKED,str(error),request=request)


def check_bound_exception(audit_ref,origin,candidate_id,artifact_ref,*,purpose):
    try:
        saved=_load(audit_ref)
        _need(saved.get('kind')=='mx5_thread_exception_audit' and saved.get('schema_version')==1
              and saved.get('action')==Action.READY.value and saved.get('execution_authorized') is False,'not a review-ready audit-only record')
        request=saved['details']['request']
        _need(request['origin_preview']==origin.data['preview_ref'] and request['candidate_id']==candidate_id,'consumer origin/ID mismatch')
        _need(purpose in ('guarded_correctness_wd0','guarded_measurement_wd0','guarded_watchdog_wd1'),'unsupported explicit consumer purpose')
        expected=request['watchdog512'] if purpose=='guarded_watchdog_wd1' else request['lineage512'][-1]
        _need(expected is not None and artifact_ref==expected,'consumer artifact/WD role mismatch')
        current=audit_thread_exception(origin,candidate_id,request['minimum576'],request['lineage512'],catalogs=request['catalogs'],d7=request['d7'],watchdog512=request['watchdog512'])
        _need(current.action==Action.READY and legacy.FrozenJSON.freeze(current.as_dict())==legacy.FrozenJSON.freeze(saved),'bound audit/evidence changed at consumption')
        return _out(Action.READY,'binding reverified for future explicit consumer; legacy512 execution remains blocked',
                    audit=audit_ref,artifact=artifact_ref,purpose=purpose,original_candidate_id=candidate_id)
    except Exception as error:return _out(Action.BLOCKED,str(error))
