"""Substantive candidate-bound D7 v2 CPU review. Never authorizes execution."""
from dataclasses import dataclass
from enum import Enum
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics

from . import mx5_canonical576_history as history
from . import mx5_thread_exception as old
from . import mx5_resolved_top4 as base
from .mx5_resources import inspect
from .mx5_select import check_file,record

CHECKS=('spill','wait_locations','prefetch_amount','prefetch_issue_point','partition','i1')
I1_MODE='baseline_probe_with_current_static_scope'
I1_NOT_CLAIMED='current_source_order_repeat_timing'


class Action(str,Enum):
    READY='review_ready_qualified_d7_v2'
    BLOCKED='blocked_evidence'
    NEEDS='needs_qualified_evidence'
    USE_576='use_576_solver_path'


@dataclass(frozen=True)
class Outcome:
    action:Action
    reason:str
    frozen:old.legacy.FrozenJSON
    @property
    def data(self):return self.frozen.data
    def as_dict(self):
        return dict(kind='mx5_qualified_d7_audit',schema_version=2,action=self.action.value,reason=self.reason,
                    details=self.data,execution_authorized=False,final=False,default_enabled=False)


def _need(value,reason):
    if not value:raise ValueError(reason)

def _out(action,reason,**data):return Outcome(action,reason,old.legacy.FrozenJSON.freeze(data))
def _load(ref):return history._load(ref)
def _same(a,b):return history._same(a,b)
def _sha(body):return hashlib.sha256(body).hexdigest()


def span(ref,start,end):
    lines=check_file(ref).read_bytes().splitlines(keepends=True)
    _need(type(start) is int and type(end) is int and 1<=start<=end<=len(lines),'invalid evidence line range')
    return dict(file=ref,start_line=start,end_line=end,text_sha256=_sha(b''.join(lines[start-1:end])))


def _span(value):
    _need(set(value)=={'file','start_line','end_line','text_sha256'},'unexpected report/source/MIR span fields')
    current=span(value['file'],value['start_line'],value['end_line'])
    _need(_same(current,value),'source/MIR/report text changed')
    return b''.join(check_file(value['file']).read_bytes().splitlines(keepends=True)[value['start_line']-1:value['end_line']]).decode()


def _baseline_i1(ref):
    doc=_load(ref)
    _need(set(doc)=={'kind','schema_version','provenance','source_files','rounds','run_log','guard_result','guard_journal','code_sizes','measured_binary','size_binary'},'unexpected baseline I1 evidence fields')
    _need(doc['kind']=='mx5_baseline_i1_evidence' and doc['schema_version']==1,'I1 must bind the real baseline probe, not current timing claims')
    provenance=_load(doc['provenance']);identity=provenance['implementations']['mx4_instruction_order'];digest=hashlib.sha256()
    _need(isinstance(provenance.get('base_head'),str) and provenance['base_head'],'baseline I1 source HEAD missing')
    names=[s['name'] for s in doc['source_files']]
    _need(names==identity['sources']==['experiments/mx4/stage_bench.cu'],'baseline I1 caller source identity differs')
    for item in doc['source_files']:
        digest.update(item['name'].encode()+b'\0'+check_file(item['file']).read_bytes()+b'\0')
    _need(digest.hexdigest()==identity['source_sha256'],'baseline I1 real SourceHash differs')
    for name in ('measured_binary','size_binary'):check_file(doc[name])
    sizes=_load(doc['code_sizes']);classes=('qkv','attn','o','gate_up','down');groups={};paths=set()
    _need(set(sizes)==set(classes)|{'program'},'baseline I1 code-size classes incomplete')
    for value in sizes.values():
        _need(type(value['bytes']) is int and value['bytes']>0 and isinstance(value['symbol'],str) and value['symbol'],'invalid baseline I1 code size/symbol')
    _need(len(doc['rounds'])==5,'baseline I1 requires five original fresh-process rounds')
    for index,descriptor in enumerate(doc['rounds'],1):
        path=check_file(descriptor);_need(path not in paths,'baseline I1 rounds reuse a file');paths.add(path)
        rows=list(csv.DictReader(io.StringIO(path.read_text())))
        _need(len(rows)==10,'baseline I1 needs five classes in repeat/program_order per round')
        seen=set()
        for row in rows:
            key=(row['class'],row['mode'])
            _need(key[0] in classes and key[1] in ('repeat','program_order') and key not in seen,'baseline I1 duplicate/unknown class or mode')
            _need(row['probe']=='I1' and row['model']=='qwen3_0_6b' and row['batch']=='1' and int(row['iterations'])>=1,'baseline I1 workload changed')
            median,p90=float(row['median_ns']),float(row['p90_ns'])
            _need(math.isfinite(median) and math.isfinite(p90) and 0<median<=p90,'baseline I1 contains invalid timing')
            groups.setdefault(key,[]).append(median);seen.add(key)
    log=check_file(doc['run_log']).read_text()
    _need(re.findall(r'^round=(\d+) complete$',log,re.M)==['1','2','3','4','5'],'baseline I1 accepted five-process run journal missing')
    result=_load(doc['guard_result']);journal=[json.loads(x) for x in check_file(doc['guard_journal']).read_text().splitlines() if x.strip()]
    _need(type(result.get('code')) is int and result['code']==0 and journal and journal[-1].get('phase')=='after'
          and type(journal[-1].get('exit_code')) is int and journal[-1]['exit_code']==0,'baseline I1 guard did not accept the raw samples')
    return dict(ref=ref,source_sha256=identity['source_sha256'],base_head=provenance['base_head'],rounds=5,
        median_program_order_minus_repeat_ns={c:statistics.median(groups[(c,'program_order')])-statistics.median(groups[(c,'repeat')]) for c in classes},
        historical_fresh_pid_identities_recorded=False,historical_transitive_header_identities_recorded=False,
        scope='original MX4 stage-zero I1 contract; not current native v2 timing')


def _physical(pin):
    c=old.legacy.FrozenJSON.freeze(pin).data
    c['mx_decode_v2'].pop('purpose',None)
    for name in old.CLASSES:c['mx_decode_v2'][name].pop('r_next',None)
    return c


def _node(ref,origin,index):
    node=old._d7_history(ref,origin,index)
    old.legacy._snapshot_inputs(node['snapshot'])
    resolved=_load(node['files']['resolved_pin'])['mx_decode_v2'];c=node['pin']['mx_decode_v2']
    for name in old.CLASSES:
        for field in ('tn','wn','d','part','prefetch','r_next'):
            _need(resolved[name].get(field)==c[name].get(field,-1),'D7 resolved source/pin coordinate differs')
        key=origin.candidate(index)['evidence'][name][0]['key']
        _need(all(resolved[name].get(k)==v for k,v in (('n',key['n']),('k',key['k']),('b',origin.data['preview']['request']['batch']))),'D7 resolved N/K/b changed')
        op='argmax' if name=='head' else ('kStore','kResidual','kSwiGLU')[key['op']]
        _need(resolved[name].get('op')==op and resolved[name].get('norm')==bool(key.get('norm',1)),'D7 resolved operator/norm changed')
    _need(all(resolved.get(k)==c[k] for k in ('threads','b_min','attn_prefetch')),'D7 resolved thread/attention changed')
    return node


def _family(node,receipt=None):
    options=old._normal_options(receipt or node['receipt'],node['files'],node['snapshot'],0)
    f=node['files'];s=node['snapshot']
    return old.digest(dict(source=s['source_digest'],patch=f['source_patch']['sha256'],producer=f['producer']['sha256'],compiler=f['compiler']['sha256'],options=options))


def _steps(steps,origin,index,minimum):
    _need(isinstance(steps,list) and steps,'D7 lacks actual before/after native source/MIR evidence')
    results=[]
    for step in steps:
        required={'kind','analysis','before_native','after_native','source_evidence'}
        _need(step.get('kind') in ('source_repair','budget_diagnostic') and set(step)==required|({'after_resources','after_query'} if step.get('kind')=='budget_diagnostic' else set()),'unsupported D7 step fields/kind')
        _need(isinstance(step['analysis'],str) and step['analysis'].strip(),'D7 step lacks analysis')
        before=_node(step['before_native'],origin,index);after=_node(step['after_native'],origin,index)
        _need(before['files']['binary']['sha256']!=after['files']['binary']['sha256'],'D7 before/after cannot be the same binary')
        _need(_same(_physical(before['pin']),_physical(minimum['pin'])) and _same(after['pin'],minimum['pin']),'D7 changes geometry/physical Pw/PK; current after pin must preserve canonical positive R_next')
        _need(after['snapshot']['source_digest']==_load(minimum['files']['source_snapshot'])['source_digest'],'D7 after source is not current canonical576')
        leaves=step['source_evidence'];_need(isinstance(leaves,list) and leaves,'D7 has no actual source-byte evidence')
        paths=set()
        for leaf in leaves:
            _need(set(leaf)=={'source_path','before','after'},'unexpected source evidence fields')
            path=Path(leaf['source_path']);_need(not path.is_absolute() and '..' not in path.parts and str(path) not in paths,'invalid/duplicate D7 source path');paths.add(str(path))
            for role,node in (('before',before),('after',after)):
                _need(record(check_file(leaf[role]))['sha256']==node['snapshot']['source_files'].get(str(path)),'D7 bytes differ from actual compile source')
        if step['kind']=='source_repair':
            _need(step['after_native']==minimum['files']['compile_receipt'],'D7 normal-after must be the exact bound failed576 receipt; a fitting normal576 belongs on the real576 solver path')
            _family(before)
            _need(_family(after)==minimum['family'],'D7 repaired after compiler/options family differs')
            changed={p for p in set(before['snapshot']['source_files'])|set(after['snapshot']['source_files'])
                     if p.startswith(('include/tilemega/','lib/Codegen/')) and before['snapshot']['source_files'].get(p)!=after['snapshot']['source_files'].get(p)}
            _need(changed and changed<=paths,'D7 repair must bind every changed shared/codegen source path')
        else:
            _need(step['before_native']==minimum['files']['compile_receipt'],'budget diagnostic must start at the exact bound normal576 failure')
            _need(_same(before['pin'],minimum['pin']) and before['snapshot']['source_digest']==after['snapshot']['source_digest'],'budget diagnostic changed pin/PK/R_next or source')
            _need(_family(before)==minimum['family'],'budget diagnostic before is not current normal family')
            argv=after['receipt']['argv'];pairs=[i for i,x in enumerate(argv) if re.fullmatch(r'--max-mtreg-number=\d+',x)]
            _need(len(pairs)==1 and pairs[0]>0 and argv[pairs[0]-1]=='-mllvm' and argv[pairs[0]] in ('--max-mtreg-number=227','--max-mtreg-number=255'),'only explicit227/255 nonexecuting register diagnostics are supported')
            normalized=dict(after['receipt'],argv=argv[:pairs[0]-1]+argv[pairs[0]+1:])
            _need(_family(after,normalized)==minimum['family'],'diagnostic has extra flags/compiler/source changes')
            report=inspect(check_file(after['files']['binary']),check_file(after['files']['log']),576)
            _need(_same(report,_load(step['after_resources'])),'D7 diagnostic ELF/MIR resources changed')
            _need(report['kernel']['static_smem_bytes']==0 and not report['kernel'].get('callees') and after['mir']['calls']==0,'diagnostic has an unrelated smem/callee failure')
            old._query(step['after_query'],after['files'],step['after_resources'],576,report)
        results.append(dict(kind=step['kind'],before_native=step['before_native'],after_native=step['after_native'],
            source_evidence=leaves,before_MIR=before['files']['log'],after_MIR=after['files']['log'],
            before_reference_scope='preserved diagnostic metadata; not canonical history or budget',after_is_current_canonical=True,
            query_only=step['kind']=='budget_diagnostic',execution_authorized=False))
    return results


def _checks(doc,minimum,final):
    checks=doc['checks'];_need([c.get('name') for c in checks]==list(CHECKS),'D7 ordered checks must cover spill/wait/amount/issue/partition/I1 exactly')
    known={}
    for prefix,node in (('current576',minimum),('current512',final)):
        for role,key in (('mir','log'),('pin','pin'),('generated','generated_source')):known[prefix+'_'+role]=node['files'][key]
        known[prefix+'_resources']=node['resource_ref']
    required={'spill':{'current576_mir','current576_resources'},'wait_locations':{'current576_mir','source'},
        'prefetch_amount':{'current576_pin','current576_generated'},'prefetch_issue_point':{'source','current576_mir'},
        'partition':{'current576_pin','current512_pin'},'i1':{'current512_resources','current512_mir'}}
    source=_load(minimum['files']['source_snapshot']);last=0;bound=[]
    for item in checks:
        _need(set(item)=={'name','analysis','report_span','evidence'} and isinstance(item['analysis'],str) and item['analysis'].strip(),'D7 requires substantive analysis and report references, not a confirmed boolean')
        report=item['report_span'];_need(report['file']==doc['report'] and report['start_line']>last,'D7 report checks must have ordered separate report locations')
        _need(item['name'] in _span(report),'D7 report location does not identify its check');last=report['end_line'];roles=set()
        for evidence in item['evidence']:
            _need(set(evidence)==({'role','span','source_path'} if evidence['role']=='source' else {'role','span'}),'unexpected evidence selector')
            role=evidence['role'];location=evidence['span'];_span(location)
            if role=='source':
                path=Path(evidence['source_path']);_need(str(path).startswith('include/tilemega/') and '..' not in path.parts,'D7 issue/wait source must be actual body/header')
                expected=dict(path=str((Path(source['root'])/path).resolve()),sha256=source['source_files'][str(path)])
                _need(location['file']==expected,'D7 source location is not the actual current compiled body')
            else:_need(role in known and location['file']==known[role],'D7 source/MIR/pin/resource anchor swapped')
            roles.add(role)
        _need(required[item['name']]<=roles,'D7 check lacks mandatory actual evidence roles: '+item['name']);bound.append(item)
    return bound


def current_static(final):
    report=inspect(check_file(final['files']['binary']),check_file(final['files']['log']),512)
    return dict(binary_sha256=final['files']['binary']['sha256'],source_snapshot_sha256=final['files']['source_snapshot']['sha256'],
                MIR_sha256=final['files']['log']['sha256'],program_code_bytes=report['code_bytes'],kernel_instructions=report['kernel_instructions'])


def audit_d7_v2(history_ref,lineage512,d7_ref,*,domain_tool,catalogs=()):
    try:
        bound=history.bind_final512(history_ref,lineage512,domain_tool=domain_tool,catalogs=catalogs)
        if bound.action!=history.Action.NEEDS_D7:return _out(Action.NEEDS,'qualified history/final cost evidence is incomplete',binding=bound.as_dict())
        saved=_load(history_ref);facts=saved['details'];request=facts['request'];index=request['candidate_id']
        origin=base.Qualified512Origin.capture(request['origin_preview']);minimum=facts['actual_attempts'][-1];final=bound.data['final512']
        final['resource_ref']=_load(final['ref'])['resources'];doc=_load(d7_ref)
        fields={'kind','schema_version','history','origin_sha256','candidate_id','canonical576','final512','resource_budget','spec_difference','assessment','not_claimed','report','steps','checks','i1'}
        _need(set(doc)==fields and doc['kind']=='mx5_qualified_d7_review' and doc['schema_version']==2,'D7 v2 requires a full substantive review document, not confirmed=true')
        expected576=dict(pin=minimum['pin'],binary_sha256=minimum['files']['binary']['sha256'],family_sha256=minimum['family'],p0_bundle=request['p0_bundle'])
        expected512=dict(artifact=final['ref'],pin=final['pin'],binary_sha256=final['files']['binary']['sha256'],family_sha256=final['family'])
        _need(doc['history']==history_ref and doc['origin_sha256']==origin.sha256 and type(doc['candidate_id']) is int and doc['candidate_id']==index,'D7 independent origin/history/ID changed')
        _need(_same(doc['canonical576'],expected576) and _same(doc['final512'],expected512),'D7 canonical PK/positive R_next/pins/binaries/source-family binding changed')
        _need(_same(doc['resource_budget'],bound.data['resource_budget']),'D7 shared reduction history/budget reset or changed')
        _need(facts.get('spec_difference')==old.SPEC_DIFFERENCE and doc['spec_difference']==old.SPEC_DIFFERENCE
              and minimum['resources']['registers']==168 and minimum['resources']['stack_bytes']>0,'only the explicitly authorized observed168+stack branch is supported by D7 v2')
        _need(doc['assessment']=='normal_canonical576_floor_still_spills_at_observed_168'
              and doc['not_claimed']=='required_registers_gt227_or_global_576_impossible','D7 cannot assert a physical register lower bound/global576 impossibility')
        c=minimum['pin']['mx_decode_v2'];_need(all(c[n]['prefetch']==c[n]['d'] and type(c[n]['r_next']) is int and c[n]['r_next']>0 for n in old.CLASSES),'D7 requires actual canonical-attention allD positive-reference floor')
        text=check_file(doc['report']).read_text()
        _need(all(token in text for token in (history_ref['sha256'],origin.sha256,expected576['binary_sha256'],expected512['binary_sha256'],old.SPEC_DIFFERENCE,doc['not_claimed'])),'D7 report does not bind these actual artifacts and the narrow difference')
        steps=_steps(doc['steps'],origin,index,minimum);checks=_checks(doc,minimum,final)
        i1=doc['i1'];_need(set(i1)=={'mode','not_claimed','baseline','current_static'} and i1['mode']==I1_MODE and i1['not_claimed']==I1_NOT_CLAIMED,'I1 baseline/current-static scope must be explicit')
        baseline=_baseline_i1(i1['baseline']);footprint=current_static(final)
        _need(_same(i1['current_static'],footprint),'I1 current ELF/MIR footprint does not bind this exact final512 artifact')
        return _out(Action.READY,'qualified budget/source/D7 review bound; execution/default remain unauthorized',
            history=history_ref,d7=d7_ref,origin_preview=request['origin_preview'],original_candidate=origin.candidate(index),
            resource_budget=bound.data['resource_budget'],prediction=bound.data['prediction'],minimum576=minimum,final512=final,
            steps=steps,ordered_checks=checks,i1=dict(mode=I1_MODE,baseline=baseline,current_static=footprint,not_claimed=I1_NOT_CLAIMED),
            limitations=['current_source_order_repeat_timing_not_measured','not_a_physical_gt227_lower_bound','no_E2E_correctness_or_default_promotion'],
            selection_anchor=0,validator_source=record(Path(__file__)))
    except Exception as error:return _out(Action.BLOCKED,str(error))
