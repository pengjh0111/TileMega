"""Bind a genuine completed ordinary batch to a plain cost view, never a plan authorization."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path

from . import mx5_plan_resources as legacy
from .mx5_select import check_file,record,write

ROOT=Path(legacy.__file__).resolve().parents[2]
MODE='ordinary_completed_original_batch_v1'

def need(value,reason):
    if not value:raise ValueError(reason)

def _load(ref):return json.loads(check_file(ref).read_text())
def _rows(path):return [json.loads(s)for s in Path(path).read_text().splitlines()if s.strip()]
def _same(a,b):return legacy.FrozenJSON.freeze(a)==legacy.FrozenJSON.freeze(b)
def _q():
    p=ROOT/'experiments/mx5/qualify_campaign.py';s=importlib.util.spec_from_file_location('mx5_ordinary_trusted_qualifier',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def _tools(q):return [record(p)for p in (q.__file__,q.table.__file__,q.table.compiler_probe.__file__,q.scheduler.__file__,ROOT/'experiments/mx5/campaign.py')]

def _witnesses(job):
    d=Path(job['directory'])
    result=dict(keys=record(job['keys']),checked=record(d/'checked.jsonl'),references=record(d/'references.json'),rounds=[record(d/f'round{n}.jsonl')for n in (1,2,3)],processes=[record(d/f'process{n}.json')for n in (1,2,3)],logs=[record(d/f'round{n}.log')for n in (1,2,3)],guard=dict(result=record(d/'guard/guard_result.json'),journal=record(d/'guard/guard.jsonl')),context=record(d/'ROOT_RUN_CONTEXT.json'))
    if(d/'scheduler_job.json').exists():result['scheduler']=record(d/'scheduler_job.json')
    return result

def _admission(evidence,manifest,job,checked,q):
    ad=_load(evidence['admission']);need(ad['campaign']==evidence['campaign'] and ad['fresh_processes_per_batch']==3,'ordinary admission/campaign/fresh-process mismatch')
    need(type(ad['iterations_minimum']) is int and ad['iterations_minimum']>=3 and ad['rotation_bytes_minimum']>=256*1024**2,'ordinary timing admission protocol missing')
    need(check_file(ad['selected'])==Path(manifest['checked']).resolve(),'ordinary selected source differs from full original manifest')
    for name in ('actual_checked','actual_raw','actual_query_summary'):check_file(ad[name])
    for item in ad['tools']:check_file(item)
    snapshot=_load(ad['source_snapshot']);root=Path(snapshot['root']);files=snapshot['files']
    for name,h in files.items():
        p=Path(name);need(not p.is_absolute() and '..'not in p.parts,'invalid ordinary source path');check_file(dict(path=str(root/p),sha256=h))
    runner=root/'experiments/mx5/campaign.py';bench=root/'experiments/mx5'/('stage_bench.py'if job['kind']=='gemm'else 'head_stage_bench.py')
    need(record(runner) in ad['tools'] and record(bench) in ad['tools'],'ordinary runner/bench lacks original admission witness')
    need(record(runner)['sha256']==record(ROOT/'experiments/mx5/campaign.py')['sha256'],'unsupported ordinary runner; no three-subprocess source witness')
    need(files.get('experiments/mx5/campaign.py')==record(runner)['sha256'] and files.get(str(bench.relative_to(root)))==record(bench)['sha256'],'archived runner/bench differ from retained source snapshot')
    ctx=_load(evidence['witnesses']['context'])
    need(ctx['admission']==evidence['admission'] and ctx['manifest']==evidence['campaign'] and ctx['job']==evidence['job_index'] and ctx['iterations_minimum']==ad['iterations_minimum'] and ctx['child_PYTHONPATH']==str(root/'python'),'ordinary actual run context/admission changed')
    libraries={str(check_file(x['binary'])):x for x in ad['libraries']};visited=set()
    for row in checked:
        path=str(Path(row['library']).resolve());need(path in libraries,'ordinary batch library absent from original admission');lib=libraries[path]
        if path not in visited:
            binary=check_file(lib['binary']);meta=_load(lib['metadata']);check_file(lib['source_archive']);visited.add(path)
            need(meta.get('bench') is True and meta.get('body')==row['kind'] and meta.get('returncode')==0,'ordinary library is not an admitted bench build')
            need(meta['source_sha256']==ad['source_sha256']==row['provenance']['source_sha256'] and meta['binary_sha256']==lib['binary']['sha256']==row['provenance']['binary_sha256'] and meta['base_head']==row['provenance']['base_head'],'ordinary library/source identity mismatch')
            q.table.compiler_probe.validate_normal_command(row['kind'],q.canonical(row['kind'],row['key']),meta['command'],binary)
            legacy._verify_archive(row,[dict(source_sha256=meta['source_sha256'],archive=lib['source_archive'])],files)
    return ad,bench

def replay(evidence):
    fields={'kind','schema_version','campaign','job_index','admission','qualified_parent','request','tools','witnesses','card','card_evidence'}
    need(set(evidence)==fields and evidence['kind']=='mx5_ordinary_completed_batch_evidence' and evidence['schema_version']==1,'wrong ordinary completed-batch evidence schema')
    q=_q();need(evidence['tools']==_tools(q),'trusted ordinary qualifier/runner tools changed')
    manifest=_load(evidence['campaign']);i=evidence['job_index'];need(type(i) is int and 0<=i<len(manifest['jobs']) and manifest['processes_per_key']==3,'invalid original batch index/three-process manifest')
    job=manifest['jobs'][i];need(job['kind'] in ('gemm','head'),'ordinary variant reuse is GEMM/head only')
    witnesses=evidence['witnesses'];need(witnesses==_witnesses(job),'original full batch witnesses changed; no projected keys/process/raw files')
    need(q.scheduler.batch_state(job)=='completed','original batch is incomplete')
    d=Path(job['directory']);keys=_load(witnesses['keys']);checked=_rows(check_file(witnesses['checked']));references=_load(witnesses['references'])['references']
    need(len(keys)==len(checked)==job['rows'] and references==[r['r_next_reference']for r in checked],'original full-batch keys/checked/references differ')
    source_path=Path(manifest['checked']);need(record(source_path)['sha256']==manifest['checked_sha256'],'original campaign checked source changed')
    source={q.table.key_id(r['kind'],q.canonical(r['kind'],r['key'])):r for r in _rows(source_path)}
    for key,row in zip(keys,checked):
        kind,coordinate=q.checked_identity(row)
        need(kind==job['kind'] and coordinate==q.canonical(kind,key) and key.get('library')==row.get('library') and source.get(q.table.key_id(kind,coordinate))==row,'original checked key/source identity changed')
    ad,bench=_admission(evidence,manifest,job,checked,q)
    pids=set()
    for number,ref in enumerate(witnesses['processes'],1):
        process=_load(ref);argv=process['command']
        need(type(process.get('round')) is int and process['round']==number and type(process.get('returncode')) is int and process['returncode']==0,'ordinary process round/status is invalid')
        need(len(argv)>1 and Path(argv[1]).resolve()==bench.resolve(),'ordinary process did not use admitted archived bench')
        if 'pid'in process:
            pid=process['pid'];need(type(pid) is int and pid>0 and pid not in pids,'provided ordinary process PIDs must be positive and distinct');pids.add(pid)
        need(argv.count('--iterations')==1 and int(argv[argv.index('--iterations')+1])>=ad['iterations_minimum'],'ordinary requested iteration protocol changed')
    inputs=q.Inputs();card,card_evidence=q.card_for(job,inputs,ROOT/'runs/mx5/gpu_jobs.log')
    # The shared job log is append-only. Keep its captured hash, but compare its
    # exact matched lines/card, not a later hash of unrelated appended jobs.
    stable=lambda v:{k:x for k,x in v.items()if k!='snapshot_sha256'}
    need(card==evidence['card'] and stable(card_evidence)==stable(evidence['card_evidence']),'ordinary original guard/card provenance changed')
    guard=_load(witnesses['guard']['result']);observations=_rows(check_file(witnesses['guard']['journal']))
    need(type(guard.get('code')) is int and guard['code']==0 and observations and observations[-1].get('phase')=='after' and type(observations[-1].get('exit_code')) is int and observations[-1]['exit_code']==0,'ordinary guard lacks successful final observation')
    aggregate=q.table.aggregate([check_file(r)for r in witnesses['rounds']],[check_file(witnesses['references'])],card='C550 GPU '+card)
    need(len(aggregate['rows'])==len(checked),'ordinary full aggregate key count changed')
    parents={q.table.key_id(r['kind'],q.canonical(r['kind'],r['key'])):r for r in checked}
    for row in aggregate['rows']:
        parent=parents[q.table.key_id(row['kind'],row['key'])]
        need(row['provenance']==parent['provenance'] and row['resources']==parent['resources'],'ordinary aggregate provenance/resources differ from original checked source')
        for sample in row['rounds']:
            raw=sample['measurement'];need(raw['iterations']>=ad['iterations_minimum'] and raw.get('active_working_set_bytes',raw.get('working_set_bytes',0))>=ad['rotation_bytes_minimum'],'ordinary raw iteration/rotation protocol differs')
    request=evidence['request'];need(set(request)=={'kind','key'} and request['kind']==job['kind'] and q.canonical(request['kind'],request['key'])==request['key'],'noncanonical ordinary exact request')
    matches=[r for r in aggregate['rows']if r['kind']==request['kind'] and r['key']==request['key']];need(len(matches)==1,'ordinary batch lacks one exact changed-width key')
    row=matches[0];batch_id=dict(manifest=str(check_file(evidence['campaign'])),job=i)
    annotated=[]
    for cost in aggregate['rows']:
        parent=parents[q.table.key_id(cost['kind'],cost['key'])];value=copy.deepcopy(cost)
        value.update(uses=parent['uses'],resource_selection={n:parent[n]for n in ('selected_resource_step','resource_attempts')if n in parent},card=card,campaign_batch=batch_id);annotated.append(value)
    parent_doc=_load(evidence['qualified_parent']);pool=parent_doc.get('validation',parent_doc)
    need(pool.get('qualified') is True and pool.get('table_kind')=='mx5_decode_stage_costs','ordinary annotated parent is not qualified measurement evidence')
    parent_rows=[r for r in pool['rows']if r.get('campaign_batch')==batch_id]
    order=lambda r:legacy.digest(dict(kind=r['kind'],key=r['key']))
    need(len(parent_rows)==len(annotated) and _same(sorted(parent_rows,key=order),sorted(annotated,key=order)),'full original annotated parent differs from genuine aggregate/checked batch')
    legacy._validate_row(row,q.table)
    return row

def capture(campaign,job_index,admission,qualified_parent,kind,key):
    q=_q();m=json.loads(Path(campaign).read_text());need(type(job_index) is int and 0<=job_index<len(m['jobs']),'invalid original batch index');job=m['jobs'][job_index]
    inputs=q.Inputs();card,card_evidence=q.card_for(job,inputs,ROOT/'runs/mx5/gpu_jobs.log')
    evidence=dict(kind='mx5_ordinary_completed_batch_evidence',schema_version=1,campaign=record(campaign),job_index=job_index,admission=record(admission),qualified_parent=record(qualified_parent),request=dict(kind=kind,key=key),tools=_tools(q),witnesses=_witnesses(job),card=card,card_evidence=card_evidence)
    row=replay(evidence)
    return dict(row=row,row_sha256=legacy.digest(row),evidence_mode=MODE,ordinary_campaign=evidence)

def verify(entry,row):
    need(entry.get('evidence_mode')==MODE,'ordinary evidence mode is not explicit')
    if 'qualification_tool'in entry:need(entry['qualification_tool']==record(ROOT/'experiments/mx5/stage_table.py'),'ordinary entry qualifier differs from trusted tool')
    evidence=entry['ordinary_campaign'];actual=replay(evidence)
    need(_same(row,actual) and entry.get('row_sha256',legacy.digest(row))==legacy.digest(actual),'ordinary cost view differs from full original aggregate')
    if 'row'in entry:need(_same(entry['row'],row),'ordinary entry row differs')
    for name,expected in (('fresh_processes',evidence['witnesses']['processes']),('keys',evidence['witnesses']['keys']),('guard',evidence['witnesses']['guard'])):
        if name in entry:need(entry[name]==expected,'ordinary entry supplies different '+name)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('campaign','admission','qualified-parent','key','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--job',type=int,required=True);p.add_argument('--kind',choices=('gemm','head'),required=True);a=p.parse_args()
    view=capture(a.campaign,a.job,a.admission,a.qualified_parent,a.kind,json.loads(a.key.read_text()))
    write(a.out,dict(kind='mx5_ordinary_campaign_cost_view',schema_version=1,**view,execution_authorized=False,final=False,default_enabled=False))
if __name__=='__main__':main()
