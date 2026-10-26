"""Frozen multi-scope MX5 catalog; scoped schema3 documents remain unchanged."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

KIND='mx5_decode_stage_cost_catalog'
MODEL=('layers','hidden','heads','hkv','head_dim','q_per_kv','inter','vocab','capacity')
REQUEST=('batch','threads','grid','past_lo','past_hi')
SOURCES=('gemm','head','attention')
POLICY=('execution_512_approved','global_dt576_policy_approved','target_defaults_changed')


def need(value,reason):
    if not value:raise ValueError('MX5 catalog: '+reason)


def load(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def record(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def write(path,value):
    with Path(path).open('x') as out:json.dump(value,out,indent=2,allow_nan=False);out.write('\n')


def check_file(ref):
    need(set(ref)=={'path','sha256'} and isinstance(ref['path'],str),'invalid file reference')
    path=Path(ref['path']).resolve();need(sha(path)==ref['sha256'],'file hash changed: '+str(path));return path


def _scope(scope):
    need(set(scope)=={'model_id','model',*REQUEST,'source_sha256'},'scope fields differ')
    model=scope['model'];need(set(model)=={*MODEL,'qk_norm'} and type(model['qk_norm']) is bool,'model descriptor fields differ')
    need(all(type(model[k]) is int and model[k]>0 for k in MODEL),'model descriptor requires positive integer coordinates')
    need(isinstance(scope['model_id'],str) and scope['model_id'],'scope model label missing')
    need(all(type(scope[k]) is int for k in REQUEST) and scope['batch'] in (1,2,4,8,16) and scope['threads'] in (512,576)
         and (scope['grid'],scope['past_lo'],scope['past_hi'],model['capacity'])==(104,64,1086,1088),'unsupported scope request')
    sources=scope['source_sha256'];need(set(sources)==set(SOURCES),'source identity fields differ')
    need(all(isinstance(v,str) and len(v)==64 and not set(v)-set('0123456789abcdef') for v in sources.values()),'invalid source hash')
    return dict(model=model,**{k:scope[k] for k in REQUEST})


def _key(scope):return json.dumps(_scope(scope),sort_keys=True,separators=(',',':'))


def _validate_leaf(path,table,domain_tool):
    from .mx5_resolved_top4 import _complete_domain
    return _complete_domain(dict(stage_table=record(path)),table,domain_tool)


def _leaf_header(table,scope,coverage):
    need(table.get('schema_version')==3 and table.get('table_kind')=='mx5_decode_stage_costs' and table.get('qualified') is True
         and table.get('qualification_scope')=='complete_outcomes_single_request_v1' and table.get('outcome_domain_complete') is True
         and table.get('full_table') is False and table.get('diagnostic_reasons')==[] and not table.get('synthetic_fixture'),'leaf is not qualified complete schema3')
    need(all(table.get(k) is False for k in POLICY),'leaf grants unreviewed policy/default approval')
    need(table['scope']==scope and table['coverage']==coverage,'leaf scope/coverage differs from catalog')
    need(table['measured_domain_complete'] is (not table['resource_rejections']),'leaf measured completeness differs from actual rejections')


def load_catalog(path):
    path=Path(path).resolve();catalog_ref=record(path);doc=load(path)
    fields={'schema_version','table_kind','qualified','scope_count','source_sha256','domain_tool','tables','provenance',*POLICY}
    need(set(doc)==fields and doc['schema_version']==1 and doc['table_kind']==KIND and doc['qualified'] is True,'wrong catalog schema')
    need(all(doc[k] is False for k in POLICY),'catalog cannot authorize execution/defaults')
    need(type(doc['scope_count']) is int and doc['scope_count']==50 and len(doc['tables'])==50,'complete published catalog requires50 scopes')
    domain=check_file(doc['domain_tool']);index=load(check_file(doc['provenance']['index']));check_file(doc['provenance']['packer'])
    need(index.get('qualified_scoped_tables')==50 and len(index['tables'])==50,'recorded export index is not complete50')
    indexed={(item['model_id'],item['batch'],item['threads']):item for item in index['tables']}
    need(len(indexed)==50,'duplicate export-index scope')
    seen=set();models={};tables=[];outcomes={}
    for item in doc['tables']:
        need(set(item)=={'scope','table','coverage'},'catalog member fields differ')
        scope=item['scope'];key=_key(scope);need(key not in seen,'ambiguous/duplicate actual model/request scope');seen.add(key)
        original=indexed.get((scope['model_id'],scope['batch'],scope['threads']))
        need(original is not None and original['table']['sha256']==item['table']['sha256'] and original['coverage']==item['coverage']
             and load(check_file(original['scope']))==scope,'catalog member differs from recorded export index')
        need(scope['source_sha256']==doc['source_sha256'],'catalog has mixed source epochs')
        model=scope['model_id'];shape=json.dumps(scope['model'],sort_keys=True)
        need(model not in models or models[model][0]==shape,'one model label changes geometry across requests')
        models.setdefault(model,(shape,set()))[1].add((scope['batch'],scope['threads']))
        ref=item['table'];need(set(ref)=={'path','sha256'},'invalid member reference')
        relative=Path(ref['path']);need(not relative.is_absolute() and '..' not in relative.parts,'catalog leaves require stable relative paths')
        leaf=(path.parent/relative).resolve();need(leaf.is_relative_to(path.parent),'catalog leaf escapes stable asset directory')
        need(sha(leaf)==ref['sha256'],'catalog leaf hash changed');table=load(leaf)
        _leaf_header(table,scope,item['coverage']);_validate_leaf(leaf,table,domain)
        for rejected,group in ((False,table['rows']),(True,table['resource_rejections'])):
            for row in group:
                base=dict(row['key']);base.pop('prefetch_kind' if row['kind']=='attention' else 'prefetch')
                slot=(row['kind'],json.dumps(base,sort_keys=True,separators=(',',':')))
                value=(rejected,json.dumps(row,sort_keys=True,separators=(',',':')))
                need(outcomes.setdefault(slot,value)==value,'conflicting duplicate physical cost/rejection across scopes')
        tables.append((item,leaf,table))
    expected={(b,t) for b in (1,2,4,8,16) for t in (512,576)}
    need(all(requests==expected for _,requests in models.values()),'model scopes omit a declared batch/thread request')
    check_file(catalog_ref)
    for item,leaf,_ in tables:need(sha(leaf)==item['table']['sha256'],'catalog asset changed during validation')
    return doc,tables


def select_catalog(path,request):
    need(set(request)=={'model',*REQUEST},'actual native request fields differ')
    need(all(type(request[k]) is int for k in REQUEST) and set(request['model'])=={*MODEL,'qk_norm'}
         and all(type(request['model'][k]) is int and request['model'][k]>0 for k in MODEL) and type(request['model']['qk_norm']) is bool,'actual native request coordinate types differ')
    catalog_ref=record(path);doc,tables=load_catalog(path);check_file(catalog_ref);matches=[(item,leaf,table) for item,leaf,table in tables if _scope(item['scope'])==request]
    need(len(matches)==1,'no unique exact actual model/B/threads/grid/range scope; no nearest-scope or LS substitution')
    item,leaf,table=matches[0]
    fit_rows={(row['kind'],json.dumps(row['key'],sort_keys=True)) for _,_,leaf_table in tables for row in leaf_table['rows'] if row['kind']!='attention'}
    return dict(kind='mx5_decode_catalog_binding',schema_version=1,catalog=catalog_ref,scope=item['scope'],table=dict(path=str(leaf),sha256=item['table']['sha256']),
                tables=[dict(path=str(p),sha256=entry['table']['sha256']) for entry,p,_ in tables],fit_cost_rows=len(fit_rows),domain_tool=doc['domain_tool'],actual_request=request)


def validate_preview_binding(preview):
    fields=('stage_catalog','stage_catalog_scope');present=[k in preview for k in fields]
    if not any(present):return
    need(all(present),'preview lost catalog or scope binding')
    scope=preview['stage_catalog_scope'];request=_scope(scope)
    need(all(preview['request'][k]==request[k] for k in REQUEST),'preview request differs from catalog scope')
    binding=select_catalog(check_file(preview['stage_catalog']),request)
    need(binding['catalog']==preview['stage_catalog'] and binding['scope']==scope and binding['table']==preview['stage_table'],'preview root catalog selected another actual leaf/scope')
    need(preview['domain_report']['scope']==scope,'native domain report differs from selected catalog scope')
    fit=preview['fit']
    need(set(fit)=={'r2','rows','scope','catalog','source_sha256','attention_included','rejections_included'}
         and type(fit['r2']) in (int,float) and math.isfinite(fit['r2']) and type(fit['rows']) is int and fit['rows']==binding['fit_cost_rows']
         and fit['scope']=='catalog_unique_gemm_head_same_source_epoch' and fit['catalog']==preview['stage_catalog']
         and fit['source_sha256']==scope['source_sha256'] and fit['attention_included'] is False and fit['rejections_included'] is False,
         'global diagnostic fit corpus/source/count binding differs')


def pack(index_path,out,domain_tool):
    index_path=Path(index_path).resolve();out=Path(out).resolve();domain_tool=Path(domain_tool).resolve()
    need(out.name=='xcore1000_mx5_decode.json','final catalog filename must be xcore1000_mx5_decode.json')
    index_ref=record(index_path);packer_ref=record(Path(__file__));index=load(index_path);need(index.get('qualified_scoped_tables')==50 and len(index['tables'])==50,'export index is not complete50 scopes')
    need(not out.exists(),'existing catalog is preserved');assets=out.with_suffix('.scopes');need(not assets.exists(),'existing catalog assets are preserved')
    entries=[];sources=None
    # Complete validation precedes publication; pending values never become costs.
    for item in index['tables']:
        scope=load(check_file(item['scope']));_scope(scope);leaf=check_file(item['table']);table=load(leaf)
        need((item['model_id'],item['batch'],item['threads'])==(scope['model_id'],scope['batch'],scope['threads']),'export index label/request differs')
        _leaf_header(table,scope,item['coverage']);_validate_leaf(leaf,table,domain_tool)
        need(sources is None or sources==scope['source_sha256'],'export has mixed source epochs');sources=scope['source_sha256']
        raw=leaf.read_bytes();need(hashlib.sha256(raw).hexdigest()==item['table']['sha256'],'export leaf changed during qualification')
        need(record(check_file(item['scope']))['sha256']==item['scope']['sha256'],'export scope changed during qualification')
        entries.append((scope,item['table']['sha256'],raw,item['coverage']))
    entries.sort(key=lambda item:(_key(item[0]),item[0]['model_id']))
    out.parent.mkdir(parents=True,exist_ok=True);assets.mkdir();created=[]
    try:
        members=[]
        for scope,leaf_sha,raw,coverage in entries:
            destination=assets/(leaf_sha+'.json')
            if not destination.exists():
                with destination.open('xb') as output:created.append(destination);output.write(raw)
            need(sha(destination)==leaf_sha,'copied leaf bytes differ from frozen export')
            members.append(dict(scope=scope,table=dict(path=str(destination.relative_to(out.parent)),sha256=leaf_sha),coverage=coverage))
        doc=dict(schema_version=1,table_kind=KIND,qualified=True,scope_count=50,source_sha256=sources,domain_tool=record(domain_tool),tables=members,
                 provenance=dict(index=index_ref,packer=packer_ref),**{k:False for k in POLICY})
        check_file(index_ref);check_file(packer_ref)
        with out.open('x') as output:
            created.append(out);json.dump(doc,output,indent=2,allow_nan=False);output.write('\n')
        load_catalog(out)
        return record(out)
    except BaseException:
        for path in reversed(created):path.unlink()
        assets.rmdir()
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('pack');p.add_argument('--index',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--domain-tool',type=Path,required=True)
    p=sub.add_parser('resolve');p.add_argument('--catalog',type=Path,required=True);p.add_argument('--request',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='pack':print(json.dumps(pack(args.index,args.out,args.domain_tool)))
    else:write(args.out,select_catalog(args.catalog,load(args.request)))


if __name__=='__main__':main()
