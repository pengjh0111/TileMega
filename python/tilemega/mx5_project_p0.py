"""Project actual modern576 P0 pools into candidate-bound canonical inputs."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tarfile

from . import mx5_canonical576_history as canonical
from . import mx5_select as common

load=common.load
record=common.record
check_file=common.check_file
write=common.write
STAGES=canonical.legacy.STAGES
INLINE='calibration_inspect_inline_reference_v1'


def need(value,reason):
    if not value:raise ValueError(reason)


def _identity(reference):return canonical.legacy.digest(dict(kind=reference['kind'],key=reference['key']))


def read_pools(paths):
    entries={};inputs=[];skipped=0
    for supplied in paths:
        path=Path(supplied).resolve();descriptor=record(path);inputs.append(descriptor)
        text=path.read_text()
        if path.suffix=='.jsonl':rows=[(json.loads(line),dict(descriptor,line=number)) for number,line in enumerate(text.splitlines(),1) if line.strip()]
        else:
            document=json.loads(text);need(isinstance(document,dict) and set(document)=={'references'},'pool requires a references object or original inspect JSONL')
            rows=[(row,dict(descriptor,index=index)) for index,row in enumerate(document['references'])]
        for row,location in rows:
            inline='r_next_reference' in row;reference=copy.deepcopy(row['r_next_reference'] if inline else row)
            need(reference.get('compiler_probe_only') is True,'modern compiler_probe_only reference required')
            need(type(reference['key'].get('threads')) is int and reference['key']['threads'] in (512,576),'unknown P0 thread shape')
            if reference['key']['threads']==512:skipped+=1;continue
            need(reference['kind'] in ('gemm','head','attention'),'unknown P0 body kind')
            width='prefetch_kind' if reference['kind']=='attention' else 'prefetch'
            need(type(reference['key'].get(width)) is int and reference['key'][width]==0,'pool contains nonzero-prefetch reference')
            if 'raw_query' not in reference['probe']:
                need(inline and reference['kind']=='attention' and 'line' in location,'P0 reference lacks original actual raw fileSHA+line')
                reference['probe']['raw_query']={k:v for k,v in location.items() if k!='index'}
                reference['probe']['raw_query']['encoding']=INLINE
            identity=_identity(reference);need(identity not in entries,'ambiguous duplicate actual576 P0 coordinate')
            entries[identity]=dict(reference=reference,pool_record=location)
        check_file(descriptor)
    return entries,inputs,skipped


def _archive(path):
    descriptor=record(path);digest=hashlib.sha256();count=0;shared=0
    with tarfile.open(check_file(descriptor),'r:gz') as archive:
        members=[m for m in archive.getmembers() if m.isfile()];names=[m.name for m in members]
        need(len(names)==len(set(names)),'duplicate archived source member')
        for member in sorted(members,key=lambda x:x.name):
            need(not Path(member.name).is_absolute() and '..' not in Path(member.name).parts,'unsafe archived source name')
            data=archive.extractfile(member).read();digest.update(member.name.encode()+b'\0'+data+b'\0');count+=1
            shared+=member.name.startswith('include/tilemega/')
    need(shared>0,'stage archive has no shared body source');check_file(descriptor)
    return dict(source_sha256=digest.hexdigest(),archive=descriptor),dict(files=count,shared_headers=shared)


def archive_inventory(references,archives=()):
    required={ref['provenance']['source_sha256'] for ref in references};choices={};stats={}
    paths=list(archives) if archives else sorted({str(check_file(ref['probe']['source_archive'])) for ref in references})
    for path in paths:
        item,summary=_archive(path);key=item['source_sha256']
        if key not in required:continue
        if key not in choices or item['archive']['path']<choices[key]['archive']['path']:choices[key]=item;stats[key]=summary
    need(set(choices)==required,'sealed stage source archives missing for actual P0 source epoch')
    return [choices[key] for key in sorted(choices)],stats


def _validate(reference,table):
    return canonical._modern_p0(reference,reference['key'],reference['kind'],reference['provenance']['source_sha256'],table)


def _tools(table):
    return dict(projector=record(Path(__file__)),canonical_validator=record(Path(canonical.__file__)),
                stage_table=record(Path(table.__file__)),compiler_probe=record(Path(table.compiler_probe.__file__)))


def audit_pools(pools,archives,out):
    out=Path(out).resolve();need(not out.exists(),'existing pool audit is preserved')
    entries,inputs,skipped=read_pools(pools);need(entries,'no actual576 P0 references')
    table=canonical.resolved._qualifier().table;tools=_tools(table);counts={};sources={};nonfit={};maxima={};compilers={}
    for entry in entries.values():
        ref=entry['reference'];proof=_validate(ref,table);compilers[proof['compiler']['sha256']]=proof['compiler'];kind=ref['kind'];source=ref['provenance']['source_sha256']
        need(kind not in sources or sources[kind]==source,'pool mixes body source epochs');sources[kind]=source
        counts[kind]=counts.get(kind,0)+1;nonfit[kind]=nonfit.get(kind,0)+int(not ref['fits'])
        old=maxima.setdefault(kind,dict(mtreg=0,local_bytes=0))
        for key in old:old[key]=max(old[key],proof['resources'][key])
    inventory,archive_stats=archive_inventory([x['reference'] for x in entries.values()],archives)
    for ref in [*inputs,*tools.values()]:check_file(ref)
    out.mkdir(parents=True);write(out/'stage_sources.json',inventory)
    result=dict(kind='mx5_modern576_p0_pool_audit',schema_version=1,pools=inputs,validated576=len(entries),skipped512=skipped,by_kind=counts,
        nonfitting_compiler_probes=nonfit,observed_maxima=maxima,source_sha256=sources,stage_sources=record(out/'stage_sources.json'),archive_members=archive_stats,
        tools=tools,retained_compilers=list(compilers.values()),compiler_binding='retained_executable_at_validation',original_compiler_identity_recorded=False,original_build_compiler_sha256=None,
        candidate_origin=None,candidate_bundle_created=False,cost_rows_created=0,execution_authorized=False,default_enabled=False)
    write(out/'audit.json',result);return result


def project(origin_path,candidate_id,pools,archives,domain_tool,out):
    out=Path(out).resolve();need(not out.exists(),'existing projection is preserved')
    origin_ref=record(origin_path);origin=canonical.resolved.Qualified512Origin.capture(origin_ref);candidate=origin.candidate(candidate_id)
    preview=origin.data['preview'];domain=canonical.resolved._complete_domain(preview,load(check_file(preview['stage_table'])),domain_tool)
    entries,inputs,skipped=read_pools(pools);references={};selected=[];bindings=[];missing=[]
    table=canonical.resolved._qualifier().table;tools=_tools(table)
    for name in STAGES:
        references[name]=[]
        for row in candidate['evidence'][name]:
            key=dict(row['key'],threads=576);key['prefetch_kind' if name=='attention' else 'prefetch']=0
            identity=canonical.legacy.digest(dict(kind=row['kind'],key=key));entry=entries.get(identity)
            if entry is None:missing.append(dict(stage=name,kind=row['kind'],key=key));continue
            ref=entry['reference']
            canonical._modern_p0(ref,key,row['kind'],row['provenance']['source_sha256'],table)
            references[name].append(ref);selected.append(ref);bindings.append(dict(stage=name,key=key,pool_record=entry['pool_record'],raw_query=ref['probe']['raw_query']))
    need(selected or missing,'empty original candidate evidence')
    # Missing coordinates are requests, never borrowed B/L/thread/source rows.
    bundle=dict(kind='mx5_canonical576_p0',schema_version=2,origin_sha256=origin.sha256,candidate_id=candidate_id,references=references)
    expected_sources=set(domain['scope']['source_sha256'].values())
    inventory_refs=[x['reference'] for x in entries.values() if x['reference']['provenance']['source_sha256'] in expected_sources]
    inventory,archive_stats=archive_inventory(inventory_refs,archives);need({x['source_sha256'] for x in inventory}==expected_sources,'origin stage archives are incomplete')
    for ref in [*inputs,*tools.values()]:check_file(ref)
    origin.verify();out.mkdir(parents=True);write(out/'p0_bundle.json',bundle);write(out/'stage_sources.json',inventory)
    derived=canonical.derive_primary576(origin_ref,candidate_id,record(out/'p0_bundle.json'),domain_tool=domain_tool)
    result=dict(kind='mx5_candidate576_p0_projection',schema_version=1,action=derived.action.value,reason=derived.reason,origin_preview=origin_ref,origin_sha256=origin.sha256,
        candidate_id=candidate_id,original_candidate=candidate,selection_anchor=0,pools=inputs,tools=tools,skipped512=skipped,bindings=bindings,missing=missing,
        p0_bundle=record(out/'p0_bundle.json'),stage_sources=record(out/'stage_sources.json'),archive_members=archive_stats,derivation=derived.as_dict(),
        execution_authorized=False,final=False,default_enabled=False)
    write(out/'projection.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    for mode in ('audit-pools','project'):
        p=sub.add_parser(mode);p.add_argument('--pool',type=Path,action='append',required=True);p.add_argument('--archive',type=Path,action='append',default=[]);p.add_argument('--out',type=Path,required=True)
        if mode=='project':
            p.add_argument('--origin',type=Path,required=True);p.add_argument('--candidate',type=int,choices=range(4),required=True);p.add_argument('--domain-tool',type=Path,required=True)
    args=parser.parse_args()
    try:
        result=audit_pools(args.pool,args.archive,args.out) if args.mode=='audit-pools' else project(args.origin,args.candidate,args.pool,args.archive,args.domain_tool,args.out)
    except (ValueError,KeyError,TypeError,OSError,tarfile.TarError) as error:result=dict(action='blocked_evidence',reason=str(error),execution_authorized=False)
    print(json.dumps(result));return 2 if result.get('action')=='blocked_evidence' else 0


if __name__=='__main__':raise SystemExit(main())
