"""Synthetic CPU fixtures only: no device query, timing or real cost table."""
import argparse
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import pytest

REPO=Path(__file__).resolve().parents[2]
MAIN=Path(os.environ.get('TILEMEGA_TEST_ROOT',REPO))
spec=importlib.util.spec_from_file_location('outcome_cpu_qualifier',REPO/'experiments/mx5/qualify_campaign.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
t=q.table


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n');return t.compiler_probe.record(path)


def lines(path,rows):path.write_text(''.join(json.dumps(x)+'\n' for x in rows))


@pytest.fixture(scope='session')
def domain(tmp_path_factory):
    compiler=shutil.which('c++')
    if not compiler:pytest.skip('host C++ compiler unavailable')
    binary=tmp_path_factory.mktemp('outcome_host_geometry')/'domain'
    subprocess.run([compiler,'-std=c++17','-O2','-I'+str(MAIN/'include'),str(REPO/'experiments/mx5/domain.cpp'),'-o',str(binary)],check=True)
    return binary


class Proofs:
    def __init__(self,root):
        self.root=root;root.mkdir();self.sequence=0
        self.compiler=root/'synthetic-compiler';self.compiler.write_text('CPU synthetic compiler, never executed\n');self.compiler.chmod(0o700)
        self.sources={};self.archives={}
        for kind in q.KINDS:
            name='experiments/mx5/'+('stage_bench.cu' if kind=='gemm' else kind+'_stage_bench.cu')
            body=('// CPU synthetic '+kind+' bench fixture\n').encode()
            path=root/(kind+'.tar.gz')
            with tarfile.open(path,'w:gz') as archive:
                member=tarfile.TarInfo(name);member.size=len(body);archive.addfile(member,io.BytesIO(body))
            self.archives[kind]=path;self.sources[kind]=hashlib.sha256(name.encode()+b'\0'+body+b'\0').hexdigest()

    def make(self,kind,key,*,reject=True,resources=None):
        directory=self.root/str(self.sequence);self.sequence+=1;directory.mkdir()
        binary=directory/'CPU_fixture.so';binary.write_bytes(('CPU synthetic binary '+str(self.sequence)).encode())
        field='prefetch_kind' if kind=='attention' else 'prefetch'
        meta=dict(body=kind,bench=True,returncode=0,threads=key['threads'],source_sha256=self.sources[kind],
                  binary_sha256=t.compiler_probe.sha(binary),base_head='CPU_FIXTURE')
        defs={'MX5_TEST_THREADS':key['threads']}
        if kind=='attention':
            meta.update(dim=key['dim'],query=key['q'],prefetch_kind=key[field]);defs.update(MX5_TEST_DIM=key['dim'],MX5_TEST_Q=key['q'],MX5_BENCH_PREFETCH_KIND=key[field])
        else:
            meta.update(tn=key['tn'],depth=key['depth'],pw=key[field]);defs.update(MX5_TEST_TN=key['tn'],MX5_TEST_DEPTH=key['depth'],MX5_BENCH_PW=key[field])
        name='experiments/mx5/'+('stage_bench.cu' if kind=='gemm' else kind+'_stage_bench.cu')
        meta['command']=[str(self.compiler),'-std=c++17','-O3','-shared','-Xcompiler','-fPIC,-resource-usage','-fno-strict-aliasing','-DTILEMEGA_SYNC_V3=1','-DTILEMEGA_WATCHDOG=0',*['-D%s=%d'%x for x in defs.items()],'-Iinclude',name,'-o',str(binary)]
        metadata=directory/'build.json';save(metadata,meta)
        prov=dict(executor='mx_static_stage_bench',implementation='mx5_'+kind+'_v2',source_sha256=self.sources[kind],
                  binary_sha256=meta['binary_sha256'],base_head='CPU_FIXTURE')
        attrs=resources or dict(mtreg=160,local_bytes=8 if reject else 0,static_smem_bytes=0,resident_ctas=1)
        raw=dict(library=str(binary),kind=kind,key=key,resources=attrs,provenance=prov,fits=not reject,query_only=True,iterations=0,query_status=0)
        raw_path=directory/'query.json';save(raw_path,raw)
        result=dict(kind=kind,key=copy.deepcopy(key),resources=attrs,provenance=prov,fits=not reject,
                    probe=dict(query_only=True,iterations=0,query_status=0,binary=t.compiler_probe.record(binary),
                               build_metadata=t.compiler_probe.record(metadata),source_archive=t.compiler_probe.record(self.archives[kind]),
                               compiler=t.compiler_probe.record(self.compiler),raw_query=t.compiler_probe.record(raw_path)))
        if reject:
            result.update(compiler_binding='retained_executable_at_qualification',original_compiler_identity_recorded=False,original_build_compiler_sha256=None,build_policy='mx5_build_bodies_default_v1',disposition='primary_resource_rejected',primary_resource_rejection=True,r_next=160,
                          r_next_reference=self.make(kind,dict(key,**{field:0}),reject=False))
        else:result['compiler_probe_only']=True
        return result


def scope(proofs):
    return dict(model_id='CPU_SYNTHETIC_MODEL',model=dict(layers=2,hidden=1664,heads=26,hkv=13,head_dim=64,
        q_per_kv=2,inter=1664,vocab=16384,capacity=1088,qk_norm=False),batch=1,threads=576,grid=104,
        past_lo=64,past_hi=1086,source_sha256=proofs.sources.copy())


def keys(request,domain):
    expected,classes=q.scoped_domain(request,domain);result=[]
    for (kind,coordinates),first in expected.items():
        key=dict(coordinates);budget=60
        key['prefetch_kind' if kind=='attention' else 'prefetch']=1 if kind=='attention' else max(key['depth'],min(first,budget//(key['tn']//4)))
        result.append((kind,key))
    return result


def campaigns(root,request,selected):
    """Exercise actual existing campaign/guard/round qualification, all synthetic."""
    root.mkdir();jobs=[];checked_all=[];inventory=[]
    for kind in q.KINDS:
        records=[key for k,key in selected if k==kind]
        if not records:continue
        directory=root/kind;directory.mkdir();checked=[];round_rows=[]
        for key in records:
            provenance=dict(executor='mx_static_stage_bench',implementation='mx5_'+kind+'_v2',source_sha256=request['source_sha256'][kind],binary_sha256='c'*64,base_head='CPU_FIXTURE')
            reference=dict(kind=kind,key=dict(key,**{'prefetch_kind' if kind=='attention' else 'prefetch':0}),
                           provenance=dict(provenance,binary_sha256='b'*64),resources=dict(mtreg=160,local_bytes=0,static_smem_bytes=0,resident_ctas=1))
            use=[dict(model=request['model_id'],stage=kind)]
            attrs=dict(mtreg=200,local_bytes=0,static_smem_bytes=0,resident_ctas=1)
            checked.append(dict(kind=kind,key=key,library='/CPU/fixture/'+kind+'.so',uses=use,query_only=True,
                query_status=0,fits=True,selected_resource_step=0,provenance=provenance,resources=attrs,r_next_reference=reference))
            base=dict(key);base.pop('prefetch_kind' if kind=='attention' else 'prefetch');inventory.append(dict(kind=kind,key=base,uses=use))
            round_rows.append(dict(key=dict(key,library=checked[-1]['library']),**provenance,**attrs,median_ns=100,p90_ns=101,
                                   iterations=200,copies=8,working_set_bytes=256*1024**2,r_next_reference=reference,
                                   test_fixture='CPU synthetic times, never a real measured table'))
        keyfile=directory/'keys.json';save(keyfile,[dict(x['key'],library=x['library']) for x in checked])
        lines(directory/'checked.jsonl',checked);save(directory/'references.json',dict(references=[x['r_next_reference'] for x in checked]))
        for number in (1,2,3):
            output=directory/('round%d.jsonl'%number);lines(output,[dict(x,round=number) for x in round_rows]);output.with_suffix('.log').write_text('CPU fixture\n')
            save(directory/('process%d.json'%number),dict(round=number,returncode=0,command=['python','CPU_fixture_bench.py','--keys',str(keyfile),'--out',str(output),'--round',str(number)]))
        guard=directory/'guard';guard.mkdir();save(guard/'guard_result.json',dict(code=0));lines(guard/'guard.jsonl',[dict(phase='after',exit_code=0)])
        save(directory/'scheduler_job.json',dict(card=7,keys_sha256=t.compiler_probe.sha(keyfile),command=['CPU_fixture_guard','7','900',str(guard)]))
        jobs.append(dict(kind=kind,keys=str(keyfile),directory=str(directory),rows=len(checked),keys_sha256=t.compiler_probe.sha(keyfile)))
        checked_all.extend(checked)
    checked_path=root/'checked.jsonl';lines(checked_path,checked_all);inventory_path=root/'inventory.jsonl';lines(inventory_path,inventory)
    manifest=root/'campaign.json';save(manifest,dict(processes_per_key=3,rows=len(checked_all),checked=str(checked_path),checked_sha256=t.compiler_probe.sha(checked_path),jobs=jobs))
    return argparse.Namespace(manifest=[manifest],inventory=inventory_path,threads=request['threads'],job_log=root/'gpu_jobs.log',check_only=False)


def test_complete_cost_and_rejection_union_uses_real_campaign_qualifier(tmp_path,domain):
    proofs=Proofs(tmp_path/'proofs');request=scope(proofs);all_keys=keys(request,domain)
    kind,key=all_keys[0];rejected=proofs.make(kind,key);path=tmp_path/'rejected.jsonl';lines(path,[rejected])
    measured=campaigns(tmp_path/'campaigns',request,all_keys[1:])
    value=q.qualify_outcomes(request,measured,[path],domain)
    assert value['schema_version']==3 and value['outcome_domain_complete'] and not value['measured_domain_complete']
    assert value['coverage']['expected_keys']==len(all_keys) and value['coverage']['rejected_keys']==1
    assert not value['execution_512_approved'] and not value['target_defaults_changed']
    assert 'median_ns' not in value['resource_rejections'][0]
    assert len(value['rows'])+1==len(all_keys)
    lines(path,[])
    with pytest.raises(ValueError,match='incomplete schema3'):q.qualify_outcomes(request,measured,[path],domain)


@pytest.mark.parametrize('field,value',[('query_only',False),('iterations',None),('iterations',1),('query_status',None),('query_status',7)])
def test_modern_primary_query_witness_is_mandatory(tmp_path,field,value):
    p=Proofs(tmp_path/'proofs');key=dict(n=32768,k=4096,b=1,tn=16,wn=1,depth=1,op=0,norm=1,part=1,threads=576,prefetch=13)
    row=p.make('gemm',key);row['probe'][field]=value
    with pytest.raises(t.TableError):t.validate_primary_rejection(row)


@pytest.mark.parametrize('change',['null_attrs','invalid_late_attrs','missing_compiler','changed_compiler','wrong_command','raw_different_key','no_raw','fake_cost','old_reference','forced_register_flags','wrong_compiler_binding','forged_original_compiler'])
def test_rejection_cannot_be_synthesized_from_status_or_console_numbers(tmp_path,change):
    p=Proofs(tmp_path/'proofs');key=dict(n=32768,k=4096,b=1,tn=16,wn=1,depth=1,op=0,norm=1,part=1,threads=576,prefetch=13)
    row=p.make('gemm',key)
    if change=='null_attrs':row['resources']['mtreg']=None
    elif change=='invalid_late_attrs':row['resources'].update(mtreg=268,resident_ctas=None)
    elif change=='missing_compiler':row['probe'].pop('compiler')
    elif change=='changed_compiler':p.compiler.write_text('changed after capture')
    elif change=='wrong_command':
        path=Path(row['probe']['build_metadata']['path']);meta=json.loads(path.read_text());meta['command'][0]='/wrong/compiler';row['probe']['build_metadata']=save(path,meta)
    elif change=='raw_different_key':
        path=Path(row['probe']['raw_query']['path']);raw=json.loads(path.read_text());raw['key']['wn']=2;row['probe']['raw_query']=save(path,raw)
    elif change=='forced_register_flags':
        path=Path(row['probe']['build_metadata']['path']);meta=json.loads(path.read_text());meta['command'].insert(1,'--maxrregcount=227');row['probe']['build_metadata']=save(path,meta)
    elif change=='wrong_compiler_binding':row['compiler_binding']='original_build_hash'
    elif change=='forged_original_compiler':row['original_build_compiler_sha256']='c'*64
    elif change=='no_raw':row['probe'].pop('raw_query')
    elif change=='fake_cost':row['median_ns']=0
    else:row['r_next_reference'].pop('compiler_probe_only')
    with pytest.raises(t.TableError):t.validate_primary_rejection(row)


def test_occupancy_zero_is_ineligibility_only_and_never_a_fake_cost(tmp_path):
    p=Proofs(tmp_path/'proofs');key=dict(n=32768,k=4096,b=1,tn=16,wn=1,depth=1,op=0,norm=1,part=1,threads=576,prefetch=13)
    row=p.make('gemm',key,resources=dict(mtreg=227,local_bytes=0,static_smem_bytes=0,resident_ctas=0))
    kind,actual,_,attrs=t.validate_primary_rejection(row)
    assert kind=='gemm' and actual==key and attrs['resident_ctas']==0
    assert all(x not in row for x in ('median_ns','p90_ns','rounds'))


def test_scope_is_regenerated_not_inferred_from_threads_or_model_label(tmp_path,domain):
    p=Proofs(tmp_path/'proofs');request=scope(p);items=keys(request,domain)
    rows=[p.make(kind,key) for kind,key in items];path=tmp_path/'rejections.jsonl';lines(path,rows)
    document=q.qualify_outcomes(request,None,[path],domain)
    assert document['coverage']['costed_keys']==0 and document['outcome_domain_complete']
    assert all(x['costed']==0 for x in document['coverage']['by_class'].values())
    wrong=copy.deepcopy(request);wrong['threads']=512
    with pytest.raises(ValueError,match='outside exact requested'):q.qualify_outcomes(wrong,None,[path],domain)
    wrong=copy.deepcopy(request);wrong['source_sha256']['gemm']='f'*64
    with pytest.raises(ValueError,match='source differs'):q.qualify_outcomes(wrong,None,[path],domain)
    lines(path,rows+[rows[0]])
    with pytest.raises(ValueError,match='duplicate or contradictory'):q.qualify_outcomes(request,None,[path],domain)


def test_wrong_primary_width_cannot_cover_a_geometry(tmp_path,domain):
    p=Proofs(tmp_path/'proofs');request=scope(p);kind,key=keys(request,domain)[0]
    assert kind=='gemm';key['prefetch']+=1
    row=p.make(kind,key);path=tmp_path/'wrong-width.jsonl';lines(path,[row])
    with pytest.raises(ValueError,match='prescribed primary width'):q.qualify_outcomes(request,None,[path],domain)


def test_original_jsonl_line_and_pure_mir_observation_are_supported(tmp_path):
    p=Proofs(tmp_path/'proofs');key=dict(n=32768,k=4096,b=1,tn=16,wn=1,depth=1,op=0,norm=1,part=1,threads=576,prefetch=13)
    row=p.make('gemm',key)
    raw=json.loads(Path(row['probe']['raw_query']['path']).read_text())
    file=tmp_path/'original-attempts.jsonl';lines(file,[dict(raw,query_status=7),raw])
    row['probe']['raw_query']=dict(t.compiler_probe.record(file),line=2)
    path=Path(row['probe']['build_metadata']['path']);meta=json.loads(path.read_text())
    assert 'compiler_sha256' not in meta
    meta['command'][1:1]=['-mllvm','-print-after=stack-frame-layout']
    row['probe']['build_metadata']=save(path,meta)
    assert t.validate_primary_rejection(row)[1]==key
    row['probe']['raw_query']['line']=1
    with pytest.raises(t.TableError,match='zero-execution'):t.validate_primary_rejection(row)


def test_normal_policy_matches_current_builder_without_invoking_it(tmp_path):
    spec=importlib.util.spec_from_file_location('outcome_default_builder',REPO/'experiments/mx5/build_bodies.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    p=Proofs(tmp_path/'proofs');key=dict(n=32768,k=4096,b=1,tn=16,wn=1,depth=1,op=0,norm=1,part=1,threads=576,prefetch=13)
    row=p.make('gemm',key);meta=json.loads(Path(row['probe']['build_metadata']['path']).read_text())
    args=argparse.Namespace(threads=576,pw=13)
    expected=builder.compiler_command(args,dict(MX5_TEST_TN=16,MX5_TEST_DEPTH=1),Path('experiments/mx5/stage_bench.cu'),Path(row['probe']['binary']['path']))
    assert meta['command'][1:]==expected[1:]


def alter_command(reference, extra, *, trailing=True):
    path=Path(reference['probe']['build_metadata']['path']);meta=json.loads(path.read_text())
    if trailing:meta['command']+=extra
    else:meta['command'][1:1]=extra
    reference['probe']['build_metadata']=save(path,meta)


@pytest.mark.parametrize('which',['primary','p0'])
@pytest.mark.parametrize('trailing',[False,True])
def test_mir_observer_position_preserves_valid_primary_and_p0(tmp_path,which,trailing):
    p=Proofs(tmp_path/'proofs');key=dict(n=32768,k=4096,b=1,tn=16,wn=1,depth=1,op=0,norm=1,part=1,threads=576,prefetch=13)
    row=p.make('gemm',key)
    alter_command(row if which=='primary' else row['r_next_reference'],['-mllvm','-print-after=stack-frame-layout'],trailing=trailing)
    assert t.validate_primary_rejection(row)[1]==key


def test_schema3_rejection_p0_override_rejected_without_changing_schema2(tmp_path,domain):
    p=Proofs(tmp_path/'proofs');request=scope(p);items=keys(request,domain)
    rejected=[p.make(kind,key) for kind,key in items]
    reference=rejected[0]['r_next_reference'];alter_command(reference,['-mllvm','--max-mtreg-number=255'])
    # Legacy schema2 does not acquire the schema3 normal-flags rule.
    assert t.validate_reference(reference)[3]['mtreg']==160
    with pytest.raises(t.TableError,match='forced-register flags'):t.validate_primary_rejection(rejected[0])
    path=tmp_path/'all-rejected.jsonl';lines(path,rejected)
    with pytest.raises(ValueError,match='forced-register flags'):q.qualify_outcomes(request,None,[path],domain)


def test_schema3_cost_modern_p0_cannot_hide_flags_by_dropping_marker(tmp_path):
    p=Proofs(tmp_path/'proofs');key=dict(n=32768,k=4096,b=1,tn=16,wn=1,depth=1,op=0,norm=1,part=1,threads=576,prefetch=0)
    reference=p.make('gemm',key,reject=False);alter_command(reference,['-mllvm','--max-mtreg-number=255'])
    assert t.validate_reference(reference)[3]['mtreg']==160
    with pytest.raises(t.TableError,match='forced-register flags'):t.validate_schema3_reference(reference)
    reference.pop('compiler_probe_only')
    with pytest.raises(t.TableError,match='contains build evidence'):t.validate_schema3_reference(reference)
    assert t.validate_reference(reference)[3]['mtreg']==160
    for field in ('probe','fits'):reference.pop(field)
    assert t.validate_schema3_reference(reference)[3]['mtreg']==160


def test_wrong_cost_width_cannot_publish_a_complete_scope(tmp_path,domain):
    p=Proofs(tmp_path/'proofs');request=scope(p);items=keys(request,domain)
    assert items[0][1]['prefetch']==6
    items[0][1]['prefetch']=1
    measured=campaigns(tmp_path/'campaigns',request,items)
    with pytest.raises(ValueError,match='selected_resource_step'):
        q.qualify_outcomes(request,measured,[],domain)


def set_campaign_selection(measured, kind, key, selection):
    manifest=json.loads(measured.manifest[0].read_text())
    paths=[Path(manifest['checked']),*(Path(job['directory'])/'checked.jsonl' for job in manifest['jobs'] if job['kind']==kind)]
    for path in paths:
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row['kind']==kind and row['key']==key:row.update(copy.deepcopy(selection))
        lines(path,rows)
    manifest['checked_sha256']=t.compiler_probe.sha(paths[0]);save(measured.manifest[0],manifest)


def selection_for(request, key, initial):
    provenance=dict(executor='mx_static_stage_bench',implementation='mx5_gemm_v2',source_sha256=request['source_sha256']['gemm'],binary_sha256='c'*64,base_head='CPU_FIXTURE')
    resources=dict(mtreg=200,local_bytes=0,static_smem_bytes=0,resident_ctas=1)
    return dict(selected_resource_step=1,resource_attempts=[
        dict(resource_step=0,prefetch=initial,key=dict(key,prefetch=initial),query_status=0,query_only=True,fits=False,
             resources=dict(resources,mtreg=228),provenance=dict(provenance,binary_sha256='f'*64)),
        dict(resource_step=1,prefetch=key['prefetch'],key=key,query_status=0,query_only=True,fits=True,
             resources=resources,provenance=provenance)])


def test_valid_measured_halving_chain_survives_schema3_publication(tmp_path,domain):
    p=Proofs(tmp_path/'proofs');request=scope(p);items=keys(request,domain)
    initial=items[0][1]['prefetch'];assert initial==6
    kind,key=items[0];key['prefetch']=3
    measured=campaigns(tmp_path/'campaigns',request,items)
    set_campaign_selection(measured,kind,key,selection_for(request,key,initial))
    doc=q.qualify_outcomes(request,measured,[],domain)
    row=next(row for row in doc['rows'] if row['kind']==kind and row['key']==key)
    assert row['key']['prefetch']==3 and row['resource_selection']['selected_resource_step']==1
    assert doc['r_next_policy_audit']==dict(modern_normal_checked=0,legacy_cost_only_unverifiable=len(items))


@pytest.mark.parametrize('change',['missing_attempt','query_error','false_failure','first_passing','wrong_source','wrong_final_binary','wrong_next_width'])
def test_invalid_halving_evidence_does_not_become_cost_qualification(tmp_path,change):
    p=Proofs(tmp_path/'proofs');request=scope(p)
    key=dict(n=3328,k=1664,b=1,tn=16,wn=1,depth=1,op=0,norm=1,part=1,threads=576,prefetch=3)
    selection=selection_for(request,key,6)
    attempts=selection['resource_attempts']
    if change=='missing_attempt':attempts.pop(0)
    elif change=='query_error':attempts[0]['query_status']=7
    elif change=='false_failure':attempts[0]['resources'].update(mtreg=200,resident_ctas=0)
    elif change=='first_passing':attempts[0]['fits']=True
    elif change=='wrong_source':attempts[0]['provenance']['source_sha256']='f'*64
    elif change=='wrong_final_binary':attempts[-1]['provenance']['binary_sha256']='f'*64
    else:attempts[-1]['key']=dict(key,prefetch=2)
    row=dict(kind='gemm',key=key,resource_selection=selection,resources=dict(mtreg=200,local_bytes=0,static_smem_bytes=0,resident_ctas=1),provenance=selection_for(request,key,6)['resource_attempts'][-1]['provenance'])
    with pytest.raises((ValueError,KeyError)):t.validate_selected_resource(row,6)
