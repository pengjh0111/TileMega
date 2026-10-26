"""CPU synthetic native/P0 consistency fixtures, never a real table or GPU query."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tarfile

import pytest

ROOT=Path(os.environ.get('TILEMEGA_TEST_ROOT',Path(__file__).resolve().parents[2]))
spec=importlib.util.spec_from_file_location('canonical_history_resolved_fixtures',ROOT/'test/python/test_mx5_resolved_top4.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
MODULE=Path(os.environ.get('TILEMEGA_CANONICAL576_MODULE',ROOT/'python/tilemega/mx5_canonical576_history.py'))
spec=importlib.util.spec_from_file_location('tilemega.mx5_canonical576_history',MODULE)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)


def save(path,data):return f.save(path,data)


class Fixture:
    def __init__(self,root,monkeypatch,*,two_halves=False,attention_mt=168,long_chain=False):
        root.mkdir(parents=True,exist_ok=True)
        self.root=root;self.h=f.Fixture(root,monkeypatch,threads=512);self.q=m.resolved._qualifier();self.counter=0
        # Shared synthetic archive has all callers plus the actual fixture header.
        entries={'include/tilemega/Backend/Mx/Fixture.h':self.h.h.h.header.read_bytes()}
        for kind in ('gemm','head','attention'):
            entries['experiments/mx5/'+('stage_bench.cu' if kind=='gemm' else kind+'_stage_bench.cu')]=('// CPU synthetic '+kind+' source\n').encode()
        archive=root/'all-stage-sources.tar.gz';digest=hashlib.sha256()
        with tarfile.open(archive,'w:gz') as out:
            for name,body in sorted(entries.items()):
                item=tarfile.TarInfo(name);item.size=len(body);out.addfile(item,io.BytesIO(body));digest.update(name.encode()+b'\0'+body+b'\0')
        self.sha=digest.hexdigest();self.sources=[dict(source_sha256=self.sha,archive=m.record(archive))]
        self.h.h.h.stage_sha=self.sha;self.h.h.h.sources=self.sources
        for candidate in self.h.preview['candidates']:
            for name,rows in candidate['evidence'].items():
                for row in rows:
                    row['provenance']['source_sha256']=self.sha
                    if name=='attention':row['key'].update(b=1,b_min=1,capacity=1088,prefetch_kind=candidate['pin']['mx_decode_v2']['attn_prefetch'])
        self.h.table['scope']['source_sha256']={k:self.sha for k in ('gemm','head','attention')}
        self.h.preview['stage_table']=save(root/'CPU_SYNTHETIC_table.json',self.h.table)
        save(self.h.h.preview_file,self.h.preview);self.origin_ref=m.record(self.h.h.preview_file)
        self.origin=m.resolved.Qualified512Origin.capture(self.origin_ref);self.h.h.origin=self.origin;self.h.h.h.origin=self.origin
        expected={};classes={};references={}
        for name,rows in self.origin.candidate(0)['evidence'].items():
            references[name]=[];classes[name]=[]
            for row in rows:
                key=dict(row['key'],threads=576);key['prefetch_kind' if name=='attention' else 'prefetch']=0
                index=self.q.base_id(row['kind'],key);expected[index]=16 if long_chain and name=='qkv' else 4 if two_halves and name=='qkv' else 1;classes[name].append(index)
                mt=attention_mt if name=='attention' else 150 if long_chain and name=='qkv' else 160 if two_halves and name=='qkv' else 216
                references[name].append(self.p0(row['kind'],key,archive,mt))
        monkeypatch.setattr(self.q,'scoped_domain',lambda scope,tool:(expected,classes))
        monkeypatch.setattr(m.resolved,'_qualifier',lambda:self.q)
        self.bundle=dict(kind='mx5_canonical576_p0',schema_version=2,origin_sha256=self.origin.sha256,candidate_id=0,references=references)
        self.p0_ref=save(root/'p0.json',self.bundle);self.domain=self.h.domain

    def p0(self,kind,key,archive,mt):
        folder=self.root/('p0_%d'%self.counter);self.counter+=1;folder.mkdir()
        binary=folder/'CPU_P0.so';binary.write_bytes(b'CPU synthetic P0 binary')
        source='experiments/mx5/'+('stage_bench.cu' if kind=='gemm' else kind+'_stage_bench.cu')
        meta=dict(body=kind,bench=True,returncode=0,threads=576,source_sha256=self.sha,
            binary_sha256=m.record(binary)['sha256'],base_head='CPU_SYNTHETIC')
        defs={'MX5_TEST_THREADS':576}
        if kind=='attention':
            meta.update(dim=key['dim'],query=key['q'],prefetch_kind=0);defs.update(MX5_TEST_DIM=key['dim'],MX5_TEST_Q=key['q'],MX5_BENCH_PREFETCH_KIND=0)
        else:
            meta.update(tn=key['tn'],depth=key['depth'],pw=0);defs.update(MX5_TEST_TN=key['tn'],MX5_TEST_DEPTH=key['depth'],MX5_BENCH_PW=0)
        meta['command']=[str(self.h.h.h.compiler),'-std=c++17','-O3','-shared','-Xcompiler','-fPIC,-resource-usage','-fno-strict-aliasing','-DTILEMEGA_SYNC_V3=1','-DTILEMEGA_WATCHDOG=0',*['-D%s=%d'%x for x in defs.items()],'-Iinclude',source,'-o',str(binary),'-mllvm','-print-after=stack-frame-layout']
        provenance=dict(executor='mx_static_stage_bench',implementation='mx5_'+kind+'_v2',source_sha256=self.sha,binary_sha256=meta['binary_sha256'],base_head='CPU_SYNTHETIC')
        attrs=dict(mtreg=mt,local_bytes=616,static_smem_bytes=0,resident_ctas=1)
        raw=dict(kind=kind,key=key,resources=attrs,provenance=provenance,fits=False,library=str(binary),query_only=True,query_status=0,iterations=0,card='CPU_SYNTHETIC_NOT_GPU')
        rawpath=folder/'raw.jsonl';rawpath.write_text(json.dumps(raw)+'\n')
        return dict(kind=kind,key=key,resources=attrs,provenance=provenance,fits=False,compiler_probe_only=True,
            probe=dict(query_only=True,query_status=0,iterations=0,binary=m.record(binary),build_metadata=save(folder/'build.json',meta),source_archive=m.record(archive),raw_query={**m.record(rawpath),'line':1}))

    def derive(self):return m.derive_primary576(self.origin_ref,0,self.p0_ref,domain_tool=self.domain)
    def audit(self,refs):return m.audit_history576(self.origin_ref,0,self.p0_ref,refs,domain_tool=self.domain)
    def attempt(self,pin,ordinal=0,**kwargs):
        ref=self.h.h.artifact(pin,index=0,ordinal=ordinal,**dict(dict(mt=168,frame=8),**kwargs))
        node=m._load(ref);node.update(kind='mx5_canonical576_attempt',schema_version=2)
        return save(Path(ref['path']).with_name('canonical576.json'),node)
    def history(self):
        result=self.derive();assert result.action==m.Action.PRIMARY,result.reason
        refs=[];pin=result.data['canonical_primary_pin']
        for ordinal in range(4):
            refs.append(self.attempt(pin,ordinal=ordinal));result=self.audit(refs)
            if result.action!=m.Action.NEXT_576:break
            pin=result.data['next_pin']
        return refs,result


def test_zero_count_requires_exact_failed_primary_and_keeps_canonical_nonzero_attention(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);primary=h.derive()
    assert primary.action==m.Action.PRIMARY,primary.reason
    pin=primary.data['canonical_primary_pin'];assert pin['mx_decode_v2']['attn_prefetch']==1
    assert all(pin['mx_decode_v2'][n]['prefetch']==pin['mx_decode_v2'][n]['d'] and pin['mx_decode_v2'][n]['r_next']==216 for n in m.CLASSES)
    refs,result=h.history();assert result.action==m.Action.HISTORY,result.reason
    assert result.data['used_576_reductions']==0 and result.data['remaining_512_reductions']==3 and result.data['floor_is_actual_primary']
    assert result.data['canonical_attention_kind']==1 and not result.as_dict()['execution_authorized']
    assert result.data['spec_difference']==m.old.SPEC_DIFFERENCE and result.data['not_claimed']=='required_registers_gt227_or_global_576_impossible'
    assert all(p['resources']['local_bytes']==616 for values in result.data['derivation']['p0'].values() for p in values)
    assert m.old.audit_thread_exception(h.origin,0,refs[0],[]).action==m.old.Action.BLOCKED


def test_actual_two_step_576_chain_spends_shared_budget_and_cannot_claim_zero(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch,two_halves=True);refs,result=h.history()
    assert result.action==m.Action.HISTORY,result.reason
    assert result.data['used_576_reductions']==2 and result.data['remaining_512_reductions']==1
    assert [a['shared_class'] for a in result.data['authorizations']]==['qkv','qkv']
    assert not result.data['floor_is_actual_primary']
    assert h.audit([refs[-1]]).action==m.Action.BLOCKED
    assert h.audit([refs[0],refs[-1]]).action==m.Action.BLOCKED
    save(tmp_path/'history.json',result.as_dict());saved=result.as_dict();saved['details']['used_576_reductions']=0;saved['details']['remaining_512_reductions']=3
    forged=save(tmp_path/'forged_zero.json',saved)
    assert m.bind_final512(forged,[],domain_tool=h.domain).action==m.Action.BLOCKED


@pytest.mark.parametrize('change',[
    lambda p:p['mx_decode_v2'].update(attn_prefetch=0),
    lambda p:p['mx_decode_v2']['qkv'].pop('r_next'),
    lambda p:p['mx_decode_v2']['qkv'].update(tn=32),
    lambda p:p['mx_decode_v2']['qkv'].update(prefetch=2),
])
def test_noncanonical_kind_width_unset_reference_or_shape_never_proves_zero(tmp_path,monkeypatch,change):
    h=Fixture(tmp_path,monkeypatch);pin=h.derive().data['canonical_primary_pin'];change(pin)
    result=h.audit([h.attempt(pin)])
    assert result.action==m.Action.BLOCKED


def test_manual_floor_skips_canonical_nonfloor_primary_is_rejected(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch,two_halves=True);pin=h.derive().data['canonical_primary_pin'];pin['mx_decode_v2']['qkv']['prefetch']=1
    assert h.audit([h.attempt(pin)]).action==m.Action.BLOCKED


def test_missing_attention_modern_refs_has_no_derived_pin_or_cost(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);h.bundle['references'].pop('attention');h.p0_ref=save(tmp_path/'missingA.json',h.bundle)
    result=h.derive();assert result.action==m.Action.NEEDS_P0 and result.data['canonical_primary_pin'] is None
    assert len(result.data['missing'])==5


@pytest.mark.parametrize('change',[
    lambda r:r.update(compiler_probe_only=False),
    lambda r:r['resources'].update(mtreg=1),
    lambda r:r['key'].update(threads=512),
    lambda r:r['provenance'].update(source_sha256='0'*64),
    lambda r:r['probe'].update(query_status=1),
    lambda r:r['probe'].update(raw_query=None),
])
def test_p0_source_shape_status_raw_or_reference_mutation_rejected(tmp_path,monkeypatch,change):
    h=Fixture(tmp_path,monkeypatch);change(h.bundle['references']['qkv'][0]);h.p0_ref=save(tmp_path/'badp0.json',h.bundle)
    assert h.derive().action==m.Action.BLOCKED


def test_program_query_anomaly_is_not_a_retry_authorization(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);pin=h.derive().data['canonical_primary_pin']
    assert h.audit([h.attempt(pin,resident=0)]).action==m.Action.BLOCKED
    noquery=h.audit([h.attempt(pin,query=False)])
    assert noquery.action==m.Action.NEEDS_QUERY and not noquery.as_dict()['execution_authorized']


def test_canonical_attention_two_is_not_replaced_by_legacy_floor_zero(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch,attention_mt=150);pin=h.derive().data['canonical_primary_pin']
    assert pin['mx_decode_v2']['attn_prefetch']==2
    result=h.audit([h.attempt(pin)]);assert result.action==m.Action.HISTORY,result.reason
    pin['mx_decode_v2']['attn_prefetch']=0
    assert h.audit([h.attempt(pin)]).action==m.Action.BLOCKED


def test_fitting_actual576_vetoes512_history_without_prediction_transfer(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);pin=h.derive().data['canonical_primary_pin']
    result=h.audit([h.attempt(pin,mt=167,frame=0)])
    assert result.action==m.Action.USE_576,result.reason
    assert 'effective_prediction' not in result.data and not result.as_dict()['execution_authorized']
    saved=save(tmp_path/'fits.json',result.as_dict())
    assert m.bind_final512(saved,[],domain_tool=h.domain).action==m.Action.BLOCKED


def test_failed_compiler_callee_and_non_normal_flags_stop_history(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);pin=h.derive().data['canonical_primary_pin']
    node=m._load(h.attempt(pin));receipt=m._load(node['native_receipt']);receipt.update(returncode=1,wait_status=256)
    node['native_receipt']=save(tmp_path/'compiler_failed.json',receipt)
    assert h.audit([save(tmp_path/'failed.json',node)]).action==m.Action.BLOCKED
    ref=h.attempt(pin,extra=('-mllvm','--max-mtreg-number=255'))
    assert h.audit([ref]).action==m.Action.BLOCKED
    ref=h.attempt(pin,mir_extra='  CALL target_function')
    assert h.audit([ref]).action==m.Action.BLOCKED


def test_changed_source_or_query_cannot_rehash_a_valid_history(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);pin=h.derive().data['canonical_primary_pin']
    ref=h.attempt(pin);node=m._load(ref)
    query=m._load(node['query']);query['attributes']['numRegs']+=1
    node['query']=save(tmp_path/'wrong_query.json',query)
    assert h.audit([save(tmp_path/'query_changed.json',node)]).action==m.Action.BLOCKED
    receipt=m._load(node['native_receipt']);source=m._load(receipt['files']['source_snapshot']);source['source_files']['include/tilemega/Backend/Mx/Fixture.h']='0'*64
    receipt['files']['source_snapshot']=save(tmp_path/'bad_source.json',source);receipt['source_snapshot_sha256']=receipt['files']['source_snapshot']['sha256']
    node['native_receipt']=save(tmp_path/'receipt_bad_source.json',receipt)
    assert h.audit([save(tmp_path/'source_changed.json',node)]).action==m.Action.BLOCKED


def test_noncanonical_width_requires_real_full_parent_chain_not_fourth_retry(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch,two_halves=True);refs,result=h.history()
    assert h.audit([*refs,refs[-1],refs[-1]]).action==m.Action.BLOCKED
    # An actually fitting intermediate parent cannot authorize a child.
    primary=h.derive().data['canonical_primary_pin'];fit=h.attempt(primary,mt=167,frame=0)
    assert h.audit([fit,refs[1]]).action==m.Action.BLOCKED


def test_remaining_budget_binds_independent512_costs_and_never_resets(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch,two_halves=True);_,history=h.history()
    proof=save(tmp_path/'history.json',history.as_dict())
    assert history.data['remaining_512_reductions']==1
    parent=h.h.h.artifact(mt=256,frame=8,query=False)
    next_=m.old.next_width_diagnostic(h.origin,0,[parent]);assert next_.action==m.old.Action.NEXT_WIDTH
    child=h.h.h.artifact(next_.data['next_pin'],ordinal=1)
    missing=m.bind_final512(proof,[parent,child],domain_tool=h.domain)
    assert missing.action==m.Action.NEEDS_COSTS and missing.data['prediction'] is None,missing.reason
    catalog=h.h.h.catalog(next_.data['authorization'])
    result=m.bind_final512(proof,[parent,child],domain_tool=h.domain,catalogs=[catalog])
    assert result.action==m.Action.NEEDS_D7,result.reason
    assert result.data['resource_budget']['remaining_reductions']==0
    assert result.data['prediction']['origin_predicted_tpot_ns']==h.origin.candidate(0)['predicted_tpot_ns']
    assert result.data['prediction']['effective_predicted_tpot_ns']!=result.data['prediction']['origin_predicted_tpot_ns']
    assert not result.as_dict()['execution_authorized']
    # A second actual512 reduction exceeds the cross-shape budget, even though
    # the old independent512 lineage helper can describe two local reductions.
    failed_child=h.h.h.artifact(next_.data['next_pin'],ordinal=1,mt=256,frame=8,query=False)
    next2=m.old.next_width_diagnostic(h.origin,0,[parent,failed_child]);assert next2.action==m.old.Action.NEXT_WIDTH
    child2=h.h.h.artifact(next2.data['next_pin'],ordinal=2)
    exceeded=m.bind_final512(proof,[parent,failed_child,child2],domain_tool=h.domain)
    assert exceeded.action==m.Action.BLOCKED and 'remaining' in exceeded.reason


def test_512_failure_only_uses_remaining_budget_or_stops_at_floor(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch,two_halves=True);_,history=h.history();proof=save(tmp_path/'history.json',history.as_dict())
    parent=h.h.h.artifact(mt=256,frame=8,query=False)
    result=m.bind_final512(proof,[parent],domain_tool=h.domain)
    assert result.action==m.Action.NEXT_512,result.reason
    assert result.data['resource_budget']['remaining_reductions']==1
    failed_child=h.h.h.artifact(result.data['next_pin'],ordinal=1,mt=256,frame=8,query=False)
    exhausted=m.bind_final512(proof,[parent,failed_child],domain_tool=h.domain)
    assert exhausted.action==m.Action.BLOCKED and 'exhausted' in exhausted.reason


def test_third_reduction_before_floor_stops_without_an_unrecorded_fourth(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch,long_chain=True);refs,result=h.history()
    assert len(refs)==4 and result.action==m.Action.BLOCKED
    assert 'exhausted' in result.reason


def test_retained_probe_compiler_forced_budget_and_raw_latency_rejected(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);reference=h.bundle['references']['qkv'][0]
    metadata=m._load(reference['probe']['build_metadata']);metadata['command']+=['-mllvm','--max-mtreg-number=255']
    reference['probe']['build_metadata']=save(tmp_path/'forced_build.json',metadata)
    h.p0_ref=save(tmp_path/'forced_p0.json',h.bundle)
    result=h.derive();assert result.action==m.Action.BLOCKED and 'flags' in result.reason
    h=Fixture(tmp_path/'other',monkeypatch)
    reference=h.bundle['references']['qkv'][0];raw=json.loads(Path(reference['probe']['raw_query']['path']).read_text());raw['median_ns']=0
    path=h.root/'latency.jsonl';path.write_text(json.dumps(raw)+'\n');reference['probe']['raw_query']={**m.record(path),'line':1}
    h.p0_ref=save(h.root/'latency_p0.json',h.bundle)
    assert h.derive().action==m.Action.BLOCKED


@pytest.mark.parametrize('field,value',[('static_smem_bytes',4),('resident_ctas',0)])
def test_p0_smem_or_occupancy_anomaly_rejected_even_with_consistent_raw_attrs(tmp_path,monkeypatch,field,value):
    h=Fixture(tmp_path,monkeypatch);reference=h.bundle['references']['qkv'][0]
    reference['resources'][field]=value
    raw=json.loads(Path(reference['probe']['raw_query']['path']).read_text())
    raw['resources']=copy.deepcopy(reference['resources'])
    path=tmp_path/('consistent_'+field+'.jsonl');path.write_text(json.dumps(raw)+'\n')
    reference['probe']['raw_query']={**m.record(path),'line':1}
    # General nonexecuting-reference validation preserves these real attributes;
    # canonical576 v2 adds the root P0 helper's strict smem/residency policy.
    assert h.q.table.validate_schema3_reference(reference)[3][field]==value
    h.p0_ref=save(tmp_path/'abnormal_p0.json',h.bundle)
    result=h.derive()
    assert result.action==m.Action.BLOCKED and 'P0 static shared memory or occupancy anomaly' in result.reason
