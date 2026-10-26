"""CPU synthetic catalog/consumer fixtures; no real costs, GPU or approval."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

ROOT=Path(os.environ.get('TILEMEGA_TEST_ROOT',Path(__file__).resolve().parents[2]))
TREE=Path(os.environ.get('TILEMEGA_CATALOG_TREE',ROOT))


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value


@pytest.fixture
def api():return module('tilemega.mx5_catalog',TREE/'python/tilemega/mx5_catalog.py')


def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n');return path


class Harness:
    def __init__(self,root,api,monkeypatch):
        self.root=root;self.api=api;self.deep=[]
        # Domain and every leaf are explicitly CPU placeholders. Only this deep
        # qualification boundary is injected; catalog/hash/scope logic is real.
        self.domain=root/'CPU_SYNTHETIC_DOMAIN';self.domain.write_bytes(b'CPU synthetic trusted-tool fixture, never executed')
        def validate(path,table,domain):
            assert table['provenance']['fixture']=='CPU synthetic empty leaf, NOT stage measurements'
            assert Path(domain)==self.domain
            self.deep.append(api.record(path))
        monkeypatch.setattr(api,'_validate_leaf',validate)
        self.items=[]
        for label in range(5):
            for batch in (1,2,4,8,16):
                for threads in (512,576):
                    scope=dict(model_id='CPU_SYNTHETIC_%d'%label,model=dict(layers=label+2,hidden=1664,heads=26,hkv=13,head_dim=64,q_per_kv=2,inter=1664,vocab=16384,capacity=1088,qk_norm=False),
                        batch=batch,threads=threads,grid=104,past_lo=64,past_hi=1086,source_sha256={kind:'1'*64 for kind in api.SOURCES})
                    coverage=dict(expected_keys=0,costed_keys=0,rejected_keys=0,missing_keys=0)
                    table=dict(schema_version=3,table_kind='mx5_decode_stage_costs',qualified=True,qualification_scope='complete_outcomes_single_request_v1',scope=scope,coverage=coverage,
                        outcome_domain_complete=True,measured_domain_complete=True,full_table=False,diagnostic_reasons=[],rows=[],resource_rejections=[],
                        r_next_policy_audit=dict(modern_normal_checked=0,legacy_cost_only_unverifiable=0),
                        provenance=dict(fixture='CPU synthetic empty leaf, NOT stage measurements'),**{k:False for k in api.POLICY})
                    folder=root/('export/%s_B%d_t%d'%(scope['model_id'],batch,threads));s=save(folder/'scope.json',scope);p=save(folder/'stage_costs_schema3.json',table)
                    self.items.append(dict(model_id=scope['model_id'],batch=batch,threads=threads,scope=api.record(s),table=api.record(p),coverage=coverage))
        self.index=save(root/'TABLE_INDEX_50.json',dict(tables=self.items,qualified_scoped_tables=50,test_fixture='CPU synthetic export; not qualified by real domain validator'))
        self.out=root/'published/xcore1000_mx5_decode.json'

    def pack(self):return self.api.pack(self.index,self.out,self.domain)
    def request(self,index=0):return self.api._scope(self.api.load(self.items[index]['scope']['path']))
    def preview(self):
        binding=self.api.select_catalog(self.out,self.request());table=self.api.load(binding['table']['path']);scope=binding['scope']
        producer=self.root/'CPU_PRODUCER';producer.write_text('CPU synthetic producer, not executed')
        report={k:copy.deepcopy(table[k]) for k in ('scope','coverage','resource_rejections','r_next_policy_audit')};report.update(execution_512_approved=False,final=False,default_enabled=False,ranking_claim='no_top4_claim')
        return dict(schema_version=1,kind='mx5_decode_candidate_preview',diagnostic=True,final=False,solver_selected=False,default_enabled=False,
            stage_table=binding['table'],stage_catalog=binding['catalog'],stage_catalog_scope=scope,input=self.api.record(producer),producer=self.api.record(producer),
            request=dict(**{k:scope[k] for k in self.api.REQUEST},capacity=1088,runtime_target=str(self.root/'target.json')),
            preview_mode='schema3_qualified_domain_diagnostic',source_schema_version=3,schema3_diagnostic_only=True,execution_authorized=False,results_consumption_allowed=False,
            fit=dict(r2=0,rows=0,scope='catalog_unique_gemm_head_same_source_epoch',catalog=binding['catalog'],source_sha256=copy.deepcopy(scope['source_sha256']),attention_included=False,rejections_included=False),
            domain_report=report,status='no_executable_plan',available_candidate_count=0,candidates=[])


def test_pack_preserves_all_leaf_bytes_and_resolves_by_actual_geometry_not_filename(tmp_path,monkeypatch,api):
    h=Harness(tmp_path,api,monkeypatch);descriptor=h.pack();doc,tables=api.load_catalog(h.out)
    assert descriptor==api.record(h.out) and len(tables)==50
    originals={item['table']['sha256']:Path(item['table']['path']).read_bytes() for item in h.items}
    for item,path,_ in tables:
        assert not Path(item['table']['path']).is_absolute() and path.read_bytes()==originals[item['table']['sha256']]
    binding=api.select_catalog(h.out,h.request(17));assert binding['scope']==api.load(h.items[17]['scope']['path'])
    assert binding['table']['sha256']==h.items[17]['table']['sha256'] and len(binding['tables'])==50
    assert len(h.deep)>=150
    before=h.out.read_bytes()
    with pytest.raises(ValueError,match='preserved'):h.pack()
    assert h.out.read_bytes()==before


@pytest.mark.parametrize('change',['missing','duplicate','source','coverage','cost_placeholder','path_escape','absolute_asset'])
def test_catalog_missing_duplicate_epoch_header_or_asset_tamper_rejected(tmp_path,monkeypatch,api,change):
    h=Harness(tmp_path,api,monkeypatch);h.pack();doc=api.load(h.out)
    if change=='missing':doc['tables'].pop()
    elif change=='duplicate':doc['tables'][1]=copy.deepcopy(doc['tables'][0])
    elif change=='source':doc['tables'][0]['scope']['source_sha256']['head']='2'*64
    elif change=='coverage':doc['tables'][0]['coverage']['missing_keys']=1
    elif change=='path_escape':doc['tables'][0]['table']['path']='../export/not_a_leaf.json'
    elif change=='absolute_asset':doc['tables'][0]['table']['path']=str((h.out.parent/doc['tables'][0]['table']['path']).resolve())
    else:
        leaf=h.out.parent/doc['tables'][0]['table']['path'];table=api.load(leaf);table['qualified']=False;save(leaf,table);doc['tables'][0]['table']['sha256']=api.sha(leaf)
    save(h.out,doc)
    with pytest.raises(ValueError):api.load_catalog(h.out)


@pytest.mark.parametrize('change',['model','batch','threads','range'])
def test_no_exact_scope_never_chooses_nearest_or_fallback(tmp_path,monkeypatch,api,change):
    h=Harness(tmp_path,api,monkeypatch);h.pack();request=h.request()
    if change=='model':request['model']['layers']=99
    elif change=='batch':request['batch']=32
    elif change=='threads':request['threads']=640
    else:request['past_hi']=1085
    with pytest.raises(ValueError,match='unique exact'):api.select_catalog(h.out,request)


def test_failed_pack_rolls_back_only_its_new_files_and_cannot_rename_final_entry(tmp_path,monkeypatch,api):
    h=Harness(tmp_path,api,monkeypatch)
    with pytest.raises(ValueError,match='filename'):api.pack(h.index,tmp_path/'wrong.json',h.domain)
    index=api.load(h.index);index['tables'][1]=copy.deepcopy(index['tables'][0]);save(h.index,index)
    with pytest.raises(ValueError,match='duplicate'):h.pack()
    assert not h.out.exists() and not h.out.with_suffix('.scopes').exists()
    assert all(Path(x['table']['path']).is_file() for x in h.items)


def test_real_qualification_boundary_rejects_the_explicit_empty_synthetic_leaves(tmp_path,monkeypatch,api):
    real=api._validate_leaf;h=Harness(tmp_path,api,monkeypatch);monkeypatch.setattr(api,'_validate_leaf',real)
    with pytest.raises((ValueError,KeyError)):h.pack()
    assert not h.out.exists()


def test_shared_preview_validation_keeps_direct_schema3_and_catalog_root_leaf_binding(tmp_path,monkeypatch,api):
    h=Harness(tmp_path,api,monkeypatch);h.pack();preview=h.preview()
    select=module('tilemega._catalog_select_cpu',TREE/'python/tilemega/mx5_select.py')
    path=save(tmp_path/'preview.json',preview)
    assert select.validate_preview(path,allow_schema3_diagnostic=True)==preview
    with pytest.raises(ValueError,match='explicit allow_schema3'):select.validate_preview(path)
    direct=copy.deepcopy(preview);direct.pop('stage_catalog');direct.pop('stage_catalog_scope');save(path,direct)
    assert select.validate_preview(path,allow_schema3_diagnostic=True)==direct
    for change in ('root','scope','leaf','missing_binding','fit_count','fit_source'):
        changed=copy.deepcopy(preview)
        if change=='root':changed['stage_catalog']['sha256']='0'*64
        elif change=='scope':changed['stage_catalog_scope']['model']['layers']=88
        elif change=='leaf':changed['stage_table']=h.items[1]['table']
        elif change=='missing_binding':changed.pop('stage_catalog_scope')
        elif change=='fit_count':changed['fit']['rows']=1
        else:changed['fit']['source_sha256']['gemm']='f'*64
        save(path,changed)
        with pytest.raises(ValueError):select.validate_preview(path,allow_schema3_diagnostic=True)


def test_fresh_replay_retains_catalog_root_and_all_original_candidate_values(tmp_path,monkeypatch,api):
    h=Harness(tmp_path,api,monkeypatch);h.pack();preview=h.preview();preview['candidates']=[dict(index=i,pin={'CPU_SYNTHETIC':i},ranks=[i,0,0,0,0,0],predicted_tpot_ns=i+1,predicted_stage_ns={'CPU_SYNTHETIC':1},evidence={'CPU_SYNTHETIC':i}) for i in range(4)]
    select=module('tilemega._catalog_select_replay_cpu',TREE/'python/tilemega/mx5_select.py')
    runner=module('tilemega._catalog_runner_cpu',TREE/'python/tilemega/mx5_resolved_runner.py')
    monkeypatch.setattr(runner,'common',select)
    # Native/qualification boundary injection only; replay command and original
    # four-candidate comparisons below are the production consumer functions.
    monkeypatch.setattr(select,'validate_preview',lambda path,**kw:api.load(path))
    prefix=tmp_path/'fresh';path=save(Path(str(prefix)+'.mx5_candidates.json'),preview);process=dict(returncode=0,argv=select.native_solve_command(preview,prefix))
    assert process['argv'][process['argv'].index('--mx-decode-v2-solve')+1]==str(h.out)
    assert runner._fresh(preview,api.record(path),process)==preview
    for field in ('stage_catalog','stage_catalog_scope','stage_table'):
        changed=copy.deepcopy(preview);changed.pop(field);save(path,changed)
        with pytest.raises(ValueError,match='differs'):runner._fresh(preview,api.record(path),process)
    changed=copy.deepcopy(preview);changed['candidates'][0]['predicted_tpot_ns']+=1;save(path,changed)
    with pytest.raises(ValueError,match='top4/anchor'):runner._fresh(preview,api.record(path),process)
    changed=copy.deepcopy(preview);changed['fit']['r2']=0.5;save(path,changed)
    with pytest.raises(ValueError,match='global fit diagnostic'):runner._fresh(preview,api.record(path),process)
    direct=copy.deepcopy(preview);direct.pop('stage_catalog');direct.pop('stage_catalog_scope')
    assert select.native_solve_command(direct,prefix)[select.native_solve_command(direct,prefix).index('--mx-decode-v2-solve')+1]==preview['stage_table']['path']


def test_global_corpus_conflicting_physical_rows_cannot_hide_in_different_models(tmp_path,monkeypatch,api):
    h=Harness(tmp_path,api,monkeypatch)
    for number,index in enumerate((0,10)):
        item=h.items[index];path=Path(item['table']['path']);table=api.load(path)
        table['rows']=[dict(kind='gemm',key=dict(n=128,k=128,b=1,threads=512,tn=16,wn=1,depth=1,prefetch=1,op=0,norm=1,part=1),CPU_SYNTHETIC_NO_MEASUREMENT=number)]
        save(path,table);item['table']=api.record(path)
    save(h.index,dict(tables=h.items,qualified_scoped_tables=50))
    with pytest.raises(ValueError,match='conflicting duplicate'):h.pack()
    assert not h.out.exists()


@pytest.mark.parametrize('field,value',[('batch',True),('threads',512.0),('model',None)])
def test_actual_request_types_are_not_coerced_into_a_published_scope(tmp_path,monkeypatch,api,field,value):
    h=Harness(tmp_path,api,monkeypatch);h.pack();request=h.request();request[field]=value
    with pytest.raises((ValueError,TypeError)):api.select_catalog(h.out,request)


def test_relative_leaf_assets_survive_bundle_copy_without_rewriting_leaf_bytes(tmp_path,monkeypatch,api):
    import shutil
    h=Harness(tmp_path,api,monkeypatch);h.pack();moved=tmp_path/'relocated';shutil.copytree(h.out.parent,moved)
    binding=api.select_catalog(moved/h.out.name,h.request())
    assert Path(binding['table']['path']).is_relative_to(moved)
    assert binding['table']['sha256']==h.items[0]['table']['sha256']


def test_qualified_but_unindexed_replacement_leaf_is_rejected(tmp_path,monkeypatch,api):
    h=Harness(tmp_path,api,monkeypatch);h.pack();doc=api.load(h.out);item=doc['tables'][0];leaf=h.out.parent/item['table']['path']
    leaf.write_bytes(leaf.read_bytes()+b'\n');item['table']['sha256']=api.sha(leaf);save(h.out,doc)
    with pytest.raises(ValueError,match='recorded export index'):api.load_catalog(h.out)


def test_pack_does_not_capture_changed_leaf_bytes_after_validation(tmp_path,monkeypatch,api):
    h=Harness(tmp_path,api,monkeypatch);previous=api._validate_leaf;changed=False
    def mutate(path,table,domain):
        nonlocal changed
        previous(path,table,domain)
        if not changed:Path(path).write_bytes(Path(path).read_bytes()+b'\n');changed=True
    monkeypatch.setattr(api,'_validate_leaf',mutate)
    with pytest.raises(ValueError,match='changed during qualification'):h.pack()
    assert not h.out.exists()


@pytest.mark.parametrize('change',['root','leaf'])
def test_live_catalog_or_asset_mutation_cannot_be_rebound_to_new_hash(tmp_path,monkeypatch,api,change):
    h=Harness(tmp_path,api,monkeypatch);h.pack();previous=api._validate_leaf;changed=False
    def mutate(path,table,domain):
        nonlocal changed
        previous(path,table,domain)
        if not changed:
            target=h.out if change=='root' else Path(path);target.write_bytes(target.read_bytes()+b'\n');changed=True
    monkeypatch.setattr(api,'_validate_leaf',mutate)
    with pytest.raises(ValueError,match='changed'):api.select_catalog(h.out,h.request())
