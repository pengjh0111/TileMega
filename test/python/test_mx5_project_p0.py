"""Synthetic CPU projector fixtures; actual pools are separately audited read-only."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(os.environ.get('TILEMEGA_TEST_ROOT',Path(__file__).resolve().parents[2]))
TREE=Path(os.environ.get('TILEMEGA_P0_PROJECT_TREE',ROOT))

if os.environ.get('TILEMEGA_P0_PROJECT_CHILD')!='1':
    def test_projector_cpu_cases_in_isolated_fixture_process(tmp_path):
        result=subprocess.run([sys.executable,'-m','pytest','-q',str(Path(__file__).resolve())],env=dict(os.environ,TILEMEGA_P0_PROJECT_CHILD='1',CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES=''),capture_output=True,text=True)
        (tmp_path/'CPU_SYNTHETIC_projector.log').write_text(result.stdout+result.stderr)
        assert result.returncode==0,result.stdout+result.stderr
else:
    def module(name,path):
        spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj);return obj
    f=module('p0_project_canonical_fixtures',ROOT/'test/python/test_mx5_canonical576_history.py')
    c=module('tilemega._p0_project_canonical_cpu',TREE/'python/tilemega/mx5_canonical576_history.py');c.resolved=f.m.resolved;c.old=f.m.old;c.legacy=f.m.legacy
    m=module('tilemega._p0_project_cpu',TREE/'python/tilemega/mx5_project_p0.py');m.canonical=c

    def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n');return m.record(path)
    def jsonl(path,rows):path.write_text(''.join(json.dumps(row)+'\n' for row in rows));return m.record(path)

    class Harness:
        def __init__(self,root,monkeypatch):
            self.root=root;self.h=f.Fixture(root,monkeypatch);self.gh={};self.attention=[]
            for name,refs in self.h.bundle['references'].items():
                for ref in refs:
                    if name!='attention':self.gh.setdefault(m._identity(ref),ref);continue
                    original=copy.deepcopy(ref);original['probe'].pop('raw_query')
                    key=dict(original['key']);key.pop('prefetch_kind')
                    self.attention.append(dict(kind='attention',key=key,library=original['probe']['binary']['path'],fits=original['fits'],query_only=True,iterations=0,query_status=0,
                        r_next_reference=original,uses=[dict(model='CPU_SYNTHETIC',stage='attention')]))
            self.g=Path(save(root/'GH.references.json',dict(references=list(self.gh.values())))['path'])
            self.a=root/'A_original_inspect.jsonl';jsonl(self.a,self.attention)
            self.pools=[self.g,self.a];self.archive=Path(self.h.sources[0]['archive']['path'])
        def project(self,out='projected'):
            return m.project(self.h.origin_ref['path'],0,self.pools,[self.archive],self.h.domain,self.root/out)

    def test_projection_keeps_flat_refs_and_binds_original_inline_file_line_without_new_raw_rows(tmp_path,monkeypatch):
        h=Harness(tmp_path,monkeypatch);old=[p.read_bytes() for p in h.pools];origin=Path(h.h.origin_ref['path']).read_bytes();result=h.project()
        assert result['action']==c.Action.PRIMARY.value and result['selection_anchor']==0 and result['missing']==[]
        bundle=m.load(result['p0_bundle']['path']);assert bundle['origin_sha256']==h.h.origin.sha256 and bundle['candidate_id']==0
        for name,refs in bundle['references'].items():
            if name!='attention':assert refs[0]==h.gh[m._identity(refs[0])]
        refs=bundle['references']['attention'];assert [r['key']['length'] for r in refs]==list(c.GRID)
        for number,ref in enumerate(refs,1):
            raw=ref['probe']['raw_query'];assert raw==dict(m.record(h.a),line=number,encoding=m.INLINE)
            unbound=copy.deepcopy(ref);unbound['probe'].pop('raw_query');assert unbound==h.attention[number-1]['r_next_reference']
        assert [p.read_bytes() for p in h.pools]==old and Path(h.h.origin_ref['path']).read_bytes()==origin
        pin=result['derivation']['details']['canonical_primary_pin']['mx_decode_v2'];assert pin['attn_prefetch']==1 and pin['qkv']['r_next']==216
        assert refs[0]['fits'] is False and refs[0]['resources']['local_bytes']==616
        assert result['execution_authorized'] is False and not list((tmp_path/'projected').glob('*raw*.jsonl'))

    def test_missing_attention_grid_point_stays_explicit_needs_p0_without_borrowing(tmp_path,monkeypatch):
        h=Harness(tmp_path,monkeypatch);jsonl(h.a,h.attention[:-1]);result=h.project()
        assert result['action']==c.Action.NEEDS_P0.value
        assert result['missing']==[dict(stage='attention',kind='attention',key={**h.attention[-1]['r_next_reference']['key']})]
        assert result['derivation']['details']['canonical_primary_pin'] is None
        assert len(m.load(result['p0_bundle']['path'])['references']['attention'])==4

    @pytest.mark.parametrize('change',['status','null_status','iterations','library','key','fits','missing_inline','ns'])
    def test_original_inspect_outer_query_fields_must_match_actual_nested_reference(tmp_path,monkeypatch,change):
        h=Harness(tmp_path,monkeypatch);row=h.attention[0]
        if change=='status':row['query_status']=7
        elif change=='null_status':row['query_status']=None
        elif change=='iterations':row['iterations']=1
        elif change=='library':row['library']='not_the_queried_binary.so'
        elif change=='key':row['key']['length']=256
        elif change=='fits':row['fits']=True
        elif change=='missing_inline':row.pop('r_next_reference')
        else:row['median_ns']=0
        jsonl(h.a,h.attention)
        with pytest.raises((ValueError,KeyError)):h.project()
        assert not (tmp_path/'projected/p0_bundle.json').exists()

    @pytest.mark.parametrize('change',['duplicate','missing_raw','source','attrs','forced_flags'])
    def test_duplicate_missing_flat_witness_source_attrs_and_forced_probe_rejected(tmp_path,monkeypatch,change):
        h=Harness(tmp_path,monkeypatch);refs=list(h.gh.values())
        if change=='duplicate':refs.append(copy.deepcopy(refs[0]))
        elif change=='missing_raw':refs[0]['probe'].pop('raw_query')
        elif change=='source':refs[0]['provenance']['source_sha256']='0'*64
        elif change=='attrs':refs[0]['resources']['local_bytes']-=1
        else:
            ref=refs[0];path=Path(ref['probe']['build_metadata']['path']);meta=m.load(path);meta['command']+=['-mllvm','--max-mtreg-number=255'];ref['probe']['build_metadata']=save(path,meta)
        save(h.g,dict(references=refs))
        with pytest.raises(ValueError):h.project()

    def test_actual_archive_inventory_uses_digest_and_rejects_old_or_missing_epoch(tmp_path,monkeypatch):
        h=Harness(tmp_path,monkeypatch);result=h.project();inventory=m.load(result['stage_sources']['path'])
        assert inventory==h.h.sources and result['archive_members'][h.h.sha]['files']==4
        old=tmp_path/'stage-sources.tar.gz'
        with pytest.raises(ValueError,match='missing'):m.archive_inventory(list(h.gh.values()),[old])

    def test_pool_audit_has_no_origin_costs_or_candidate_bundle_and_preserves_nonfit_probes(tmp_path,monkeypatch):
        h=Harness(tmp_path,monkeypatch);result=m.audit_pools(h.pools,[h.archive],tmp_path/'audit')
        assert result['validated576']==len(h.gh)+5 and result['candidate_origin'] is None
        assert not result['candidate_bundle_created'] and result['cost_rows_created']==0
        assert result['nonfitting_compiler_probes']['attention']==5 and result['observed_maxima']['attention']['local_bytes']==616
        with pytest.raises(ValueError,match='preserved'):m.audit_pools(h.pools,[h.archive],tmp_path/'audit')

    def test_wrong_origin_id_and_shifted_inline_raw_line_are_not_reinterpreted(tmp_path,monkeypatch):
        h=Harness(tmp_path,monkeypatch)
        with pytest.raises(ValueError):m.project(h.h.origin_ref['path'],4,h.pools,[h.archive],h.h.domain,tmp_path/'wrong_id')
        entries,_,_=m.read_pools(h.pools);ref=next(e['reference'] for e in entries.values() if e['reference']['kind']=='attention')
        ref['probe']['raw_query']['line']=2
        with pytest.raises(ValueError,match='inline'):m._validate(ref,c.resolved._qualifier().table)

    def test_flat_legacy_path_unchanged_and_inline_marker_cannot_label_a_gemm_query(tmp_path,monkeypatch):
        h=Harness(tmp_path,monkeypatch);table=c.resolved._qualifier().table
        for ref in h.gh.values():assert m._validate(ref,table)['resources']['mtreg']==216
        ref=copy.deepcopy(next(iter(h.gh.values())));ref['probe']['raw_query']['encoding']=m.INLINE
        with pytest.raises(ValueError,match='bounded to attention'):m._validate(ref,table)

    def test_projected_bundle_and_inventory_feed_the_existing_canonical_stepper(tmp_path,monkeypatch):
        h=Harness(tmp_path,monkeypatch);preview=m.load(h.h.origin_ref['path'])
        target=save(tmp_path/'CPU_target.json',dict(fixture='CPU synthetic target, not compiled'))
        preview['request'].update(runtime_target=target['path'],grid=104)
        h.h.origin_ref=save(tmp_path/'CPU_origin_with_request.json',preview);h.h.origin=c.resolved.Qualified512Origin.capture(h.h.origin_ref)
        result=h.project();stepper=module('tilemega._project_stepper_cpu',ROOT/'python/tilemega/mx5_prepare_canonical576.py')
        stepper.history=c;monkeypatch.setattr(stepper.common,'validate_preview',lambda path,**kwargs:m.load(path))
        planned=stepper.prepare(h.h.origin_ref['path'],0,result['p0_bundle']['path'],result['stage_sources']['path'],h.h.domain,7,tmp_path/'prepared')
        assert planned['action']=='needs_actual_native_receipt'
        assert m.load(planned['build']['pin']['path'])==result['derivation']['details']['canonical_primary_pin']
        assert planned['build']['argv'][-2:]==['--watchdog','0']
