"""Synthetic CPU original512/history handoff; never real compilation, query or approval."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT=Path(os.environ.get('TILEMEGA_TEST_ROOT',Path(__file__).resolve().parents[2]))
MODULE=Path(os.environ.get('TILEMEGA_PREPARE_ORIGIN512_MODULE',ROOT/'python/tilemega/mx5_prepare_origin512.py'))

if os.environ.get('TILEMEGA_ORIGIN512_TEST_CHILD')!='1':
    def test_isolated_synthetic_cpu_suite(tmp_path):
        env=dict(os.environ,TILEMEGA_ORIGIN512_TEST_CHILD='1',CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='')
        result=subprocess.run([sys.executable,'-m','pytest','-q',str(Path(__file__).resolve())],env=env,capture_output=True,text=True)
        (tmp_path/'CPU_SYNTHETIC_fixture.log').write_text(result.stdout+result.stderr)
        assert result.returncode==0,result.stdout+result.stderr
else:
    spec=importlib.util.spec_from_file_location('origin512_canonical_CPU_fixture',ROOT/'test/python/test_mx5_canonical576_history.py')
    f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    spec=importlib.util.spec_from_file_location('tilemega.mx5_prepare_origin512',MODULE)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);m.history=f.m

    class Fixture:
        def __init__(self,root,monkeypatch,two_halves=False):
            self.root=root;self.h=f.Fixture(root,monkeypatch,two_halves=two_halves)
            target=f.save(root/'CPU_SYNTHETIC_target.json',{'CPU_SYNTHETIC':True})
            self.h.h.preview['request']['runtime_target']=target['path']
            self.h.origin_ref=f.save(self.h.h.h.preview_file,self.h.h.preview)
            self.h.origin=m.history.resolved.Qualified512Origin.capture(self.h.origin_ref)
            self.h.h.h.origin=self.h.origin;self.h.h.h.h.origin=self.h.origin
            self.h.bundle['origin_sha256']=self.h.origin.sha256;self.h.p0_ref=f.save(root/'CPU_SYNTHETIC_p0.json',self.h.bundle)
            self.refs,self.live=self.h.history();assert self.live.action==m.history.Action.HISTORY,self.live.reason
            self.history_ref=f.save(root/'CPU_SYNTHETIC_history.json',self.live.as_dict());self.sources=f.save(root/'CPU_SYNTHETIC_sources.json',self.h.sources)
            self.state=m.prepare(self.history_ref['path'],self.sources['path'],self.h.domain,root/'prepared');self.preparation=self.state['preparation']['path']

        def artifact(self,**kwargs):
            state=m.load(self.preparation);job=state['build'];directory=Path(job['binary']).parent
            old=Path(job['pin']['path']).read_bytes();Path(job['pin']['path']).unlink();directory.rmdir()
            factory=self.h.h.h;factory.root=Path(state['workspace']);factory.count=0
            bundle=factory.artifact(index=0,**kwargs);self.bundle=m.load(bundle['path'])
            assert Path(job['pin']['path']).read_bytes()==old
            # Real native commands share one CUDA installation across576/512;
            # the synthetic factory otherwise derives -L from its output root.
            native_path=Path(self.bundle['native_receipt']['path']);native=m.load(native_path)
            native['argv']=[('-L'+str(self.root/'cuda-lib') if x.startswith('-L') else x)for x in native['argv']]
            command=Path(native['files']['command']['path']);command.write_text(shlex.join(native['argv'])+'\n')
            native['files']['command']=m.record(command);native['command_sha256']=native['files']['command']['sha256'];self.bundle['native_receipt']=f.save(native_path,native)
            if self.bundle['query']:
                qp=Path(self.bundle['query']['path']);q=m.load(qp);q['bound_inputs']['context']=self.bundle['native_receipt'];q['inputs_manifest']=f.save(Path(q['inputs_manifest']['path']),q['bound_inputs']);self.bundle['query']=f.save(qp,q)
            f.save(Path(bundle['path']),self.bundle);return self.bundle

        def consume(self,name='consumed',occupancy=None):return m.consume(self.preparation,self.root/name,occupancy)

    @pytest.mark.parametrize('two,used,left',[(False,0,3),(True,2,1)])
    def test_real_history_budget_and_exact_independent_original_preserved(tmp_path,monkeypatch,two,used,left):
        h=Fixture(tmp_path,monkeypatch,two_halves=two);state,origin,_=m._prepared(h.preparation)
        b=state['context']['resource_budget'];assert b['used_576_reductions']==used and b['used_512_reductions']==0 and b['remaining_reductions']==left
        assert state['context']['origin_candidates']==origin.data['preview']['candidates'] and [c['index']for c in state['context']['origin_candidates']]==[0,1,2,3]
        assert m.load(state['build']['pin']['path'])==origin.candidate(0)['pin']
        assert state['build']['argv'][-2:]==['--watchdog','0'] and state['selection_anchor']==0
        assert h.consume()['action']=='needs_actual_native_receipt' and not(tmp_path/'consumed').exists()

    def test_pending_real_query_then_budget_bound_D7_handoff_not_approval(tmp_path,monkeypatch):
        h=Fixture(tmp_path,monkeypatch,two_halves=True);b=h.artifact(mt=220,frame=0)
        pending=h.consume('pending');assert pending['action']=='needs_actual512_query' and not pending['execution_authorized']
        assert m.load(pending['artifact']['path'])['query'] is None
        r=h.consume('queried',b['query']['path']);assert r['action']==m.history.Action.NEEDS_D7.value,r
        bound=m.load(r['binding']['path']);assert bound['details']['resource_budget']['remaining_reductions']==1
        assert bound['details']['prediction']['origin_predicted_tpot_ns']==h.h.origin.candidate(0)['predicted_tpot_ns']
        assert not r['execution_authorized'] and not r['default_enabled']
        from tilemega import mx5_resolved_top4_v2 as consumer
        with pytest.raises(ValueError):consumer.capture_plan(h.h.origin_ref,0,domain_tool=h.h.domain,history_ref=h.history_ref,lineage512=r['lineage512'],d7_ref=None)
        with pytest.raises(ValueError,match='preserve'):h.consume('queried',b['query']['path'])

    @pytest.mark.parametrize('kwargs',[dict(mt=256,frame=0),dict(mt=220,frame=8)])
    def test_failure_returns_only_remaining_budget_next_proposal(tmp_path,monkeypatch,kwargs):
        h=Fixture(tmp_path,monkeypatch,two_halves=True);h.artifact(query=False,**kwargs);r=h.consume()
        assert r['action']==m.history.Action.NEXT_512.value,r
        bound=m.load(r['binding']['path']);assert bound['details']['resource_budget']['remaining_reductions']==1
        auth=bound['details']['authorization'];assert auth['shared_class']=='qkv' and auth['from_pw']==16 and auth['to_pw']==8
        assert len(r['lineage512'])==1 and r['catalogs']==[]
        assert m.load(m.load(h.preparation)['build']['pin']['path'])==h.h.origin.candidate(0)['pin']
        assert not(Path(m.load(h.preparation)['workspace'])/'artifact1').exists()

    @pytest.mark.parametrize('kwargs',[dict(shared=4),dict(extra=('-mllvm','--max-mtreg-number=255')),dict(mir_extra='  CALL forbidden'),dict(mir_extra='  LDX_B32 $r0'),dict(mir_transform=lambda s:s.replace('ARRIVE 4096','ARRIVE 4160'))])
    def test_contract_failure_never_becomes_query_or_reduction(tmp_path,monkeypatch,kwargs):
        h=Fixture(tmp_path,monkeypatch);h.artifact(query=False,**kwargs);r=h.consume();assert r['action']=='blocked_evidence',r
        assert not(tmp_path/'consumed/occupancy_inputs.json').exists()

    @pytest.mark.parametrize('change',[
        lambda s:s['context']['resource_budget'].update(used_576_reductions=0,remaining_reductions=3),
        lambda s:s.update(selection_anchor=1),
        lambda s:s['context']['origin_candidates'][0]['ranks'].__setitem__(0,99),
        lambda s:s['build']['argv'].append('--override'),
    ])
    def test_preparation_budget_anchor_ranks_or_command_reset_rejected(tmp_path,monkeypatch,change):
        h=Fixture(tmp_path,monkeypatch,two_halves=True);s=m.load(h.preparation);change(s);p=tmp_path/'CPU_forged.json';f.save(p,s)
        with pytest.raises(ValueError):m._prepared(p)

    def test_reset_or_bare_floor_history_cannot_prepare_original(tmp_path,monkeypatch):
        h=Fixture(tmp_path,monkeypatch,two_halves=True);saved=m.load(h.history_ref['path']);saved['details']['used_576_reductions']=0;saved['details']['remaining_512_reductions']=3
        bad=f.save(tmp_path/'CPU_counterfeit_zero.json',saved)
        with pytest.raises(ValueError):m.prepare(bad['path'],h.sources['path'],h.h.domain,tmp_path/'bad_prepare')
        saved=m.load(h.history_ref['path']);saved['details']['request']['attempts']=[saved['details']['request']['attempts'][-1]];bad=f.save(tmp_path/'CPU_bare_floor.json',saved)
        with pytest.raises(ValueError):m.prepare(bad['path'],h.sources['path'],h.h.domain,tmp_path/'bare_prepare')

    def test_fitting576_veto_is_not_a512_history(tmp_path,monkeypatch):
        h=Fixture(tmp_path,monkeypatch);saved=m.load(h.history_ref['path']);saved['action']=m.history.Action.USE_576.value;bad=f.save(tmp_path/'CPU_fitting_veto.json',saved)
        with pytest.raises(ValueError,match='veto'):m.prepare(bad['path'],h.sources['path'],h.h.domain,tmp_path/'bad_prepare')

    def test_512_source_family_cannot_change_after_actual576_history(tmp_path,monkeypatch):
        h=Fixture(tmp_path,monkeypatch);actor=h.h.h.h.h;p=actor.source_root/'CPU_epoch.py';p.write_text('# CPU synthetic source epoch difference\n');actor.source_files['CPU_epoch.py']=m.record(p)['sha256']
        h.artifact(query=False);r=h.consume();assert r['action']=='blocked_evidence' and 'source/compiler/options'in r['reason']

    def test_compiler_and_actual_MOCK_query_failures_do_not_authorize_retry(tmp_path,monkeypatch):
        h=Fixture(tmp_path,monkeypatch);b=h.artifact(query=True,resident=0);r=h.consume(occupancy=b['query']['path']);assert r['action']=='blocked_evidence'
        p=Path(b['native_receipt']['path']);native=m.load(p);native.update(returncode=7,wait_status=7<<8);f.save(p,native)
        r=h.consume('compiler_failure');assert r['action']=='blocked_evidence' and 'compiler failure'in r['reason']

    def test_explicit_run_build_uses_only_normal_command_CPU_mock(tmp_path,monkeypatch):
        h=Fixture(tmp_path,monkeypatch);calls=[]
        def fake(argv,directory,**kwargs):
            e=kwargs['env'];assert e['CUDA_VISIBLE_DEVICES']==e['MACA_VISIBLE_DEVICES']=='' and 'CUDACXX'not in e and not any(n.startswith('TILEMEGA_')for n in e)
            calls.append(argv);return dict(returncode=7,CPU_SYNTHETIC_MOCK=True)
        monkeypatch.setenv('CUDACXX','/forbidden/override');monkeypatch.setenv('TILEMEGA_MX_DIAG','1');monkeypatch.setattr(m.common,'run_command',fake)
        r=m.run_build(h.preparation);assert len(calls)==1 and not r['execution_authorized']
