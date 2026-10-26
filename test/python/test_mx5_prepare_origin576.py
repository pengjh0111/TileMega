"""Synthetic CPU parser/control tests; no real cost, compile, GPU or query evidence."""
import copy
import importlib.util
import json
import os
import shlex
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(os.environ.get('TILEMEGA_TEST_ROOT',Path(__file__).resolve().parents[2]))
MODULE=Path(os.environ.get('TILEMEGA_PREPARE_ORIGIN576_MODULE',ROOT/'python/tilemega/mx5_prepare_origin576.py'))

if os.environ.get('TILEMEGA_ORIGIN576_TEST_CHILD')!='1':
    def test_isolated_synthetic_cpu_suite(tmp_path):
        env=dict(os.environ,TILEMEGA_ORIGIN576_TEST_CHILD='1',CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='')
        p=subprocess.run([sys.executable,'-m','pytest','-q',str(Path(__file__).resolve())],env=env,capture_output=True,text=True)
        (tmp_path/'CPU_SYNTHETIC_fixture.log').write_text(p.stdout+p.stderr)
        assert p.returncode==0,p.stdout+p.stderr
else:
    spec=importlib.util.spec_from_file_location('true576_existing_CPU_fixture',ROOT/'test/python/test_mx5_resolved_top4.py')
    f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    spec=importlib.util.spec_from_file_location('tilemega.mx5_prepare_origin576',MODULE)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    m.resolved=f.m;m.policy=f.m.legacy

    class Fixture:
        def __init__(self,path,monkeypatch,index=0):
            self.h=f.Fixture(path,monkeypatch);self.root=path;self.index=index
            target=f.save(path/'CPU_SYNTHETIC_target.json',{'CPU_SYNTHETIC':True})
            self.h.preview['request']['runtime_target']=target['path']
            self.h.preview_ref=f.save(self.h.h.preview_file,self.h.preview)
            self.h.h.origin=m.resolved.Qualified576Origin.capture(self.h.preview_ref);self.h.h.h.origin=self.h.h.origin
            self.sources=f.save(path/'CPU_SYNTHETIC_sources.json',self.h.h.h.sources)
            self.state=m.prepare(self.h.preview_ref['path'],index,self.sources['path'],self.h.domain,path/'prepared')
            self.preparation=self.state['preparation']['path']

        def artifact(self,**kwargs):
            state=m.load(self.preparation);job=state['build'];directory=Path(job['binary']).parent
            original=Path(job['pin']['path']).read_bytes()
            # CPU scaffold writes synthetic ELF/MIR/receipts at the identical
            # preplanned path. No executable compiler or runtime is invoked.
            Path(job['pin']['path']).unlink();directory.rmdir()
            self.h.h.root=Path(state['workspace']);self.h.h.count=0
            bundle=self.h.h.artifact(index=self.index,**kwargs)
            assert Path(job['pin']['path']).read_bytes()==original
            self.bundle=m.load(bundle['path'])
            # The legacy fixture derives -L from its artifact root; real cucc
            # uses one installation for all originals. Keep this synthetic
            # library path fixed, rebind every changed receipt/query hash,
            # and still run the unchanged normal-options/epoch validators.
            native_path=Path(self.bundle['native_receipt']['path']);native=m.load(native_path)
            native['argv']=[('-L'+str(self.root/'CPU_cuda_lib') if x.startswith('-L') else x) for x in native['argv']]
            command=Path(native['files']['command']['path']);command.write_text(shlex.join(native['argv'])+'\n')
            native['files']['command']=m.record(command);native['command_sha256']=native['files']['command']['sha256']
            self.bundle['native_receipt']=f.save(native_path,native)
            if self.bundle['query']:
                query_path=Path(self.bundle['query']['path']);row=m.load(query_path)
                row['bound_inputs']['context']=self.bundle['native_receipt']
                row['inputs_manifest']=f.save(Path(row['inputs_manifest']['path']),row['bound_inputs'])
                self.bundle['query']=f.save(query_path,row)
            f.save(Path(bundle['path']),self.bundle)
            return self.bundle

        def consume(self,name='consumed',occupancy=None):return m.consume(self.preparation,self.root/name,occupancy)

    def test_exact_independent_origin_normal_WD0_and_all_four_unchanged(tmp_path,monkeypatch):
        h=Fixture(tmp_path,monkeypatch,index=2);state,origin,_=m._prepared(h.preparation)
        assert state['selection_anchor']==0 and state['origin_candidates']==h.h.preview['candidates']
        assert [c['index']for c in state['origin_candidates']]==list(range(4))
        assert m.load(state['build']['pin']['path'])==origin.candidate(2)['pin']
        assert state['build']['argv'][-2:]==['--watchdog','0'] and state['build']['timeout_seconds']==900
        assert '--mx-decode-v2-solve'not in state['build']['argv']
        assert h.consume()['action']=='needs_actual_native_receipt' and not(tmp_path/'consumed').exists()

    def test_pending_query_then_existing_ResolvedTop4_terminal(tmp_path,monkeypatch):
        h=Fixture(tmp_path,monkeypatch);b=h.artifact(mt=220,frame=0)
        pending=h.consume('pending');assert pending['action']=='needs_actual576_query'
        saved=m.load(pending['ledger']['path']);assert saved['attempts'][0]['verified']['classification']=='blocked' and not saved['authorizations']
        assert pending['occupancy_inputs'] and not pending['execution_authorized']
        result=h.consume('with_actual_MOCK_query',b['query']['path']);assert result['action']=='review_ready_true576_plan',result
        plan=m.load(result['resolved_plan']['path']);assert plan['route']=='576' and plan['candidate_id']==0
        assert plan['final']['effective_prediction']['effective_predicted_tpot_ns']==h.h.preview['candidates'][0]['predicted_tpot_ns']
        assert not result['execution_authorized']
        with pytest.raises(ValueError,match='preserve'):h.consume('with_actual_MOCK_query',b['query']['path'])

    def test_all_four_exact_originals_reach_existing_ResolvedTop4_in_CPU_fixture(tmp_path,monkeypatch):
        h=Fixture(tmp_path,monkeypatch);plans=[]
        original=copy.deepcopy(h.h.preview['candidates'])
        for index in range(4):
            if index:
                h.index=index;h.state=m.prepare(h.h.preview_ref['path'],index,h.sources['path'],h.h.domain,tmp_path/f'prepared{index}');h.preparation=h.state['preparation']['path']
            b=h.artifact(mt=220,frame=0);r=h.consume(f'consumed{index}',b['query']['path'])
            assert r['action']=='review_ready_true576_plan',r
            plans.append(m.load(r['resolved_plan']['path']))
        protocol=m.resolved.capture_protocol(h.h.preview_ref,h.h.domain,plans)
        path=tmp_path/'CPU_SYNTHETIC_resolved_protocol.json';f.save(path,protocol)
        result=m.resolved.validate_resolved_top4(path,domain_tool=h.h.domain)
        assert result.action==m.resolved.Action.READY,result.reason
        assert protocol['origin_candidates']==original and [x['candidate_id']for x in protocol['plans']]==[0,1,2,3]
        assert not result.as_dict()['execution_authorized']

    @pytest.mark.parametrize('kwargs',[dict(mt=228,frame=0),dict(mt=168,frame=8)])
    def test_real_parser_failure_returns_only_existing_deterministic_cost_need(tmp_path,monkeypatch,kwargs):
        h=Fixture(tmp_path,monkeypatch);h.artifact(query=False,**kwargs);r=h.consume()
        assert r['action']=='needs_stage_measurements',r
        auth=r['policy_action']['details']['authorization'];assert auth['shared_class']=='qkv' and auth['from_pw']==16 and auth['to_pw']==8
        ledger=m.load(r['ledger']['path']);assert len(ledger['attempts'])==1 and ledger['authorizations']==[] and ledger['cost_bindings']==[]
        assert ledger['effective_pin']==h.h.preview['candidates'][0]['pin']
        assert not(Path(m.load(h.preparation)['workspace'])/'artifact1').exists()

    @pytest.mark.parametrize('kwargs',[dict(shared=4),dict(extra=('-mllvm','--max-mtreg-number=255')),dict(mir_extra='  CALL forbidden'),dict(mir_extra='  LDX_B32 $r0'),dict(mir_transform=lambda s:s.replace('ARRIVE 4096','ARRIVE 4160'))])
    def test_non_resource_contract_failures_do_not_become_retry_or_query(tmp_path,monkeypatch,kwargs):
        h=Fixture(tmp_path,monkeypatch);h.artifact(query=False,**kwargs);r=h.consume()
        assert r['action']=='blocked_evidence',r
        assert not(tmp_path/'consumed/occupancy_inputs.json').exists()

    def test_compiler_failure_is_not_inferred_as_MT_failure(tmp_path,monkeypatch):
        h=Fixture(tmp_path,monkeypatch);b=h.artifact(query=False,mt=228)
        p=Path(b['native_receipt']['path']);v=m.load(p);v.update(returncode=7,wait_status=7<<8);f.save(p,v)
        r=h.consume();assert r['action']=='blocked_evidence' and 'compiler failure'in r['reason']

    @pytest.mark.parametrize('mutate',[
        lambda s:s.update(selection_anchor=1),
        lambda s:s['origin_candidates'][1]['ranks'].__setitem__(0,99),
        lambda s:s['build']['argv'].append('--force'),
        lambda s:s.update(candidate_id=True),
    ])
    def test_preparation_cannot_change_anchor_origin_or_build(tmp_path,monkeypatch,mutate):
        h=Fixture(tmp_path,monkeypatch);s=m.load(h.preparation);mutate(s);p=tmp_path/'counterfeit.json';f.save(p,s)
        with pytest.raises(ValueError):m._prepared(p)

    def test_changed_body_header_stops_source_archive_and_snapshot_gate(tmp_path,monkeypatch):
        h=Fixture(tmp_path,monkeypatch);h.artifact(query=False);h.h.h.h.header.write_text('changed CPU synthetic body')
        r=h.consume();assert r['action']=='blocked_evidence'

    def test_actual_MOCK_occupancy_zero_stops_without_retry(tmp_path,monkeypatch):
        h=Fixture(tmp_path,monkeypatch);b=h.artifact(resident=0);r=h.consume(occupancy=b['query']['path'])
        assert r['action']=='blocked_evidence' and 'occupancy failure'in r['reason']

    def test_no_build_or_GPU_on_prepare_and_normal_explicit_build_environment(tmp_path,monkeypatch):
        h=Fixture(tmp_path,monkeypatch);seen=[]
        def fake(argv,directory,**kw):
            env=kw['env'];assert env['CUDA_VISIBLE_DEVICES']==env['MACA_VISIBLE_DEVICES']==''
            assert not any(k.startswith('TILEMEGA_')for k in env) and 'CUDACXX'not in env
            assert argv[-2:]==['--watchdog','0'];seen.append(argv);return dict(returncode=7,CPU_SYNTHETIC_MOCK=True)
        monkeypatch.setenv('TILEMEGA_MX_DIAG','1');monkeypatch.setenv('CUDACXX','/forbidden/override')
        monkeypatch.setattr(m.common,'run_command',fake)
        r=m.run_build(h.preparation);assert len(seen)==1 and not r['execution_authorized']
