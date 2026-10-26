"""Synthetic CPU stepper tests. No real compiler, GPU, timings or approval."""
import copy
import ctypes as C
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(os.environ.get('TILEMEGA_TEST_ROOT',Path(__file__).resolve().parents[2]))
MODULE=Path(os.environ.get('TILEMEGA_PREPARE576_MODULE',ROOT/'python/tilemega/mx5_prepare_canonical576.py'))

if os.environ.get('TILEMEGA_PREPARE576_CHILD')!='1':
    def test_cpu_stepper_suite_in_isolated_fixture_process(tmp_path):
        # Legacy synthetic helpers deliberately reload package modules. Keep their
        # identity graph isolated from unrelated tests collected in this process.
        env=dict(os.environ,TILEMEGA_PREPARE576_CHILD='1',CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='')
        result=subprocess.run([sys.executable,'-m','pytest','-q',str(Path(__file__).resolve())],env=env,capture_output=True,text=True)
        (tmp_path/'CPU_SYNTHETIC_child.log').write_text(result.stdout+result.stderr)
        assert result.returncode==0,result.stdout+result.stderr
else:
    spec=importlib.util.spec_from_file_location('prepare576_canonical_fixture',ROOT/'test/python/test_mx5_canonical576_history.py')
    f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    spec=importlib.util.spec_from_file_location('tilemega.mx5_prepare_canonical576',MODULE)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);m.history=f.m

    def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n');return m.record(path)

    class Harness:
        def __init__(self,root,monkeypatch,*,catalog=False,cpu_process_fixture=False,**kwargs):
            self.root=root;self.h=f.Fixture(root,monkeypatch,**kwargs)
            target=save(root/'CPU_target.json',dict(fixture='CPU synthetic target, never executed'))
            preview=m.load(self.h.origin_ref['path']);preview['request'].update(runtime_target=target['path'],grid=104)
            if cpu_process_fixture:
                producer=Path(self.h.h.h.h.producer)
                producer.write_text('#!/usr/bin/env python3\nimport os,sys\nassert os.environ["CUDA_VISIBLE_DEVICES"]==os.environ["MACA_VISIBLE_DEVICES"]==""\nprint("CPU_SYNTHETIC_PRODUCER: no compiler or GPU invoked")\nsys.exit(9)\n');producer.chmod(0o755)
                preview['producer']=m.record(producer)
            if catalog:
                preview['stage_catalog']=save(root/'CPU_catalog_reference.json',dict(fixture='CPU synthetic reference only; native resolver is never executed'))
            self.h.origin_ref=save(root/'CPU_origin_with_request.json',preview)
            self.h.origin=f.m.resolved.Qualified512Origin.capture(self.h.origin_ref)
            self.h.h.h.origin=self.h.origin;self.h.h.h.h.origin=self.h.origin
            self.h.bundle['origin_sha256']=self.h.origin.sha256;self.h.p0_ref=save(root/'CPU_p0_bundle.json',self.h.bundle)
            self.sources=save(root/'CPU_sources.json',self.h.sources)
            monkeypatch.setattr(m.common,'validate_preview',lambda path,**kw:m.load(path))
            self.state=m.prepare(self.h.origin_ref['path'],0,self.h.p0_ref['path'],self.sources['path'],self.h.domain,7,root/'session')

        def build(self,**kwargs):
            step=m.load(self.state['step']['path']);job=step['build'];pin=m.load(job['pin']['path'])
            factory=self.h.h.h;factory.root=Path(step['context']['workspace']);factory.count=job['ordinal']
            # Factory writes an identical preplanned pin and a synthetic ELF/MIR/
            # native receipt. It never invokes a compiler or device runtime.
            before=Path(job['pin']['path']).read_bytes()
            # The legacy factory requires a fresh folder; recreate the synthetic
            # input at the identical path/bytes as part of that CPU scaffold.
            Path(job['pin']['path']).unlink();Path(job['binary']).parent.rmdir()
            artifact=factory.artifact(pin,ordinal=job['ordinal'],query=False,**dict(dict(mt=168,frame=8),**kwargs))
            assert Path(job['pin']['path']).read_bytes()==before
            self.artifact=artifact
            out=self.root/('query_step%d'%job['ordinal']);self.state=m.advance(self.state['step']['path'],out)
            return self.state

        def query(self,*,status=0,resident=1,card='7',change=None):
            step=m.load(self.state['step']['path']);job=step['query_job'];spec=m.load(job['inputs']['path']);o=f.m.old.occupancy
            validated=o.validate_inputs(spec,command=o.run_text);report=validated['report']['kernel']
            class Runtime:
                def attributes(self,pointer,address):
                    attrs=C.cast(pointer,C.POINTER(o.Attributes)).contents
                    attrs.numRegs=report['registers'];attrs.localSizeBytes=report['stack_bytes'];attrs.sharedSizeBytes=report['static_smem_bytes'];attrs.maxThreadsPerBlock=576;attrs.maxDynamicSharedSizeBytes=65536
                    return status
                def set_shared(self,*args):return 0
                def occupancy(self,pointer,*args):C.cast(pointer,C.POINTER(C.c_int)).contents.value=resident;return 0
            class Library:
                def shared_bytes(self):return 32816
                def stub_address(self,*args):return C.c_void_p(0x1234)
            guard=dict(physical_gpu=card,lock_path='/tmp/tilemega_mx_gpu7.lock',owner=dict(pid=11,command='python -m tilemega.bench.gpu_guard CPU_SYNTHETIC',proc_locks_line='1: FLOCK ADVISORY WRITE 11 00:00:1 0 EOF'),
                environment=dict(TILEMEGA_DEVICE_INDEX='7',CUDA_VISIBLE_DEVICES='7',MACA_VISIBLE_DEVICES='7',TILEMEGA_GPU_LOCK='/tmp/tilemega_mx_gpu7.lock',TILEMEGA_GPU_LOCK_HELD='1'))
            row=o.query(validated,Runtime(),Library(),guard);row.update(inputs_manifest=m.record(job['inputs']['path']),runtime_library=m.record(o.RUNTIME),helper_source=m.record(Path(o.__file__)))
            if change:change(row)
            save(Path(job['output']),row)
            out=self.root/('after_query%d'%step['build']['ordinal']);self.state=m.advance(self.state['step']['path'],out)
            return self.state

    def test_real_command_prepare_actual_receipt_query_and_floor_history(tmp_path,monkeypatch):
        h=Harness(tmp_path,monkeypatch);initial=m.load(h.state['step']['path']);pin=m.load(initial['build']['pin']['path'])
        assert initial['build']['argv'][-2:]==['--watchdog','0'] and initial['build']['timeout_seconds']==900
        assert pin['mx_decode_v2']['attn_prefetch']==1 and all(pin['mx_decode_v2'][n]['r_next']>0 for n in f.m.CLASSES)
        assert 'used_576_reductions' not in initial['audit']['details']
        assert h.build()['action']=='needs_actual576_query'
        step=m.load(h.state['step']['path']);assert step['query_job']['argv'][1:3]==['7','300']
        assert step['query_job']['argv'][-3:]==['--query-only','--iterations','0']
        assert h.query()['action']==f.m.Action.HISTORY.value
        final=m._validate_step(h.state['step']['path']);data=final['audit']['details']
        assert data['used_576_reductions']==0 and data['remaining_512_reductions']==3 and data['canonical_attention_kind']==1
        assert not final['execution_authorized'] and 'substantive' in final['needs'][-1]
        assert all(c['index']==i for i,c in enumerate(m.load(final['context']['origin_preview']['path'])['candidates']))

    def test_two_prescribed_halves_require_actual_query_and_resume_without_budget_reset(tmp_path,monkeypatch):
        h=Harness(tmp_path,monkeypatch,two_halves=True);widths=[]
        for ordinal in range(3):
            state=m._validate_step(h.state['step']['path']);widths.append(m.load(state['build']['pin']['path'])['mx_decode_v2']['qkv']['prefetch'])
            h.build();waiting=h.state;out=tmp_path/('still_waiting%d'%ordinal)
            assert m.advance(waiting['step']['path'],out)['action']=='needs_actual576_query' and not out.exists()
            h.query()
        assert widths==[4,2,1]
        final=m._validate_step(h.state['step']['path']);assert final['audit']['details']['remaining_512_reductions']==1
        changed=m.load(h.state['step']['path']);changed['attempts']=[];bad=save(tmp_path/'reset.json',changed)
        with pytest.raises(ValueError,match='reset|stripped'):m._validate_step(bad['path'])

    def test_three_reductions_do_not_create_fourth_or_bare_floor(tmp_path,monkeypatch):
        h=Harness(tmp_path,monkeypatch,long_chain=True)
        for ordinal in range(4):
            h.build()
            if h.state['action']=='blocked_evidence':break
            h.query()
        assert h.state['action']=='blocked_evidence' and 'exhausted' in h.state['reason']
        assert not (tmp_path/'session/artifact4').exists()

    @pytest.mark.parametrize('kwargs',[dict(shared=4),dict(extra=('-mllvm','--max-mtreg-number=255')),dict(mir_extra='  CALL forbidden_callee')])
    def test_smem_forced_flags_or_callee_never_authorize_query_or_next_width(tmp_path,monkeypatch,kwargs):
        h=Harness(tmp_path,monkeypatch,two_halves=True);result=h.build(**kwargs)
        assert result['action']=='blocked_evidence' and not (tmp_path/'session/artifact1').exists()

    @pytest.mark.parametrize('kwargs',[dict(status=1),dict(resident=0),dict(card='6'),dict(change=lambda r:r.update(query_status=None)),dict(change=lambda r:r.update(inputs_manifest={'path':'wrong','sha256':'0'*64}))])
    def test_actual_query_status_residency_card_or_binding_anomaly_stops(tmp_path,monkeypatch,kwargs):
        h=Harness(tmp_path,monkeypatch,two_halves=True);h.build();result=h.query(**kwargs)
        assert result['action']=='blocked_evidence' and not (tmp_path/'session/artifact1').exists()

    def test_fitting576_veto_keeps_true576_origin_separate(tmp_path,monkeypatch):
        h=Harness(tmp_path,monkeypatch);h.build(mt=167,frame=0);assert h.query()['action']==f.m.Action.USE_576.value
        state=m._validate_step(h.state['step']['path']);assert 'independent576_origin_job' not in state
        assert 'independent true576' in state['needs'][0]
        assert state['audit']['details']['actual_attempts'][-1]['pin']['mx_decode_v2']['threads']==576

    @pytest.mark.parametrize('change',['pin','id','epoch','approval'])
    def test_planned_pin_id_epoch_or_boolean_approval_mutation_rejected(tmp_path,monkeypatch,change):
        h=Harness(tmp_path,monkeypatch);path=Path(h.state['step']['path']);state=m.load(path)
        if change=='pin':
            pin=m.load(state['build']['pin']['path']);pin['mx_decode_v2']['attn_prefetch']=0;save(Path(state['build']['pin']['path']),pin);state['build']['pin']=m.record(state['build']['pin']['path'])
        elif change=='id':state['context']['candidate_id']=1
        elif change=='epoch':Path(state['context']['helpers']['history']['path']).resolve();state['context']['helpers']['history']['sha256']='f'*64
        else:state['confirmed_d7']=True
        bad=save(path.with_name('bad.json'),state)
        with pytest.raises(ValueError):m._validate_step(bad['path'])

    def test_run_build_executes_only_exact_cpu_command_and_preserves_actual_failure(tmp_path,monkeypatch):
        h=Harness(tmp_path,monkeypatch,cpu_process_fixture=True)
        result=m.run_build(h.state['step']['path'])
        assert result['process']['returncode']==9 and result['process']['pid']>1
        assert result['process']['argv'][-2:]==['--watchdog','0']
        process_folder=Path(result['process_file']['path']).parent
        assert 'CPU_SYNTHETIC_PRODUCER' in (process_folder/'stdout.log').read_text()
        assert m.advance(h.state['step']['path'],tmp_path/'no_receipt')['action']=='blocked_evidence'
        with pytest.raises(ValueError,match='partial'):m.run_build(h.state['step']['path'])

    def test_missing_p0_stays_materials_without_pin_or_fake_reference(tmp_path,monkeypatch):
        h=Harness(tmp_path,monkeypatch);bundle=m.load(h.h.p0_ref['path']);bundle['references'].pop('attention');p0=save(tmp_path/'incomplete_p0.json',bundle)
        result=m.prepare(h.h.origin_ref['path'],0,p0['path'],h.sources['path'],h.h.domain,7,tmp_path/'missing_materials')
        assert result['action']==f.m.Action.NEEDS_P0.value and result['details']['details']['missing']
        assert not (tmp_path/'missing_materials/artifact0').exists()

    def test_fitting576_catalog_emits_real_independent_solver_command_without_pin_or_predictions(tmp_path,monkeypatch):
        h=Harness(tmp_path,monkeypatch,catalog=True);h.build(mt=167,frame=0);h.query()
        state=m._validate_step(h.state['step']['path']);job=state['independent576_origin_job'];preview=m.load(h.h.origin_ref['path'])
        assert job['argv'][job['argv'].index('--mx-decode-v2-threads')+1]=='576'
        assert job['argv'][job['argv'].index('--mx-decode-v2-solve')+1]==preview['stage_catalog']['path']
        assert '--evaluate-configs' not in job['argv'] and set(job)=={'argv','timeout_seconds','purpose'}

    def test_failed_actual_cucc_receipt_and_forged_resource_report_stop(tmp_path,monkeypatch):
        h=Harness(tmp_path,monkeypatch);original=m.receipts.load_native
        def failed(path):
            ref,value=original(path);value['returncode']=1
            return ref,value
        monkeypatch.setattr(m.receipts,'load_native',failed)
        assert h.build()['action']=='blocked_evidence'

    def test_missing_native_materials_have_no_new_attempt_or_claimed_zero(tmp_path,monkeypatch):
        h=Harness(tmp_path,monkeypatch);out=tmp_path/'unready'
        result=m.advance(h.state['step']['path'],out)
        assert result['action']=='needs_actual_native_receipt' and not out.exists()
        assert result['missing'][0].endswith('.compile_receipt.json')

    def test_existing_outputs_and_saved_budget_are_never_overwritten(tmp_path,monkeypatch):
        h=Harness(tmp_path,monkeypatch)
        with pytest.raises(ValueError,match='preserved'):m.prepare(h.h.origin_ref['path'],0,h.h.p0_ref['path'],h.sources['path'],h.h.domain,7,tmp_path/'session')
        h.build();h.query();state=m.load(h.state['step']['path']);state['audit']['details']['remaining_512_reductions']=99
        bad=save(tmp_path/'budget_override.json',state)
        with pytest.raises(ValueError,match='budget'):m._validate_step(bad['path'])

    def test_existing_variant_api_accepts_plan_only_width_costs_without_fake_stage_failure(tmp_path,monkeypatch):
        h=f.Fixture(tmp_path,monkeypatch);builder=h.h.h
        parent=builder.artifact(mt=256,frame=108,query=False)
        action=f.m.old.next_width_diagnostic(h.origin,0,[parent]);assert action.action==f.m.old.Action.NEXT_WIDTH
        child=builder.artifact(action.data['next_pin'],ordinal=1)
        catalog=builder.catalog(action.data['authorization'])
        result=f.m.old.bind_descendant_costs(h.origin,0,[parent,child],[catalog])
        assert result.action==f.m.old.Action.COST_BOUND,result.as_dict()
        doc=m.load(catalog['path']);assert 'resource_selection' not in doc['rows'][0]['row']
        assert doc['rows'][0]['authorization']['kind']=='plan_resource_retry'
        doc['rows'][0]['row']['resource_selection']={'selected_resource_step':1}
        doc['rows'][0]['row_sha256']=f.m.legacy.digest(doc['rows'][0]['row'])
        bad=save(tmp_path/'fake_stage_selection.json',doc)
        with pytest.raises(ValueError,match='fabricated stage-bench'):f.m.legacy.VariantCatalog.load(bad['path'])
