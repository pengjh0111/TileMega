"""CPU synthetic consistency fixtures. No compiler, GPU, real query or approval."""
import copy
import ctypes as C
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
import types

import pytest

ROOT=Path(os.environ.get('TILEMEGA_TEST_ROOT',Path(__file__).resolve().parents[2]))
MODULE=Path(os.environ.get('TILEMEGA_THREAD_EXCEPTION_MODULE',ROOT/'python/tilemega/mx5_thread_exception.py'))
spec=importlib.util.spec_from_file_location('tilemega.mx5_thread_exception',MODULE)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
spec=importlib.util.spec_from_file_location('thread_exception_cpu_fixtures',ROOT/'test/python/test_mx5_plan_resources.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n');return m.record(path)


class Fixture:
    def __init__(self,root,monkeypatch,widths=None,depths=None):
        self.h=f.Harness(root,monkeypatch,widths=widths,depths=depths);self.root=root;self.count=0
        self.preview=copy.deepcopy(self.h.preview);self.preview['request']['threads']=512
        for candidate in self.preview['candidates']:
            candidate['pin']['mx_decode_v2']['threads']=512
            for rows in candidate['evidence'].values():
                for row in rows:row['key']['threads']=512
        self.preview_file=root/'native-preview.json';save(self.preview_file,self.preview)
        monkeypatch.setattr(m,'validate_preview',lambda path:copy.deepcopy(json.loads(Path(path).read_text())))
        self.origin=m.Origin.capture(m.record(self.preview_file));self.h.origin=self.origin
        runtime=root/'CPU_fake_runtime.so';runtime.write_bytes(b'CPU synthetic runtime identity')
        monkeypatch.setattr(m.occupancy,'RUNTIME',runtime)
        def tools(argv):
            if argv[0]=='nm':return '0000000000000001 W '+m.occupancy.STUB+'\n0000000000000002 T tm_plan_shared_bytes\n'
            return '<tm_plan_shared_bytes>:\n  2: b8 30 80 00 00 mov $0x8030,%eax\n  7: c3 ret\n\n'
        monkeypatch.setattr(m.occupancy,'run_text',tools)

    def pin(self,index=0,minimum=False):
        pin=copy.deepcopy(self.origin.candidate(index)['pin'])
        if minimum:
            c=pin['mx_decode_v2'];c.update(threads=576,attn_prefetch=0,purpose='resource_probe')
            for name in m.CLASSES:c[name]['prefetch']=c[name]['d'];c[name].pop('r_next')
        return pin

    def artifact(self,pin=None,*,index=0,ordinal=0,mt=220,frame=0,shared=0,resident=1,query=True,watchdog=0,extra=(),mir_extra='',mir_transform=None):
        pin=copy.deepcopy(pin or self.pin(index));c=pin['mx_decode_v2'];threads=c['threads'];candidate=self.origin.candidate(index)
        folder=self.root/('artifact%d'%self.count);self.count+=1;folder.mkdir();out=folder/'plan.so'
        out.write_bytes(f.elf_fixture()+str(self.count).encode())
        generated=Path(str(out)+'.cu')
        text='#include <tilemega/Codegen/tasks/mx/DecodeProgramV2.cuh>\nnamespace tm_mx_plan { struct Plan {\n'
        text+='static constexpr int kLayers=2,kThreads=%d,kGrid=104,kBatch=1,kCapacity=1088,kPastLo=64,kPastHi=1086;\n'%threads
        resolved=copy.deepcopy(c)
        for name,emitted in zip(m.CLASSES,('Qkv','O','GateUp','Down','Head')):
            item=c[name];key=candidate['evidence'][name][0]['key'];args='%d,%d,%d'%(item['tn'],item['wn'],item['d'])
            body='MxHead<'+args+','+str(item['prefetch'])+'>' if name=='head' else 'MxGemm<'+args+',MxOpV2::'+('kStore','kResidual','kSwiGLU')[key['op']]+','+('true' if key['norm'] else 'false')+',MxPart::'+item['part']+','+str(item['prefetch'])+'>'
            text+='using '+emitted+' = '+body+';\nusing '+emitted+'Shape = MxStaticShapeV2<%d,%d,kBatch,kThreads>;\n'%(key['n'],key['k'])
            if name=='down':text+='using DownLast = '+body+';\n'
            resolved[name].update(n=key['n'],k=key['k'],b=1,op='argmax' if name=='head' else ('kStore','kResidual','kSwiGLU')[key['op']],norm=bool(key.get('norm',1)),r_next=item.get('r_next',-1))
        text+='using Attn = MxAttentionV2<64,2,1,false,%d,kCapacity>;\nusing AttnShape = MxStaticAttentionShapeV2<kBatch,13,kThreads>;\n};}\n'%c['attn_prefetch']
        generated.write_text(text)
        log=Path(str(out)+'.ptxas.log');log.write_text(f.mir_fixture(mt,frame,shared).replace('_Zmx_decode_kernel_v2X',m.occupancy.KERNEL).replace('bb.0:\n','bb.0:\n  successors: %bb.1\nbb.1:\n  successors: %bb.1, %bb.2\n  LDG_B32 $r1\n  MMA_16X16X16BF16\n  ARRIVE 4096\nbb.2:\n').replace('# End machine code',mir_extra+'\n# End machine code'))
        if mir_transform is not None:log.write_text(mir_transform(log.read_text()))
        pinref=save(folder/'requested.pin.json',pin);resolvedref=save(Path(str(out)+'.mx5_pin.json'),dict(mx_decode_v2=resolved))
        snapshot=dict(root=str(self.h.source_root),head='CPU_SYNTHETIC',source_files=self.h.source_files,
           source_digest=m.digest(self.h.source_files),worktree_diff_sha256=m.record(self.h.patch)['sha256'],compiler=str(self.h.producer),compiler_sha256=m.record(self.h.producer)['sha256'])
        snapshotref=save(Path(str(out)+'.source.json'),snapshot)
        argv=[str(self.h.compiler),'-std=c++17','-O3','-Xcompiler','-resource-usage','-fno-strict-aliasing',
          '-DTILEMEGA_SYNC_V3=1','-DTILEMEGA_MX_STATIC_DECODE_V2=1','-DTILEMEGA_WATCHDOG='+str(watchdog),'-DTILEMEGA_TRACE_STAGE=0','-DTILEMEGA_MX_DIAG=0',
          '-mllvm','-print-after=stack-frame-layout','-shared','-Xcompiler=-fPIC','-x','cu','-I'+str(self.h.source_root/'include'),str(generated),'-L'+str(self.root/'cuda-lib'),'-lcudart','-o',str(out),*extra]
        command=folder/'command.txt';command.write_text(shlex.join(argv)+'\n')
        files=dict(pin=pinref,resolved_pin=resolvedref,generated_source=m.record(generated),command=m.record(command),source_snapshot=snapshotref,
          source_patch=m.record(self.h.patch),compiler=m.record(self.h.compiler),producer=m.record(self.h.producer),log=m.record(log),binary=m.record(out))
        receipt=dict(schema_version=1,kind='mx5_native_compile_receipt',argv=argv,returncode=0,wait_status=0,changed_inputs=[],files=files,
          same_compile_mir=files['log']['sha256'],binary_sha256=files['binary']['sha256'])
        for name,ref in files.items():receipt[name+'_sha256']=ref['sha256']
        receiptref=save(Path(str(out)+'.compile_receipt.json'),receipt)
        report=m.inspect(out,log,threads);resource_ref=save(Path(str(out)+'.resources.json'),report)
        query_ref=None
        if query:
            inputs=dict(kind='mx5_plan_occupancy_inputs',schema_version=1,threads=threads,context_kind='compile_receipt',binary=files['binary'],
                 generated_source=files['generated_source'],resources=resource_ref,log=files['log'],context=receiptref,source_snapshot=snapshotref)
            inputs_ref=save(folder/'query.inputs.json',inputs)
            validated=m.occupancy.validate_inputs(inputs,command=m.occupancy.run_text)
            class Runtime:
                def attributes(self,pointer,address):
                    attrs=C.cast(pointer,C.POINTER(m.occupancy.Attributes)).contents
                    attrs.numRegs=mt;attrs.localSizeBytes=frame;attrs.sharedSizeBytes=shared;attrs.maxThreadsPerBlock=threads;attrs.maxDynamicSharedSizeBytes=65536
                    return 0
                def set_shared(self,*args):return 0
                def occupancy(self,pointer,*args):C.cast(pointer,C.POINTER(C.c_int)).contents.value=resident;return 0
            class Library:
                def shared_bytes(self):return 32816
                def stub_address(self,*args):return C.c_void_p(0x1234)
            guard=dict(physical_gpu='7',lock_path='/tmp/tilemega_mx_gpu7.lock',owner=dict(pid=11,command='python -m tilemega.bench.gpu_guard CPU_SYNTHETIC',proc_locks_line='1: FLOCK ADVISORY WRITE 11 00:00:1 0 EOF'),environment=dict(TILEMEGA_DEVICE_INDEX='7',CUDA_VISIBLE_DEVICES='7',MACA_VISIBLE_DEVICES='7',TILEMEGA_GPU_LOCK='/tmp/tilemega_mx_gpu7.lock',TILEMEGA_GPU_LOCK_HELD='1'))
            row=m.occupancy.query(validated,Runtime(),Library(),guard)
            row.update(inputs_manifest=inputs_ref,runtime_library=m.record(m.occupancy.RUNTIME),helper_source=m.record(Path(m.occupancy.__file__)))
            query_ref=save(folder/'query.json',row)
        bundle=dict(kind='mx5_thread_exception_artifact',schema_version=1,origin_sha256=self.origin.sha256,candidate_id=index,ordinal=ordinal,
                    native_receipt=receiptref,resources=resource_ref,query=query_ref,stage_sources=self.h.sources)
        return save(folder/'artifact.json',bundle)

    def minimum(self,**kwargs):return self.artifact(self.pin(minimum=True),mt=168,frame=8,**kwargs)
    def audit(self,minimum,lineage,**kwargs):return m.audit_thread_exception(self.origin,0,minimum,lineage,**kwargs)
    def catalog(self,authorization):
        action=types.SimpleNamespace(details=m.legacy.FrozenJSON.freeze(dict(authorization=authorization)))
        path=self.h.add_variant(action,ns=20);doc=json.loads(path.read_text())
        for entry in doc['rows']:
            folder=Path(entry['row']['rounds'][0]['input']['path']).parent
            keys=save(folder/'keys.json',[entry['row']['key']]);processes=[]
            for sample in entry['row']['rounds']:
                number=sample['round'];processes.append(save(folder/('fresh%d.json'%number),dict(round=number,pid=100+number,returncode=0,argv=['CPU_FIXTURE_BENCH','--keys',keys['path'],'--out',sample['input']['path'],'--round',str(number),'--iterations',str(sample['measurement']['iterations'])])))
            journal=folder/'guard.jsonl';journal.write_text(json.dumps(dict(phase='after',exit_code=0))+'\n')
            entry.update(keys=keys,fresh_processes=processes,guard=dict(result=save(folder/'guard_result.json',dict(code=0)),journal=m.record(journal)))
        return save(path,doc)


def test_real_contract_missing_d7_stays_review_only(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);minimum=h.minimum();final=h.artifact()
    result=h.audit(minimum,[final])
    assert result.action==m.Action.NEEDS_D7,result.as_dict()
    assert not result.as_dict()['execution_authorized'] and result.data['prediction']['effective_predicted_tpot_ns']==h.origin.candidate(0)['predicted_tpot_ns']


def test_fitting_minimum_vetoes_512_even_without_final_artifact(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);minimum=h.artifact(h.pin(minimum=True),mt=167,frame=0)
    result=h.audit(minimum,[])
    assert result.action==m.Action.USE_576 and not result.as_dict()['execution_authorized']


def test_missing_costs_do_not_create_effective_prediction_and_real_chain_binds(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);minimum=h.minimum();parent=h.artifact(mt=256,frame=108,query=False)
    next_=m.next_width_diagnostic(h.origin,0,[parent]);assert next_.action==m.Action.NEXT_WIDTH,next_.as_dict()
    assert next_.data['authorization']['shared_class']=='qkv'
    child=h.artifact(next_.data['next_pin'],ordinal=1)
    result=h.audit(minimum,[parent,child]);assert result.action==m.Action.NEEDS_COSTS and result.data['prediction'] is None,result.as_dict()
    cost=m.bind_descendant_costs(h.origin,0,[parent,child],[h.catalog(next_.data['authorization'])])
    assert cost.action==m.Action.COST_BOUND and cost.data['prediction']['original_ranks']==[0]*6,cost.as_dict()
    assert cost.data['prediction']['origin_predicted_tpot_ns']==h.origin.candidate(0)['predicted_tpot_ns']


def amended(ref,change):
    path=Path(ref['path']);value=json.loads(path.read_text());change(value)
    candidate=path.with_name(path.stem+'.amended'+path.suffix)
    return save(candidate,value)


def complete_d7(h,minimum,final):
    mnode=m._artifact(minimum,h.origin,0,minimum=True,require_query=True)
    fnode=m._artifact(final,h.origin,0)
    diagnostic=h.artifact(h.pin(minimum=True),mt=233,frame=0,resident=0,extra=('-mllvm','--max-mtreg-number=255'))
    before=m._load(minimum);after=m._load(diagnostic)
    text='\n'.join((m.SPEC_DIFFERENCE,h.origin.sha256,mnode['files']['binary']['sha256'],fnode['files']['binary']['sha256'],
        'CPU synthetic before/after control only. Same source, normal floor168/frame8 versus forced233/frame0/resident0. No GPU observation was made. This is not a physical >227 lower-bound claim.'))
    report=h.root/'CPU_SYNTHETIC_D7.txt';report.write_text(text)
    doc=dict(kind='mx5_candidate_d7_review',schema_version=1,origin_sha256=h.origin.sha256,candidate_id=0,family_sha256=mnode['family'],
        spec_difference=m.SPEC_DIFFERENCE,minimum576_binary_sha256=mnode['files']['binary']['sha256'],final512_binary_sha256=fnode['files']['binary']['sha256'],
        assessment='normal_floor_576_still_spills_at_observed_168',not_claimed='required_registers_gt227_or_global_576_impossible',report=m.record(report),
        steps=[dict(kind='budget_diagnostic',analysis='CPU fixture controlled compile-option evidence, not execution approval',before_native=before['native_receipt'],after_native=after['native_receipt'],
            source_evidence=[dict(source_path='include/tilemega/Backend/Mx/Fixture.h',before=m.record(h.h.header),after=m.record(h.h.header))],
            after_resources=after['resources'],after_query=after['query'])])
    return save(h.root/'CPU_SYNTHETIC_D7.json',doc)


def test_fully_bound_synthetic_review_ready_is_still_not_execution_authority(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);minimum=h.minimum();final=h.artifact();d7=complete_d7(h,minimum,final)
    ready=h.audit(minimum,[final],d7=d7)
    assert ready.action==m.Action.READY,ready.as_dict()
    assert ready.data['execution_authorized'] is False and ready.as_dict()['execution_authorized'] is False
    audit=save(tmp_path/'CPU_SYNTHETIC_AUDIT.json',ready.as_dict())
    consumed=m.check_bound_exception(audit,h.origin,0,final,purpose='guarded_measurement_wd0')
    assert consumed.action==m.Action.READY and not consumed.as_dict()['execution_authorized'],consumed.as_dict()
    assert m.check_bound_exception(audit,h.origin,1,final,purpose='guarded_measurement_wd0').action==m.Action.BLOCKED
    assert m.check_bound_exception(audit,h.origin,0,final,purpose='serve_default').action==m.Action.BLOCKED
    assert m.check_bound_exception(audit,h.origin,0,final,purpose='guarded_watchdog_wd1').action==m.Action.BLOCKED


@pytest.mark.parametrize('field,value',[('query_status',None),('query_status',7),('occupancy_returncode',1),('resident_ctas',None),('iterations',1),('launches',1),('dynamic_shared_bytes',16),('execution_authorized',True)])
def test_unknown_or_mutated_query_never_becomes_zero_or_success(tmp_path,monkeypatch,field,value):
    h=Fixture(tmp_path,monkeypatch);minimum=h.minimum();final=h.artifact()
    def changed(bundle):bundle['query']=amended(bundle['query'],lambda x:x.update({field:value}))
    final=amended(final,changed)
    result=h.audit(minimum,[final]);assert result.action==m.Action.BLOCKED,result.as_dict()


@pytest.mark.parametrize('change',['candidate','ordinal','generated','source','report','compiler','override','r_next','attention'])
def test_artifact_identity_geometry_and_source_mismatches_stop(tmp_path,monkeypatch,change):
    h=Fixture(tmp_path,monkeypatch);minimum=h.minimum();pin=h.pin()
    if change=='r_next':pin['mx_decode_v2']['qkv']['r_next']+=1
    if change=='attention':pin['mx_decode_v2']['attn_prefetch']=0
    final=h.artifact(pin,extra=('-mllvm','--max-mtreg-number=255') if change=='override' else ())
    if change=='candidate':final=amended(final,lambda x:x.update(candidate_id=1))
    elif change=='ordinal':final=amended(final,lambda x:x.update(ordinal=2))
    elif change in ('generated','source','compiler'):
        native=m._load(m._load(final)['native_receipt']);field={'generated':'generated_source','source':'source_patch','compiler':'compiler'}[change]
        Path(native['files'][field]['path']).write_text('changed after actual recorded identity')
    elif change=='report':
        final=amended(final,lambda x:x.update(resources=amended(x['resources'],lambda r:r['kernel'].update(registers=1))))
    assert h.audit(minimum,[final]).action==m.Action.BLOCKED


def test_minimum_cannot_copy512_rnext_or_use_nonfloor_prefetch(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch)
    for name in ('fake_rnext','nonfloor'):
        pin=h.pin(minimum=True)
        if name=='fake_rnext':pin['mx_decode_v2']['qkv']['r_next']=150
        else:pin['mx_decode_v2']['qkv']['prefetch']+=1
        minimum=h.artifact(pin,mt=168,frame=8)
        assert h.audit(minimum,[]).action==m.Action.BLOCKED


def test_no_skip_no_wrong_largest_no_geometry_change_and_no_fourth_retry(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);parent=h.artifact(mt=256,frame=108,query=False);lineage=[parent]
    for ordinal in (1,2,3):
        action=m.next_width_diagnostic(h.origin,0,lineage);assert action.action==m.Action.NEXT_WIDTH,action.as_dict()
        if ordinal==1:
            wrong=copy.deepcopy(action.data['next_pin']);wrong['mx_decode_v2']['qkv']['prefetch']=16;wrong['mx_decode_v2']['head']['prefetch']=1
            assert m.next_width_diagnostic(h.origin,0,[parent,h.artifact(wrong,ordinal=1,mt=256,frame=4,query=False)]).action==m.Action.BLOCKED
        lineage.append(h.artifact(action.data['next_pin'],ordinal=ordinal,mt=256,frame=4,query=False))
    assert m.next_width_diagnostic(h.origin,0,lineage).action==m.Action.BLOCKED
    assert m.next_width_diagnostic(h.origin,0,lineage[1:]).action==m.Action.BLOCKED


def test_all_floor_failure_stays_a_defect_not_another_candidate(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch,widths={n:1 for n in m.CLASSES});minimum=h.minimum()
    failed=h.artifact(mt=256,frame=4,query=False)
    result=h.audit(minimum,[failed]);assert result.action==m.Action.BLOCKED and 'all Pw equal D' in result.reason
    assert m.next_width_diagnostic(h.origin,0,[failed]).action==m.Action.BLOCKED


def test_fitting_parent_and_changed_epoch_cannot_authorize_width_diagnostic(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);fit=h.artifact()
    assert m.next_width_diagnostic(h.origin,0,[fit]).action==m.Action.BLOCKED
    parent=h.artifact(mt=256,frame=4,query=False);next_=m.next_width_diagnostic(h.origin,0,[parent])
    # Wrong geometry does not become a new child even with genuine child resource files.
    pin=next_.data['next_pin'];pin['mx_decode_v2']['qkv']['d']=2
    child=h.artifact(pin,ordinal=1,mt=256,frame=4,query=False)
    assert m.next_width_diagnostic(h.origin,0,[parent,child]).action==m.Action.BLOCKED


def test_bare_boolean_d7_and_mismatched_review_files_never_qualify(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);minimum=h.minimum();final=h.artifact()
    fake=save(tmp_path/'confirmed.json',dict(confirmed=True))
    assert h.audit(minimum,[final],d7=fake).action==m.Action.BLOCKED
    d7=complete_d7(h,minimum,final)
    wrong=amended(d7,lambda x:x.update(candidate_id=1))
    assert h.audit(minimum,[final],d7=wrong).action==m.Action.BLOCKED
    wrong=amended(d7,lambda x:x.update(not_claimed='required_registers_gt227'))
    assert h.audit(minimum,[final],d7=wrong).action==m.Action.BLOCKED


def test_rank_prediction_and_post_audit_artifact_changes_are_detected(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);minimum=h.minimum();final=h.artifact();d7=complete_d7(h,minimum,final)
    ready=h.audit(minimum,[final],d7=d7);assert ready.action==m.Action.READY
    audit=save(tmp_path/'ready.json',ready.as_dict())
    value=json.loads(h.preview_file.read_text());value['candidates'][0]['ranks'][0]=1;value['candidates'][0]['predicted_tpot_ns']+=1;save(h.preview_file,value)
    assert m.check_bound_exception(audit,h.origin,0,final,purpose='guarded_measurement_wd0').action==m.Action.BLOCKED


@pytest.mark.parametrize('problem',['row_hash','wrong_key','fake_bench_failure','same_pid','failed_process','wrong_keys','guard_failed'])
def test_changed_width_costs_require_exact_source_rows_and_three_fresh_processes(tmp_path,monkeypatch,problem):
    h=Fixture(tmp_path,monkeypatch);parent=h.artifact(mt=256,frame=8,query=False)
    action=m.next_width_diagnostic(h.origin,0,[parent]);child=h.artifact(action.data['next_pin'],ordinal=1)
    catalog=h.catalog(action.data['authorization']);doc=m._load(catalog);entry=doc['rows'][0]
    if problem=='row_hash':entry['row_sha256']='0'*64
    elif problem=='wrong_key':entry['row']['key']['prefetch']=4
    elif problem=='fake_bench_failure':entry['row']['resource_selection']={'selected_resource_step':1}
    elif problem=='same_pid':
        original=m._load(entry['fresh_processes'][0])['pid']
        entry['fresh_processes'][1]=amended(entry['fresh_processes'][1],lambda x:x.update(pid=original))
    elif problem=='failed_process':entry['fresh_processes'][1]=amended(entry['fresh_processes'][1],lambda x:x.update(returncode=1))
    elif problem=='wrong_keys':entry['keys']=save(tmp_path/'wrong-keys.json',[])
    else:entry['guard']['result']=save(tmp_path/'guard-failure.json',dict(code=1))
    catalog=save(tmp_path/'bad-catalog.json',doc)
    result=m.bind_descendant_costs(h.origin,0,[parent,child],[catalog])
    assert result.action==m.Action.BLOCKED,result.as_dict()
    assert 'prediction' not in result.data


@pytest.mark.parametrize('body',['  LDX_B128 $r1\n','  STX_B128 $r1\n','  CALL external\n','  LDG_B32 :: (load from %ir.x, addrspace 0)\n',
    '  successors: %bb.3\nbb.3:\n  successors: %bb.3, %bb.4\n  LDG_B32\n  MMA_16X16X16BF16\n  ARRIVE 4160\nbb.4:\n  RET\n'])
def test_final512_mir_hard_gates_do_not_pass_bad_generic_calls_or_rings(tmp_path,monkeypatch,body):
    h=Fixture(tmp_path,monkeypatch);minimum=h.minimum();bad=h.artifact(mir_extra=body)
    result=h.audit(minimum,[bad]);assert result.action==m.Action.BLOCKED,result.as_dict()


def test_barrier_wait_outside_an_mma_ring_is_not_a_false_ring_failure(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);minimum=h.minimum();final=h.artifact(mir_extra='  ARRIVE 4160\n')
    assert h.audit(minimum,[final]).action==m.Action.NEEDS_D7


def test_watchdog_role_needs_its_own_exact_query_and_resource_proof(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);minimum=h.minimum();final=h.artifact();wd=h.artifact(watchdog=1)
    result=h.audit(minimum,[final],watchdog512=wd);assert result.action==m.Action.NEEDS_D7,result.as_dict()
    assert h.audit(minimum,[final],watchdog512=final).action==m.Action.BLOCKED
    wd=amended(wd,lambda x:x.update(ordinal=1))
    assert h.audit(minimum,[final],watchdog512=wd).action==m.Action.BLOCKED


def test_original_prediction_cannot_be_overridden_in_an_artifact_bundle(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);minimum=h.minimum();final=h.artifact()
    final=amended(final,lambda x:x.update(predicted_tpot_ns=1,ranks=[1]*6))
    assert h.audit(minimum,[final]).action==m.Action.BLOCKED


def test_legacy_policy_remains_identical(tmp_path,monkeypatch):
    before=copy.deepcopy(m.legacy.POLICY);h=Fixture(tmp_path,monkeypatch)
    h.audit(h.minimum(),[h.artifact()])
    assert m.legacy.POLICY==before and m.legacy.POLICY['threads']=='576_only'


def test_malformed_evidence_returns_typed_blocked_instead_of_parser_tracebacks(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);minimum=h.minimum();final=h.artifact()
    malformed=save(tmp_path/'not-a-review.json',[])
    assert h.audit(minimum,[final],d7=malformed).action==m.Action.BLOCKED
    assert m.check_bound_exception(malformed,h.origin,0,final,purpose='guarded_measurement_wd0').action==m.Action.BLOCKED
    assert m.next_width_diagnostic(h.origin,0,[malformed]).action==m.Action.BLOCKED
    assert m.bind_descendant_costs(h.origin,0,[malformed]).action==m.Action.BLOCKED


def test_missing_zero_query_and_forged_query_thread_do_not_qualify(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);minimum=h.minimum();no_query=h.artifact(query=False)
    assert h.audit(minimum,[no_query]).action==m.Action.BLOCKED
    final=h.artifact();final=amended(final,lambda x:x.update(query=amended(x['query'],lambda q:q.update(threads=576))))
    assert h.audit(minimum,[final]).action==m.Action.BLOCKED


@pytest.mark.parametrize('change',[
    lambda text:text.replace('MMA_16X16X16BF16','FADD_F32'),
    lambda text:text.replace('  successors: %bb.1\n','  successors: %bb.2\n',1),
])
def test_complete_dense_zero_recognized_mma_loop_is_missing_evidence(tmp_path,monkeypatch,change):
    h=Fixture(tmp_path,monkeypatch);minimum=h.minimum()
    final=h.artifact(mir_transform=change)
    result=h.audit(minimum,[final])
    assert result.action==m.Action.BLOCKED,result.as_dict()
    assert 'lacks actual natural MMA-loop evidence' in result.reason


@pytest.mark.parametrize('change,message',[
    (lambda text:text[:text.index('# End machine code')], 'truncated/incomplete'),
    (lambda text:text.replace('# End machine code for function '+m.occupancy.KERNEL+'.',
        '# End machine code for function wrong_kernel.'), 'mismatched'),
    (lambda text:text.replace('# End machine code for function '+m.occupancy.KERNEL+'.',
        '# Machine code for function next_kernel:\nbb.0:\n# End machine code for function next_kernel.'), 'before another header'),
    (lambda text:text+'# Machine code for function incomplete_host:\nbb.0:\n  RET\n', 'truncated/incomplete'),
    (lambda text:text+'# Machine code for function incomplete_host', 'boundary marker'),
    (lambda text:text.replace('# End machine code for function '+m.occupancy.KERNEL+'.', '# End machine code for function'), 'boundary marker'),
    (lambda text:text.replace('bb.2:', 'bb.1:'), 'duplicate'),
    (lambda text:text.replace('successors: %bb.1, %bb.2','successors: %bb.1, %bb.999'), 'successor coverage'),
    (lambda text:text.replace('bb.0:', 'not_a_block:').replace('bb.1:', 'not_a_block:').replace('bb.2:', 'not_a_block:'), 'missing/duplicate'),
])
def test_incomplete_or_overwritten_mir_cannot_be_clean(tmp_path,monkeypatch,change,message):
    h=Fixture(tmp_path,monkeypatch);artifact=h.artifact()
    receipt=m._load(m._load(artifact)['native_receipt']);log=m.check_file(receipt['files']['log'])
    broken=tmp_path/'incomplete.mir';broken.write_text(change(log.read_text()))
    with pytest.raises(ValueError,match=message):m._mir(m.record(broken),m.occupancy.KERNEL)


@pytest.mark.parametrize('failure',['incomplete_report','parser_error'])
def test_trusted_parser_failure_or_partial_function_report_does_not_mean_clean(tmp_path,monkeypatch,failure):
    h=Fixture(tmp_path,monkeypatch);artifact=h.artifact()
    receipt=m._load(m._load(artifact)['native_receipt'])
    def partial(argv,**kwargs):
        Path(argv[argv.index('--out')+1]).write_text(json.dumps(dict(functions=[],failures=[])))
        return types.SimpleNamespace(returncode=0 if failure=='incomplete_report' else 2,stderr='CPU synthetic parser failure')
    monkeypatch.setattr(m.subprocess,'run',partial)
    with pytest.raises(ValueError,match='incomplete function coverage|trusted MIR parser failed'):
        m._mir(receipt['files']['log'],m.occupancy.KERNEL)
