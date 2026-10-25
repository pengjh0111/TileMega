"""CPU synthetic fixtures only: no real measured table, GPU process or build."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import struct
import sys
import tarfile

import pytest

ROOT = Path(os.environ.get('TILEMEGA_TEST_ROOT', Path(__file__).resolve().parents[2]))
MODULE = Path(os.environ.get('TILEMEGA_MX5_PLAN_RESOURCES_MODULE', ROOT / 'python/tilemega/mx5_plan_resources.py'))
spec = importlib.util.spec_from_file_location('tilemega.mx5_plan_resources', MODULE)
m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n'); return m.record(path)


def elf_fixture():
    names = b'\0.text\0.shstrtab\0'; code = b'CPU fixture device bytes, not executable'
    header = bytearray(64); header[:6] = b'\x7fELF\x02\x01'
    struct.pack_into('<Q', header, 40, 64 + len(code) + len(names)); struct.pack_into('<HHH', header, 58, 64, 3, 2)
    sections = bytes(64) + struct.pack('<IIQQQQIIQQ', 1,1,6,0,64,len(code),0,0,16,0)
    sections += struct.pack('<IIQQQQIIQQ', 7,3,0,0,64+len(code),len(names),0,0,1,0)
    elf = bytes(header)+code+names+sections
    magic=b'__CLANG_OFFLOAD_BUNDLE__'; target=b'maca-mxc-metaxgpu--xcore1002'
    offset=len(magic)+8+24+len(target)
    return b'CPU fixture host bytes'+magic+struct.pack('<QQQQ',1,offset,len(elf),len(target))+target+elf


def mir_fixture(mt=228, stack=0, shared=0):
    return f'''Function properties for _Zmx_decode_kernel_v2X
    {stack} bytes stack frame
Used {mt} MTregisters, 30 STregisters, {shared} bytes shared mem
# Machine code for function _Zmx_decode_kernel_v2X:
bb.0:
  renamable $r0 = LDG_B32 $r2, 0
  ARRIVE 4096
# End machine code for function _Zmx_decode_kernel_v2X.
'''


class Harness:
    def __init__(self, tmp_path, monkeypatch, widths=None, depths=None):
        self.root=tmp_path; self.rows=[]; self.counter=0
        self.source_root=tmp_path/'source'; self.source_root.mkdir()
        self.header=self.source_root/'include/tilemega/Backend/Mx/Fixture.h'; self.header.parent.mkdir(parents=True)
        self.header.write_text('// CPU synthetic shared body fixture\n')
        files={'include/tilemega/Backend/Mx/Fixture.h':m.record(self.header)['sha256']}
        archive=tmp_path/'stage-sources.tar.gz'; digest=hashlib.sha256()
        entries={'include/tilemega/Backend/Mx/Fixture.h':self.header.read_bytes(),
                 'experiments/mx5/stage_bench.cu':b'// CPU synthetic bench fixture\n'}
        with tarfile.open(archive,'w:gz') as output:
            for name,data in sorted(entries.items()):
                digest.update(name.encode()+b'\0'+data+b'\0'); item=tarfile.TarInfo(name); item.size=len(data)
                output.addfile(item,io.BytesIO(data))
        self.stage_sha=digest.hexdigest(); self.sources=[dict(source_sha256=self.stage_sha,archive=m.record(archive))]
        self.source_files=files
        self.compiler=tmp_path/'synthetic-device-compiler'; self.compiler.write_text('CPU fixture only\n')
        self.producer=tmp_path/'synthetic-native-producer'; self.producer.write_text('CPU fixture producer\n')
        self.patch=tmp_path/'source.patch'; self.patch.write_text('CPU fixture source archive patch\n')
        widths=widths or dict(qkv=16,o=16,gate_up=8,down=4,head=2)
        depths=depths or {name:1 for name in m.CLASSES}
        configs={name:dict(tn=32 if name=='gate_up' else 16,wn=4 if name=='head' else 1,d=depths[name],
            part='kTileRange',prefetch=widths[name],r_next=150) for name in m.CLASSES}
        pin=dict(mx_decode_v2=dict(**configs,threads=576,b_min=1,attn_prefetch=2,purpose='manual',
                                 diagnostic=True,final=False,solver_selected=False,default_enabled=False))
        stages=dict(qkv=100,o=200,gate_up=300,down=400,head=600,attention=500)
        evidence={}
        for name in m.CLASSES:
            c=configs[name]; kind='head' if name=='head' else 'gemm'
            key=dict(n=8192 if name=='head' else 4096,k=4096,b=1,tn=c['tn'],wn=c['wn'],depth=c['d'],threads=576,prefetch=c['prefetch'])
            if kind=='gemm':key.update(op=2 if name=='gate_up' else 1 if name in ('o','down') else 0,
                                      norm=int(name in ('qkv','gate_up')),part=1)
            evidence[name]=[dict(kind=kind,key=key,r_next=150,provenance=dict(source_sha256=self.stage_sha),
                                 test_fixture='CPU synthetic origin row, not qualified measured data')]
        evidence['attention']=[dict(kind='attention',key=dict(length=length,dim=64,q=2,hkv=13,qk_norm=0),provenance=dict(source_sha256=self.stage_sha))
                               for length in (65,256,512,768,1088)]
        candidates=[]
        for i in range(4):
            times={**stages,'qkv':100+i}
            candidates.append(dict(index=i,pin=copy.deepcopy(pin),predicted_tpot_ns=times['head']+2*sum(times[n] for n in m.STAGES if n!='head'),
                                   predicted_stage_ns=times,ranks=[i,0,0,0,0,0],evidence=copy.deepcopy(evidence)))
        self.preview=dict(synthetic_fixture=True,producer=m.record(self.producer),candidates=candidates,
                          request=dict(batch=1,threads=576,capacity=1088,past_lo=64,past_hi=1086))
        self.preview_path=tmp_path/'synthetic-native-preview.json'; save(self.preview_path,self.preview)
        monkeypatch.setattr(m,'validate_preview',lambda path:copy.deepcopy(self.preview))
        self.origin=m.OriginCertificate.capture(self.preview_path)

    def start(self,index=0):return m.PlanLedger.start(self.origin,index)

    def attempt(self,ledger,mt=228,stack=0,shared=0,occupancy=1,query_status=0,compiler_rc=0,extra_flag=None,mutate=None,binary=True):
        directory=self.root/('attempt%d'%self.counter);self.counter+=1;directory.mkdir()
        pin=save(directory/'pin.json',ledger.pin.data)
        generated=directory/'plan.so.cu'
        text='// CPU synthetic generated source fixture; never compiled\nstatic constexpr int kLayers=2,kThreads=576,kGrid=104,kBatch=1,kCapacity=1088,kPastLo=64,kPastHi=1086;\n'
        config=ledger.pin.data['mx_decode_v2']
        for name,emitted in zip(m.CLASSES,('Qkv','O','GateUp','Down','Head')):
            c=config[name];key=self.origin.candidate(ledger.candidate_id)['evidence'][name][0]['key']
            args='%d,%d,%d'%(c['tn'],c['wn'],c['d'])
            if name=='head':body='MxHead<'+args+','+str(c['prefetch'])+'>'
            else:body='MxGemm<'+args+',MxOpV2::'+('kStore','kResidual','kSwiGLU')[key['op']]+','+('true' if key['norm'] else 'false')+',MxPart::'+c['part']+','+str(c['prefetch'])+'>'
            text+='using '+emitted+' = '+body+';\nusing '+emitted+'Shape = MxStaticShapeV2<%d,%d,kBatch,kThreads>;\n'%(key['n'],key['k'])
            if name=='down':text+='using DownLast = '+body+';\n'
        text+='using Attn = MxAttentionV2<64,2,1,false,2,kCapacity>;\nusing AttnShape = MxStaticAttentionShapeV2<kBatch,13,kThreads>;\n'
        generated.write_text(text)
        so=directory/'plan.so'
        if binary:so.write_bytes(elf_fixture())
        log=directory/'plan.so.ptxas.log';log.write_text(mir_fixture(mt,stack,shared))
        argv=[str(self.compiler),'-std=c++17','-mllvm','-print-after=stack-frame-layout',
              '-DTILEMEGA_MX_STATIC_DECODE_V2=1','-DTILEMEGA_WATCHDOG=0','-DTILEMEGA_TRACE_STAGE=0','-DTILEMEGA_MX_DIAG=0']
        if extra_flag:argv.append(extra_flag)
        argv += [str(generated),'-o',str(so)]
        command=directory/'build_command.txt';command.write_text(shlex.join(argv)+'\n')
        snapshot=dict(root=str(self.source_root),head='CPU_FIXTURE',source_files=self.source_files,source_digest=m.digest(self.source_files),
                      worktree_diff_sha256=m.record(self.patch)['sha256'],compiler=str(self.producer),compiler_sha256=m.record(self.producer)['sha256'])
        snapshot_record=save(directory/'source.json',snapshot)
        value=dict(schema_version=1,kind='mx5_plan_attempt',test_fixture='CPU synthetic files, no compiler/device invocation',
            origin_sha256=self.origin.sha256,candidate_id=ledger.candidate_id,ordinal=len(ledger.attempts),pin=pin,
            resolved_pin=save(directory/'resolved_pin.json',ledger.pin.data),generated_source=m.record(generated),command=m.record(command),source_snapshot=snapshot_record,
            source_patch=m.record(self.patch),compiler=m.record(self.compiler),producer=m.record(self.producer),log=m.record(log),
            binary=m.record(so) if binary else None,resources=None,occupancy=None,stage_sources=self.sources)
        if binary:
            value['resources']=save(directory/'resources.json',m.inspect(so,log,576))
        if occupancy is not None and binary:
            value['occupancy']=save(directory/'occupancy.json',dict(kind='mx5_plan_occupancy',binary_sha256=value['binary']['sha256'],
                                       threads=576,query_status=query_status,resident_ctas=occupancy))
        receipt=dict(argv=argv,returncode=compiler_rc,same_compile_mir=value['log']['sha256'],
                     binary_sha256=value['binary']['sha256'] if binary else None)
        for name in ('pin','resolved_pin','generated_source','source_snapshot','compiler','producer','log'):
            receipt[name+'_sha256']=value[name]['sha256']
        value['compile_receipt']=save(directory/'compile_receipt.json',receipt)
        value['program_source_epoch']=m.program_epoch(value)
        if mutate:mutate(value,directory)
        path=directory/'attempt.json';save(path,value)
        return path

    def add_variant(self,action,ns=20,change=None):
        details=action.details.data;auth=details['authorization'];name=auth['shared_class']
        base=self.origin.candidate(auth['candidate_id'])['evidence'][name][0]
        key=copy.deepcopy(base['key']);key['prefetch']=auth['to_pw'];kind=base['kind']
        directory=self.root/('variant%d'%len(self.rows));directory.mkdir()
        bench=directory/'synthetic-bench.so';bench.write_bytes(b'CPU SYNTHETIC BENCH FIXTURE')
        ref=directory/'synthetic-reference.so';ref.write_bytes(b'CPU SYNTHETIC P0 FIXTURE')
        def provenance(binary):return dict(executor='mx_static_stage_bench',implementation='mx5_'+kind+'_v2',
            source_sha256=self.stage_sha,binary_sha256=m.record(binary)['sha256'],base_head='CPU_SYNTHETIC_FIXTURE')
        resources=dict(mtreg=200,local_bytes=0,static_smem_bytes=0,resident_ctas=1)
        reference=dict(kind=kind,key={**key,'prefetch':0},resources={**resources,'mtreg':150},provenance=provenance(ref))
        rounds=[]
        for number in (1,2,3):
            measurement=dict(key={**key,'library':str(bench)},round=number,median_ns=ns,p90_ns=ns+5,iterations=3,copies=2,
                             active_working_set_bytes=268435456,**provenance(bench),**resources)
            path=directory/('process%d.jsonl'%number);path.write_text(json.dumps(measurement)+'\n')
            rounds.append(dict(round=number,input={**m.record(path),'line':1},measurement=measurement))
        row=dict(kind=kind,key=key,median_ns=ns,p90_ns=ns+5,processes=3,spread=0,resources=resources,provenance=provenance(bench),
                 rounds=rounds,r_next=150,r_next_reference=reference,test_fixture='SYNTHETIC CPU TIMES ONLY; NEVER A REAL TABLE')
        entry=dict(row=row,row_sha256=m.digest(row),authorization=auth)
        if change:change(entry)
        self.rows.append(entry)
        bundle=dict(schema_version=1,kind='mx5_decode_stage_variants',qualification_scope='plan_bound_measured_widths',
            test_fixture='CPU UNIT FIXTURE, NOT AN EXPERIMENT RESULT',qualification_tool=m.record(ROOT/'experiments/mx5/stage_table.py'),
            sources=self.sources,binaries=[m.record(bench),m.record(ref)],rows=copy.deepcopy(self.rows))
        # Each new immutable catalog retains the earlier exact binary proofs.
        if len(self.rows)>1:
            previous=json.loads((self.root/('catalog%d.json'%(len(self.rows)-1))).read_text())
            bundle['binaries'] += previous['binaries']
        path=self.root/('catalog%d.json'%len(self.rows));save(path,bundle)
        return path

    def reduce(self,ledger,ns=20):
        requested=m.next_plan_retry(ledger);assert requested.kind==m.ActionKind.NEEDS_STAGE_MEASUREMENTS
        catalog=m.VariantCatalog.load(self.add_variant(requested,ns))
        authorized=m.next_plan_retry(ledger,catalog)
        assert authorized.kind==m.ActionKind.BUILD_REDUCED_PLAN, authorized.details.data
        return ledger.commit_retry(authorized,catalog),authorized,catalog


def test_largest_numeric_reducible_ties_shared_edges_and_immutable_origin(tmp_path,monkeypatch):
    h=Harness(tmp_path,monkeypatch);before=h.origin.frozen.encoded;ledger=h.start()
    ledger=ledger.append_attempt(h.attempt(ledger));first=m.next_plan_retry(ledger)
    assert first.kind==m.ActionKind.NEEDS_STAGE_MEASUREMENTS
    data=first.details.data;assert data['authorization']['shared_class']=='qkv' and data['authorization']['to_pw']==8
    assert len(data['authorization']['affected_edges'])==3
    assert data['missing'][0]['key']['prefetch']==8 and data['missing'][0]['required_fresh_processes']==3
    assert data['prediction'] is None
    repaired,action,_=h.reduce(ledger)
    prediction=action.details.data['prediction']
    assert prediction['origin_predicted_tpot_ns']==3600 and prediction['effective_predicted_tpot_ns']==3440
    assert prediction['selection_anchor']==0 and 'not globally exact' in prediction['ranking_claim']
    repaired=repaired.append_attempt(h.attempt(repaired));second=m.next_plan_retry(repaired)
    assert second.details.data['authorization']['shared_class']=='o'
    assert h.origin.frozen.encoded==before and ledger.pin.data['mx_decode_v2']['qkv']['prefetch']==16
    changed=h.origin.data;changed['preview']['candidates'][0]['pin']['mx_decode_v2']['qkv']['prefetch']=1
    assert h.origin.candidate(0)['pin']['mx_decode_v2']['qkv']['prefetch']==16


def test_floor_D_and_largest_reducible_skip_a_larger_coordinate_at_floor(tmp_path,monkeypatch):
    h=Harness(tmp_path,monkeypatch,widths=dict(qkv=3,o=1,gate_up=1,down=1,head=4),depths=dict(qkv=1,o=1,gate_up=1,down=1,head=4))
    ledger=h.start();ledger=ledger.append_attempt(h.attempt(ledger));a=m.next_plan_retry(ledger)
    assert a.details.data['authorization']['shared_class']=='qkv' and a.details.data['authorization']['to_pw']==1
    sub=tmp_path/'floor';sub.mkdir();h=Harness(sub,monkeypatch,widths=dict(qkv=5,o=1,gate_up=1,down=1,head=1),depths=dict(qkv=3,o=1,gate_up=1,down=1,head=1))
    ledger=h.start();ledger=ledger.append_attempt(h.attempt(ledger))
    assert m.next_plan_retry(ledger).details.data['authorization']['to_pw']==3


def test_three_reductions_total_not_three_per_class(tmp_path,monkeypatch):
    h=Harness(tmp_path,monkeypatch);ledger=h.start()
    for expected in ('qkv','o','qkv'):
        ledger=ledger.append_attempt(h.attempt(ledger));ledger,a,_=h.reduce(ledger)
        assert a.details.data['authorization']['shared_class']==expected
    ledger=ledger.append_attempt(h.attempt(ledger));a=m.next_plan_retry(ledger)
    assert a.kind==m.ActionKind.BLOCKED and 'three whole-plan reductions exhausted' in a.reason
    assert len(ledger.authorizations)==3 and len(ledger.attempts)==4


@pytest.mark.parametrize('kwargs,fragment',[
    ({'shared':16},'static shared memory'),({'mt':0},'invalid integer actual MTreg'),({'compiler_rc':1},'device compiler failed'),
    ({'binary':False},'lacks binary/resource'),({'occupancy':0},'occupancy query'),
    ({'query_status':1},'occupancy query'),({'mt':220,'occupancy':None},'occupancy metadata'),
])
def test_only_actual_register_or_stack_failures_authorize_retry(tmp_path,monkeypatch,kwargs,fragment):
    h=Harness(tmp_path,monkeypatch);ledger=h.start();ledger=ledger.append_attempt(h.attempt(ledger,**kwargs))
    action=m.next_plan_retry(ledger);assert action.kind==m.ActionKind.BLOCKED and fragment in action.reason


def test_resource_fit_and_all_floor_D7_never_grants_512(tmp_path,monkeypatch):
    h=Harness(tmp_path,monkeypatch,widths={n:1 for n in m.CLASSES});ledger=h.start()
    fit=ledger.append_attempt(h.attempt(ledger,mt=220));action=m.next_plan_retry(fit)
    assert action.kind==m.ActionKind.RESOURCE_FIT and action.details.data['effective_prediction']['effective_predicted_tpot_ns']==3600
    failed=ledger.append_attempt(h.attempt(ledger,mt=228,mutate=lambda value,d:value.update(d7=dict(confirmed=True))))
    action=m.next_plan_retry(failed)
    assert action.kind==m.ActionKind.NEEDS_D7 and action.details.data['execution_512_allowed'] is False
    stack=ledger.append_attempt(h.attempt(ledger,mt=227,stack=4));action=m.next_plan_retry(stack)
    assert action.kind==m.ActionKind.BLOCKED and 'stack-only' in action.reason


def test_epoch_drift_and_changed_actual_sources_stop_the_chain(tmp_path,monkeypatch):
    h=Harness(tmp_path,monkeypatch);ledger=h.start();ledger=ledger.append_attempt(h.attempt(ledger));ledger,_,_=h.reduce(ledger)
    changed=ledger.append_attempt(h.attempt(ledger,extra_flag='-DOTHER_ALLOCATION_MODE=1'))
    assert m.next_plan_retry(changed).kind==m.ActionKind.BLOCKED and 'epoch changed' in m.next_plan_retry(changed).reason
    ledger=ledger.append_attempt(h.attempt(ledger))
    h.header.write_text('// changed actual source\n')
    assert m.next_plan_retry(ledger).kind==m.ActionKind.BLOCKED and 'immutable input changed' in m.next_plan_retry(ledger).reason


def test_attempt_receipt_and_same_compile_log_hashes_cannot_be_rewritten(tmp_path,monkeypatch):
    h=Harness(tmp_path,monkeypatch);ledger=h.start();path=h.attempt(ledger);ledger=ledger.append_attempt(path)
    raw=json.loads(path.read_text());Path(raw['log']['path']).write_text(mir_fixture(mt=220))
    action=m.next_plan_retry(ledger);assert action.kind==m.ActionKind.BLOCKED and 'immutable input changed' in action.reason


@pytest.mark.parametrize('mode,fragment',[
    ('hash','exact row hash'),('authorization','distinct plan_resource_retry'),
    ('stage_failure','fabricated stage-bench'),('rounds','three fresh measured'),
])
def test_exact_row_hash_three_fresh_rounds_and_distinct_program_authorization(tmp_path,monkeypatch,mode,fragment):
    h=Harness(tmp_path,monkeypatch);ledger=h.start();ledger=ledger.append_attempt(h.attempt(ledger));requested=m.next_plan_retry(ledger)
    def mutate(entry):
        if mode=='hash':entry['row_sha256']='0'*64
        if mode=='authorization':entry['authorization']['kind']='stage_resource_retry'
        if mode=='stage_failure':
            entry['row']['resource_selection']=dict(selected_resource_step=1,resource_attempts=[dict(fits=False),dict(fits=True)])
            entry['row_sha256']=m.digest(entry['row'])
        if mode=='rounds':
            entry['row']['rounds'].pop();entry['row_sha256']=m.digest(entry['row'])
    path=h.add_variant(requested,change=mutate)
    with pytest.raises(m.ContractError,match=fragment):m.VariantCatalog.load(path)


def test_wrong_width_or_authorization_never_produces_an_effective_prediction(tmp_path,monkeypatch):
    h=Harness(tmp_path,monkeypatch);ledger=h.start();ledger=ledger.append_attempt(h.attempt(ledger));requested=m.next_plan_retry(ledger)
    def other_plan(entry):entry['authorization']['candidate_id']=1
    catalog=m.VariantCatalog.load(h.add_variant(requested,change=other_plan));a=m.next_plan_retry(ledger,catalog)
    assert a.kind==m.ActionKind.NEEDS_STAGE_MEASUREMENTS and a.details.data['prediction'] is None
    assert a.details.data['missing'][0]['key']['prefetch']==8


def test_raw_process_rows_and_catalog_are_immutable(tmp_path,monkeypatch):
    h=Harness(tmp_path,monkeypatch);ledger=h.start();ledger=ledger.append_attempt(h.attempt(ledger));requested=m.next_plan_retry(ledger)
    path=h.add_variant(requested);catalog=m.VariantCatalog.load(path)
    row=catalog.frozen.data['document']['rows'][0]['row'];Path(row['rounds'][0]['input']['path']).write_text('{}\n')
    a=m.next_plan_retry(ledger,catalog);assert a.kind==m.ActionKind.BLOCKED and 'immutable input changed' in a.reason


def test_candidate_zero_selection_anchor_survives_repricing_other_candidates(tmp_path,monkeypatch):
    h=Harness(tmp_path,monkeypatch);ledger=h.start(2);ledger=ledger.append_attempt(h.attempt(ledger));ledger,a,_=h.reduce(ledger,ns=10000)
    p=a.details.data['prediction'];assert p['selection_anchor']==0 and p['original_candidate_id']==2
    assert p['effective_predicted_tpot_ns']>p['origin_predicted_tpot_ns']
    fit=ledger.append_attempt(h.attempt(ledger,mt=220));assert m.next_plan_retry(fit).kind==m.ActionKind.RESOURCE_FIT


def test_ledger_round_trip_replays_actual_receipts_and_refuses_overwrite(tmp_path,monkeypatch):
    h=Harness(tmp_path,monkeypatch);ledger=h.start();ledger=ledger.append_attempt(h.attempt(ledger));ledger,_,_=h.reduce(ledger)
    path=tmp_path/'ledger.json';ledger.write(path);restored=m.PlanLedger.restore(path)
    assert restored.sha256==ledger.sha256 and restored.origin.sha256==h.origin.sha256
    with pytest.raises(FileExistsError):ledger.write(path)
    raw=json.loads(path.read_text());raw['authorizations'][0]['to_pw']=1;save(path,raw)
    with pytest.raises(m.ContractError,match='restored authorization'):m.PlanLedger.restore(path)


def test_committed_cost_proof_is_rechecked_before_resource_fit(tmp_path,monkeypatch):
    h=Harness(tmp_path,monkeypatch);ledger=h.start();ledger=ledger.append_attempt(h.attempt(ledger));ledger,_,catalog=h.reduce(ledger)
    ledger=ledger.append_attempt(h.attempt(ledger,mt=220))
    row=catalog.frozen.data['document']['rows'][0]['row'];Path(row['rounds'][0]['input']['path']).write_text('{}\n')
    action=m.next_plan_retry(ledger);assert action.kind==m.ActionKind.BLOCKED and 'immutable input changed' in action.reason


def test_resolved_width_and_512_manual_claims_are_rejected(tmp_path,monkeypatch):
    h=Harness(tmp_path,monkeypatch);ledger=h.start()
    def wrong_resolved(value,directory):
        raw=json.loads(Path(value['resolved_pin']['path']).read_text());raw['mx_decode_v2']['qkv']['prefetch']=1
        value['resolved_pin']=save(Path(value['resolved_pin']['path']),raw)
    bad=ledger.append_attempt(h.attempt(ledger,mutate=wrong_resolved))
    assert m.next_plan_retry(bad).kind==m.ActionKind.BLOCKED and 'resolved pin differs' in m.next_plan_retry(bad).reason
    def threads512(value,directory):
        raw=json.loads(Path(value['pin']['path']).read_text());raw['mx_decode_v2']['threads']=512
        value['pin']=save(Path(value['pin']['path']),raw)
    bad=ledger.append_attempt(h.attempt(ledger,mutate=threads512))
    assert m.next_plan_retry(bad).kind==m.ActionKind.BLOCKED and 'changed geometry' in m.next_plan_retry(bad).reason


def test_resource_and_MIR_kernel_must_be_the_same_actual_function(tmp_path,monkeypatch):
    h=Harness(tmp_path,monkeypatch);ledger=h.start()
    def mismatch(value,directory):
        log=Path(value['log']['path']);text=log.read_text().replace('# Machine code for function _Zmx_decode_kernel_v2X:',
                                                                  '# Machine code for function _Zmx_decode_kernel_v2Y:')
        log.write_text(text);value['log']=m.record(log)
        value['resources']=save(Path(value['resources']['path']),m.inspect(Path(value['binary']['path']),log,576))
        receipt=json.loads(Path(value['compile_receipt']['path']).read_text())
        receipt.update(log_sha256=value['log']['sha256'],same_compile_mir=value['log']['sha256'])
        value['compile_receipt']=save(Path(value['compile_receipt']['path']),receipt)
    failed=ledger.append_attempt(h.attempt(ledger,mutate=mismatch));action=m.next_plan_retry(failed)
    assert action.kind==m.ActionKind.BLOCKED and 'kernel names differ' in action.reason


def test_variant_numeric_key_representation_is_not_silently_coerced(tmp_path,monkeypatch):
    h=Harness(tmp_path,monkeypatch);ledger=h.start();ledger=ledger.append_attempt(h.attempt(ledger));requested=m.next_plan_retry(ledger)
    def mutate(entry):
        entry['row']['key']['tn']=16.0;entry['row_sha256']=m.digest(entry['row'])
    with pytest.raises(ValueError,match='coordinate tn must be an integer'):
        m.VariantCatalog.load(h.add_variant(requested,change=mutate))


def test_catalog_cannot_execute_arbitrary_qualifier_code(tmp_path,monkeypatch):
    h=Harness(tmp_path,monkeypatch);ledger=h.start();ledger=ledger.append_attempt(h.attempt(ledger));requested=m.next_plan_retry(ledger)
    path=h.add_variant(requested);raw=json.loads(path.read_text())
    untrusted=tmp_path/'untrusted.py';untrusted.write_text('raise RuntimeError("must never execute")\n')
    raw['qualification_tool']=m.record(untrusted);save(path,raw)
    with pytest.raises(m.ContractError,match='arbitrary qualifier code'):m.VariantCatalog.load(path)


def test_forged_width_authorization_cannot_bind_or_commit_costs(tmp_path,monkeypatch):
    from dataclasses import replace
    h=Harness(tmp_path,monkeypatch);ledger=h.start();ledger=ledger.append_attempt(h.attempt(ledger));request=m.next_plan_retry(ledger)
    catalog=m.VariantCatalog.load(h.add_variant(request));action=m.next_plan_retry(ledger,catalog)
    before=ledger.sha256;details=action.details.data;details['authorization']['parent_attempt_sha256']='0'*64
    forged=replace(action,details=m.FrozenJSON.freeze(details))
    with pytest.raises(m.ContractError,match='deterministic action'):ledger.commit_retry(forged,catalog)
    with pytest.raises(m.ContractError,match='actual parent attempt'):
        m.bind_plan_costs(ledger,details['next_pin'],(m.FrozenJSON.freeze(details['authorization']),),catalog)
    assert ledger.sha256==before and ledger.origin.sha256==h.origin.sha256


def test_actual_generated_width_cannot_disagree_with_requested_and_resolved_pin(tmp_path,monkeypatch):
    h=Harness(tmp_path,monkeypatch);ledger=h.start()
    def wrong_code(value,directory):
        path=Path(value['generated_source']['path']);path.write_text(path.read_text().replace(
            'using Qkv = MxGemm<16,1,1,MxOpV2::kStore,true,MxPart::kTileRange,16>;',
            'using Qkv = MxGemm<16,1,1,MxOpV2::kStore,true,MxPart::kTileRange,8>;'))
        value['generated_source']=m.record(path)
        receipt=json.loads(Path(value['compile_receipt']['path']).read_text());receipt['generated_source_sha256']=value['generated_source']['sha256']
        value['compile_receipt']=save(Path(value['compile_receipt']['path']),receipt)
    bad=ledger.append_attempt(h.attempt(ledger,mutate=wrong_code));action=m.next_plan_retry(bad)
    assert action.kind==m.ActionKind.BLOCKED and 'generated body/shape differs' in action.reason


def test_existing_domain_failure_cannot_be_turned_into_a_provisional_origin(tmp_path,monkeypatch):
    def missing(path):raise ValueError('missing qualified legal coordinate: known576 P0 failure remains blocked')
    monkeypatch.setattr(m,'validate_preview',missing)
    with pytest.raises(ValueError,match='known576 P0 failure remains blocked'):
        m.OriginCertificate.capture(tmp_path/'not_a_solvable_domain.json')
