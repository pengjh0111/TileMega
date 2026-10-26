"""CPU synthetic execution harness. No real table, compiler, GPU or timing claim."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

ROOT=Path(os.environ.get('TILEMEGA_TEST_ROOT',Path(__file__).resolve().parents[2]))
MODULE=Path(os.environ.get('TILEMEGA_RESOLVED_RUNNER_MODULE',ROOT/'python/tilemega/mx5_resolved_runner.py'))
m=None


@pytest.fixture(autouse=True)
def runner_module():
    # Other legacy fixture modules deliberately reload package dependencies. Load
    # this consumer after collection so its imports do not change their graph.
    global m
    spec=importlib.util.spec_from_file_location('tilemega.mx5_resolved_runner',MODULE)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)


def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n');return m.record(path)


class Harness:
    def __init__(self,root,monkeypatch,*,times=(1_000_000,995_000,980_000,1_010_000),fail_check=False,fail_build=False):
        self.root=root;self.times=times;self.fail_check=fail_check;self.fail_build=fail_build;self.calls=[];self.pid=100;self.ids={};self.wd={}
        producer=root/'CPU_PRODUCER';producer.write_bytes(b'CPU synthetic producer, never executed')
        self.producer=m.record(producer);source=root/'source';source.mkdir();header=source/'CPU_fixture.h';header.write_text('// CPU source fixture\n')
        snapshot=dict(root=str(source),source_files={'CPU_fixture.h':m.common.sha(header)},compiler=str(producer),compiler_sha256=self.producer['sha256'])
        self.snapshot=save(root/'source.json',snapshot)
        target=save(root/'target.json',{'test_fixture':'CPU only'});bridge=save(root/'bridge.json',{'test_fixture':'CPU only'});table=save(root/'table.json',{'test_fixture':'not a qualified table'})
        req=dict(batch=1,threads=512,capacity=1088,past_lo=64,past_hi=1086,grid=104,runtime_target=target['path'])
        candidates=[];plans=[]
        for index in range(4):
            cls=dict(tn=16,wn=1,d=1,part='kTileRange',prefetch=2,r_next=150)
            pin={'mx_decode_v2':dict(**{n:dict(cls) for n in m.common.CLASSES[:-1]},threads=512,b_min=1,attn_prefetch=0,purpose='manual',diagnostic=True,final=False,solver_selected=False,default_enabled=False)}
            original=dict(index=index,pin=copy.deepcopy(pin),ranks=[index,0,0,0,0,0],predicted_tpot_ns=1_000_000+index,predicted_stage_ns={'test_fixture':1},evidence={'test_fixture':index})
            candidates.append(original)
            if index==2:pin['mx_decode_v2']['qkv']['prefetch']=1
            folder=root/('actual%d'%index);folder.mkdir();so=folder/'plan.so';so.write_bytes(('CPU synthetic actual WD0 '+str(index)).encode());self.sidecars(so)
            final=dict(binary=m.record(so),pin=pin,source_snapshot=self.snapshot,effective_prediction=dict(effective_predicted_tpot_ns=900_000+index))
            plans.append(dict(candidate_id=index,original=original,final=final));self.ids[str(so)]='CPU_WD0_%d'%index
        self.preview=dict(test_fixture='CPU synthetic native-origin/domain injection',request=req,producer=self.producer,input=bridge,stage_table=table,candidates=candidates)
        self.preview_ref=save(root/'preview.json',self.preview);self.domain=root/'CPU_DOMAIN';self.domain.write_text('CPU fixture trusted-tool identity only')
        self.protocol=root/'protocol.json';save(self.protocol,dict(synthetic_fixture=True,note='Not accepted by the production Resolved verifier'))
        self.ready=dict(protocol=m.record(self.protocol),origin_preview=self.preview_ref,domain={'domain_tool':m.record(self.domain)},plans=plans)
        monkeypatch.setattr(m.resolved,'validate_resolved_top4',lambda *a,**k:types.SimpleNamespace(action=m.resolved.v1.Action.READY,reason='CPU explicit proof-boundary injection',data=copy.deepcopy(self.ready)))
        monkeypatch.setattr(m.resolved,'check_measurement_artifact',lambda *a,**k:types.SimpleNamespace(action=m.resolved.v1.Action.READY))
        monkeypatch.setattr(m.common,'validate_preview',lambda p,**k:m.load(p))
        monkeypatch.setattr(m,'verify',self.identity)
        monkeypatch.setattr(m,'_watchdog',self.watchdog)
        monkeypatch.setattr(m,'run_command',self.run)
        model=root/'model';model.mkdir();save(model/'config.json',dict(test_fixture='CPU synthetic model, no weights'))
        self.prefill=root/'prefill.so';self.prefill.write_bytes(b'CPU prefill WD0');self.prefill_w=root/'prefill_w.so';self.prefill_w.write_bytes(b'CPU prefill WD1')
        prompts=save(root/'prompts.json',[[1]*64]);policy=save(root/'policy.json',dict(guard=True))
        self.args=argparse.Namespace(root=ROOT,protocol=self.protocol,domain_tool=self.domain,model=model,prefill_so=self.prefill,prefill_watchdog_so=self.prefill_w,
            prompt_ids=Path(prompts['path']),policy=Path(policy['path']),batch=1,gpu=7,prefill_mode='L2',needs_mib=20480,out=root/'execution',cell_dir=root/'cell')

    @staticmethod
    def sidecars(so):
        for suffix in ('.identity.json','.cu','.plan.json','.ptxas.log','.build_command.txt'):
            Path(str(so)+suffix).write_text('CPU synthetic artifact sidecar '+suffix)

    def identity(self,path):
        path=Path(path).resolve()
        if path in (self.prefill,self.prefill_w):
            return dict(artifact_id='CPU_PREFILL_'+str(int(path==self.prefill_w)),source=m.load(self.snapshot['path']),plan=dict(phase='prefill'),defines={'TILEMEGA_WATCHDOG':str(int(path==self.prefill_w))},trace=False)
        final=next(p['final'] for p in self.ready['plans'] if p['final']['binary']['path']==str(path));c=final['pin']['mx_decode_v2']
        return dict(artifact_id=self.ids[str(path)],source=m.load(self.snapshot['path']),trace=False,
            plan=dict(version=5,executor='mx_static',watchdog=0,diag=False,trace_stage=0,threads=512,b_min=c['b_min'],attn_prefetch=c['attn_prefetch'],
                capacity=1088,past_lo=64,past_hi=1086,batch_lo=1,batch_hi=1,classes={n:c[n] for n in m.common.CLASSES[:-1]}))

    def watchdog(self,so,pin,preview,wd0_snapshot,query_ref=None):
        so=Path(so);need=m.need
        need(so.is_file() and self.wd[str(so)]['pin']==pin,'CPU SDK/WD1 boundary received wrong pin/output')
        if query_ref:need(m.load(m.check_file(query_ref))['test_fixture']=='CPU synthetic actual-query substitute','CPU fixture query changed')
        result=dict(binary=m.record(so),identity_id='CPU_WD1_'+so.parent.parent.name,pin=pin,query=query_ref,source_snapshot=self.snapshot)
        return result,dict(test_fixture='CPU SDK/query boundary, no actual query')

    def run(self,argv,directory,*,timeout=900,env=None):
        directory=Path(directory);directory.mkdir(parents=True);self.pid+=1;self.calls.append(argv)
        assert Path.cwd()==ROOT and env['TILEMEGA_GPU_JOB_LOG']==str(Path(os.environ.get('TILEMEGA_GPU_JOB_LOG') or ROOT/'runs/mx5/gpu_jobs.log').resolve())
        process=dict(argv=argv,pid=self.pid,started=self.pid,returncode=0,timeout_seconds=timeout)
        if '--mx-decode-v2-solve' in argv:
            save(Path(argv[3]+'.mx5_candidates.json'),self.preview)
        elif '--watchdog' in argv:
            if self.fail_build:process['returncode']=1
            else:
                so=Path(argv[3]);so.write_bytes(('CPU synthetic WD1 '+str(self.pid)).encode());self.sidecars(so);pin=m.load(argv[argv.index('--evaluate-configs')+1]);self.wd[str(so)]=dict(pin=pin)
        else:
            guard=Path(argv[3]);guard.mkdir(parents=True);save(guard/'guard_result.json',dict(code=0));(guard/'pgid').write_text(str(self.pid+1000));(guard/'guard.jsonl').write_text(json.dumps(dict(phase='preflight',test_fixture='CPU synthetic telemetry'))+'\n'+json.dumps(dict(phase='after',exit_code=0,test_fixture='CPU synthetic telemetry'))+'\n')
            if 'tilemega.mx5_plan_occupancy' in argv:
                save(Path(argv[argv.index('--out')+1]),dict(test_fixture='CPU synthetic actual-query substitute'))
            elif 'tilemega.serving.measure' in argv:
                so=Path(argv[argv.index('--decode-so')+1]);index=next(i for i,p in enumerate(self.ready['plans']) if p['final']['binary']['path']==str(so));seconds=self.times[index]*1e-9
                execution=dict(artifact_id=self.ids[str(so)],executor='L1',loop=True)
                runs=[dict(N=n,run=j,warmup=j==0,step_ns_read=False,gpu_step_ms=[.1],decode_loop_used=n==1024,execution_identity=execution if n==1024 else {**execution,'loop':False}) for n in (1024,1) for j in range(3)]
                report=dict(test_fixture='CPU synthetic measurement, no timing was executed',batch=1,max_tokens=1024,decode_loop_used=True,step_ns_read=False,timed_tokens_identical=True,
                    execution_identity=execution,prefill_execution_identity=dict(artifact_id='CPU_PREFILL_0'),runs=runs,tpot_seconds=seconds,e2e_seconds=1+seconds*1023,ttft_seconds=1,measurement_policy=dict(guard=True))
                save(Path(argv[argv.index('--out')+1])/'measurements.json',report)
            else:
                out=Path(argv[11]);tokens=[[1]*512]
                save(out/'c2/mode_check.json',dict(batch=1,steps=512,**{'pass':True},guard_mode='exclusive',timing_eligible=False,mismatches={'L1_separate':0,'L1_loop':0}))
                for arm in ('L1_separate','L1_loop'):save(out/('c2/tokens_'+arm+'.json'),tokens)
                save(out/'c1/hf_check.json',dict(model=str(self.args.model),batch=1,max_new_tokens=512,prompt_len=64,positions=512,**{'pass':not self.fail_check},required_ratio=.99,allowed_maximum=3.0,gap_le_0_5_ratio=1.,max_gap=0.))
        save(directory/'process.json',process);return process


def test_explicit_runner_executes_original_order_exact_artifacts_and_publishes_effective_pin(tmp_path,monkeypatch):
    h=Harness(tmp_path,monkeypatch);monkeypatch.chdir(tmp_path);selected=m.orchestrate(h.args)
    assert selected['selected_index']==2 and selected['selected_pin']['mx_decode_v2']['qkv']['prefetch']==1
    assert m.load(h.args.out/'selected.pin.json')==h.ready['plans'][2]['final']['pin']
    assert (h.args.cell_dir/'decode/plan.so').resolve()==Path(h.ready['plans'][2]['final']['binary']['path'])
    assert m.verify_results(h.args.out/'results.json',protocol=h.protocol,domain_tool=h.domain)==selected
    measurements=[a for a in h.calls if 'tilemega.serving.measure' in a]
    assert [next(i for i,p in enumerate(h.ready['plans']) if p['final']['binary']['path']==a[a.index('--decode-so')+1]) for a in measurements]==[0,1,2,3,1,2,3,0]
    last_check=max(i for i,a in enumerate(h.calls) if 'bash' in a)
    assert all(i>last_check for i,a in enumerate(h.calls) if 'tilemega.serving.measure' in a)
    assert all(a[1]=='7' and a[2]=='900' for a in measurements)


def test_one_percent_rule_keeps_original_anchor_and_includes_boundary(tmp_path,monkeypatch):
    h=Harness(tmp_path,monkeypatch);artifacts=[dict(binary=p['final']['binary']) for p in h.ready['plans']]
    for value,expected in ((995,0),(990,1),(989,1)):
        result=m._selection(h.ready,[[1000,1000],[value,value],[1100,1100],[1200,1200]],artifacts,artifacts)
        assert result['selected_index']==expected


@pytest.mark.parametrize('failure',['check','build'])
def test_failure_never_measures_or_publishes_or_overwrites(tmp_path,monkeypatch,failure):
    h=Harness(tmp_path,monkeypatch,fail_check=failure=='check',fail_build=failure=='build')
    existing=h.args.cell_dir/'decode/plan.so';existing.parent.mkdir(parents=True);existing.write_bytes(b'Existing untouched cell')
    with pytest.raises(ValueError):m.orchestrate(h.args)
    assert existing.read_bytes()==b'Existing untouched cell'
    assert not any('tilemega.serving.measure' in a for a in h.calls)
    assert m.load(sorted(h.args.out.glob('state*.json'))[-1])['status']=='blocked'


@pytest.mark.parametrize('mutation',[
    lambda d:d['manifest']['measurements'][0]['process']['argv'].__setitem__(1,'6'),
    lambda d:d['manifest']['measurements'][0]['process']['argv'].__setitem__(2,'1800'),
    lambda d:d['manifest']['measurements'].pop(),
    lambda d:d['selection'].update(selected_index=0),
    lambda d:d['manifest']['measurements'][1].update(measurement_pid=d['manifest']['measurements'][0]['measurement_pid']),
])
def test_results_consumer_rejects_card_timeout_missing_pair_anchor_or_pid_change(tmp_path,monkeypatch,mutation):
    h=Harness(tmp_path,monkeypatch);h.args.cell_dir=None;m.orchestrate(h.args)
    doc=m.load(h.args.out/'results.json');mutation(doc);path=tmp_path/'changed_results.json';save(path,doc)
    with pytest.raises(ValueError):m.verify_results(path,protocol=h.protocol,domain_tool=h.domain)


def test_completed_selection_pin_tamper_and_existing_cell_are_not_overwritten(tmp_path,monkeypatch):
    h=Harness(tmp_path,monkeypatch);h.args.cell_dir=None;selected=m.orchestrate(h.args)
    save(h.args.out/'selected.pin.json',{'test_fixture':'tampered pin'})
    with pytest.raises(ValueError):m.verify_results(h.args.out/'results.json',protocol=h.protocol,domain_tool=h.domain)
    cell=tmp_path/'existing';target=cell/'decode/plan.so';target.parent.mkdir(parents=True);target.write_bytes(b'old')
    with pytest.raises(ValueError):m._publish(cell,selected)
    assert target.read_bytes()==b'old' and not (cell/'decode_w/plan.so').exists()


def test_script_explicit_missing_domain_fails_before_any_gpu_or_build(tmp_path):
    script=Path(os.environ.get('TILEMEGA_RESOLVED_SCRIPT',ROOT/'scripts/mx_serving_cell.sh'))
    env=dict(os.environ,CELLS=str(tmp_path/'cells'),MX5_RESOLVED_V2=str(tmp_path/'protocol'));env.pop('MX5_DOMAIN_TOOL',None)
    p=subprocess.run(['bash',str(script),'solve','CPU_SYNTHETIC', '1','decode'],env=env,capture_output=True,text=True)
    assert p.returncode!=0 and 'MX5_DOMAIN_TOOL' in p.stderr
    assert not list(tmp_path.rglob('plan.so'))


def test_watchdog_uses_real_receipt_resource_mir_and_zero_query_parsers(tmp_path):
    script=r'''
import importlib.util,sys,os
from pathlib import Path
import pytest
ROOT=Path(sys.argv[1]);MODULE=Path(sys.argv[2]);tmp_path=Path(sys.argv[3]);monkeypatch=pytest.MonkeyPatch()
spec=importlib.util.spec_from_file_location('tilemega.mx5_resolved_runner',MODULE);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
spec=importlib.util.spec_from_file_location('runner_strict_cpu_fixture',ROOT/'test/python/test_mx5_thread_exception.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
h=fixture.Fixture(tmp_path,monkeypatch)
# Only the identity sidecar boundary is injected here: all native receipt,
# compiler options, generated source, resource/MIR and zero-query parsers run.
monkeypatch.setattr(m,'_identity',lambda *a,**k:dict(artifact_id='CPU_SYNTHETIC_WD1_IDENTITY'))
def check(**kwargs):
    bundle=fixture.m._load(h.artifact(watchdog=1,**kwargs));native=fixture.m._load(bundle['native_receipt']);files=native['files']
    return m._watchdog(Path(files['binary']['path']),h.pin(),h.preview,m.load(files['source_snapshot']['path']),bundle['query'])
wd,inputs=check()
assert inputs['context_kind']=='compile_receipt' and wd['query'] is not None
for kwargs in (dict(mt=256),dict(frame=4),dict(shared=16),dict(resident=0),dict(extra=('-mllvm','--max-mtreg-number=255')),
               dict(mir_transform=lambda text:text.replace('MMA_16X16X16BF16','NO_MMA_OPCODE')),
               dict(mir_transform=lambda text:text.replace('# End machine code','TRUNCATED'))):
    with pytest.raises((ValueError,RuntimeError)):check(**kwargs)
'''
    result=subprocess.run([sys.executable,'-c',script,str(ROOT),str(MODULE),str(tmp_path)],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES=''),capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_fresh_solver_failure_keeps_actual_failure_process_and_never_measures(tmp_path,monkeypatch):
    h=Harness(tmp_path,monkeypatch);original=h.run
    def failed(argv,directory,**kwargs):
        if '--mx-decode-v2-solve' not in argv:return original(argv,directory,**kwargs)
        Path(directory).mkdir(parents=True)
        process=dict(argv=argv,pid=112,returncode=17)
        save(Path(directory)/'process.json',process);return process
    monkeypatch.setattr(m,'run_command',failed)
    with pytest.raises(ValueError,match='exact top4 revalidation failed'):m.orchestrate(h.args)
    state=m.load(sorted(h.args.out.glob('state*.json'))[-1])
    assert state['manifest']['native_revalidation']['returncode']==17
    assert not h.calls and not (h.args.cell_dir/'decode/plan.so').exists()


def test_publication_exposes_complete_verifiable_identity_and_rolls_back_links(tmp_path,monkeypatch):
    from tilemega.build import identity
    source=tmp_path/'actual';source.mkdir();so=source/'candidate.so';so.write_bytes(b'CPU synthetic ELF bytes')
    payload=dict(schema=1,binary_sha256=identity.sha(so))
    for suffix,key in (('.cu','generated_source_sha256'),('.plan.json','manifest_sha256'),('.ptxas.log','resources_sha256'),('.build_command.txt','build_command_sha256')):
        path=Path(str(so)+suffix);path.write_text('CPU synthetic '+suffix);payload[key]=identity.sha(path)
    payload['artifact_id']=identity.digest(payload);save(Path(str(so)+'.identity.json'),payload)
    selection=dict(selected_artifact=m.record(so),selected_watchdog=m.record(so))
    cell=tmp_path/'cell';m._publish(cell,selection)
    assert identity.verify(cell/'decode/plan.so')['artifact_id']==payload['artifact_id']
    assert identity.verify(cell/'decode_w/plan.so')['artifact_id']==payload['artifact_id']
    failed=tmp_path/'failed';original=Path.symlink_to;count=0
    def fail_second(path,*args,**kwargs):
        nonlocal count
        count+=1
        if count==2:raise OSError('CPU injected link failure')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'symlink_to',fail_second)
    with pytest.raises(OSError):m._publish(failed,selection)
    assert not failed.exists()


@pytest.mark.parametrize('external',[True,False])
def test_external_gpu_job_log_is_preserved_and_manifest_bound(tmp_path,monkeypatch,external):
    h=Harness(tmp_path,monkeypatch);log=tmp_path/'central.log' if external else ROOT/'runs/mx5/gpu_jobs.log'
    monkeypatch.setenv('TILEMEGA_GPU_JOB_LOG',str(log) if external else '')
    h.args.cell_dir=None;m.orchestrate(h.args)
    assert m.load(h.args.out/'results.json')['manifest']['gpu_job_log']==str(log)


@pytest.mark.parametrize('change',['artifact','origin_order','guard_telemetry'])
def test_consumer_rechecks_measured_artifact_original_ranks_and_raw_guard(tmp_path,monkeypatch,change):
    h=Harness(tmp_path,monkeypatch);h.args.cell_dir=None;m.orchestrate(h.args)
    path=h.args.out/'results.json';doc=m.load(path)
    if change=='artifact':
        item=doc['manifest']['measurements'][0];report=m.load(item['raw']['path']);report['execution_identity']['artifact_id']='CPU_OTHER_BINARY'
        item['raw']=save(Path(item['raw']['path']),report)
    elif change=='origin_order':
        ref=doc['manifest']['fresh_preview'];preview=m.load(ref['path']);preview['candidates'][0]['ranks']=[7]*6
        doc['manifest']['fresh_preview']=save(Path(ref['path']),preview)
    else:
        item=doc['manifest']['measurements'][0];guard=Path(item['process']['argv'][3])
        (guard/'guard.jsonl').write_text(json.dumps(dict(phase='after',exit_code=0))+'\n')
        item['guard']['guard.jsonl']=m.record(guard/'guard.jsonl')
    save(path,doc)
    with pytest.raises(ValueError):m.verify_results(path,protocol=h.protocol,domain_tool=h.domain)
