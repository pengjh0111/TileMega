"""CPU-only orchestration checks. Synthetic values are never qualified stage data."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

ROOT = Path(os.environ.get('TILEMEGA_TEST_ROOT', Path(__file__).resolve().parents[2]))
MODULE = Path(os.environ.get('TILEMEGA_MX5_SELECT_MODULE', ROOT / 'python/tilemega/mx5_select.py'))
spec = importlib.util.spec_from_file_location('tilemega.mx5_select', MODULE)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def args(tmp_path):
    return SimpleNamespace(root=ROOT, gpu=7, model=tmp_path / 'model', prefill_so=tmp_path / 'prefill.so',
        prompt_ids=tmp_path / 'prompts.json', batch=1, prefill_mode='L2', policy=tmp_path / 'policy.json')


def measurement():
    execution = dict(artifact_id='synthetic-artifact', executor='L1', loop=True)
    runs = [dict(N=n, run=i, warmup=i == 0, step_ns_read=False, gpu_step_ms=[.1],
                 decode_loop_used=n == 1024, execution_identity=execution if n == 1024 else {**execution, 'loop':False}) for n in (1024, 1) for i in range(3)]
    return dict(synthetic_fixture=True, batch=1, max_tokens=1024, decode_loop_used=True, step_ns_read=False,
                timed_tokens_identical=True, execution_identity=execution,
                prefill_execution_identity=dict(artifact_id='synthetic-prefill'), runs=runs,
                tpot_seconds=.001, e2e_seconds=2.023, ttft_seconds=1.0)


def test_fixed_measurement_protocol_command(tmp_path):
    argv = m.measurement_command(args(tmp_path), tmp_path / 'plan.so', tmp_path / 'measurement', tmp_path / 'guard')
    assert argv[:4] == [str(ROOT / 'scripts/mx_gpu_job.sh'), '7', '900', str(tmp_path / 'guard')]
    for key, value in (('--warmup', '1'), ('--repeats', '2'), ('--step-events', '0'), ('--mode', 'L1'),
                       ('--decode-loop', '1'), ('--max-new-tokens', '1024')):
        assert argv[argv.index(key) + 1] == value
    assert '--decode-so' in argv and '--prefill-so' in argv


def test_request_minus_ttft_protocol_and_instrumentation_rejections():
    preview = dict(request=dict(batch=1))
    assert m.validate_measurement(measurement(), preview, 'synthetic-artifact', 'synthetic-prefill') == 1e6
    for mutate, error in [
        (lambda d: d.update(step_ns_read=True), 'protocol'),
        (lambda d: d['runs'][1].update(gpu_step_ms=[1,2]), 'step-events=0'),
        (lambda d: d['runs'].pop(), 'warmup=1'),
        (lambda d: d['execution_identity'].update(artifact_id='other'), 'artifact'),
        (lambda d: d.update(tpot_seconds=.002), 'TPOT differs'),
        (lambda d: d.update(tpot_p50_seconds=0), 'quantiles'),
    ]:
        doc = copy.deepcopy(measurement()); mutate(doc)
        with pytest.raises(ValueError, match=error):
            m.validate_measurement(doc, preview, 'synthetic-artifact', 'synthetic-prefill')


def test_cpu_subprocesses_are_distinct_and_logs_immutable(tmp_path):
    first = m.run_command([sys.executable, '-c', 'print("CPU fixture process")'], tmp_path / 'one')
    second = m.run_command([sys.executable, '-c', 'print("CPU fixture process")'], tmp_path / 'two')
    assert first['returncode'] == second['returncode'] == 0 and first['pid'] != second['pid']
    assert first['timeout_seconds'] == 900
    with pytest.raises(FileExistsError):
        m.run_command([sys.executable, '-c', 'raise RuntimeError("must not run")'], tmp_path / 'one')


def test_stage_archive_binds_measured_body_headers_to_actual_plan(tmp_path):
    entries = {'include/tilemega/Backend/Mx/Fixture.h': b'CPU synthetic header\n',
               'experiments/mx5/stage_bench.cu': b'CPU synthetic bench\n'}
    digest = hashlib.sha256()
    archive = tmp_path / 'sources.tar.gz'
    with tarfile.open(archive, 'w:gz') as output:
        for name, data in sorted(entries.items()):
            digest.update(name.encode() + b'\0' + data + b'\0')
            member = tarfile.TarInfo(name); member.size = len(data); output.addfile(member, io.BytesIO(data))
    row = dict(provenance=dict(source_sha256=digest.hexdigest()), synthetic_fixture=True)
    candidate = dict(evidence={name: [row] for name in m.CLASSES})
    files = {name: hashlib.sha256(data).hexdigest() for name, data in entries.items() if name.startswith('include/')}
    identity = dict(source=dict(source_files=files))
    mapping = dict(sources=[dict(source_sha256=digest.hexdigest(), archive=str(archive))])
    assert len(m.verify_stage_sources(candidate, identity, mapping)) == 1
    files['include/tilemega/Backend/Mx/Fixture.h'] = 'changed'
    with pytest.raises(ValueError, match='headers differ'):
        m.verify_stage_sources(candidate, identity, mapping)


def test_diagnostic_table_cannot_enter_measurement(tmp_path):
    table = tmp_path / 'table.json'; table.write_text(json.dumps(dict(schema_version=2,
        table_kind='mx5_decode_stage_costs', qualified=False, synthetic_fixture=True)))
    empty = tmp_path / 'input'; empty.write_text('CPU fixture')
    preview = dict(schema_version=1, kind='mx5_decode_candidate_preview', diagnostic=True, final=False,
                   solver_selected=False, default_enabled=False, stage_table=m.record(table),
                   input=m.record(empty), producer=m.record(empty))
    path = tmp_path / 'preview.json'; path.write_text(json.dumps(preview))
    with pytest.raises(ValueError, match='qualified schema-v2'):
        m.validate_preview(path)


def test_explicit_native_route_rejects_unqualified_table_before_device_build(tmp_path):
    binary = Path(os.environ.get('TILEMEGA_BIN', ROOT / 'build/tools/tilemega'))
    if not binary.exists():
        pytest.skip('host CLI unavailable')
    table = tmp_path / 'diagnostic.json'; table.write_text(json.dumps(dict(schema_version=2,
        table_kind='mx5_decode_stage_costs', qualified=False, synthetic_fixture=True)))
    output = tmp_path / 'result'
    command = [str(binary), 'compile', str(tmp_path / 'missing_bridge.json'), str(output), '--serving', 'decode',
               '--frontend', 'decoder', '--emit', 'serving', '--selection', 'predicted', '--mx-decode-v2-solve', str(table)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30,
                            env={**os.environ, 'CUDACXX': '/no/device/compiler/allowed'})
    assert result.returncode != 0
    assert 'requires qualified schema-v2 stage costs' in result.stderr
    assert not Path(str(output) + '.build_command.txt').exists()
    assert not Path(str(output) + '.mx5_candidates.json').exists()


def test_unresolved_512_policy_is_not_a_blanket_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'validate_preview', lambda path: dict(request=dict(batch=1),
        candidates=[dict(pin=dict(mx_decode_v2=dict(threads=512))) for _ in range(4)]))
    a = args(tmp_path); a.preview = tmp_path / 'preview'; a.out = tmp_path / 'must-not-create'
    with pytest.raises(ValueError, match='actual 576 failures.*D7 evidence'):
        m.orchestrate(a)
    assert not a.out.exists()


def mocked_orchestration(tmp_path, monkeypatch, resource_failure=False):
    """Exercise process scheduling with injected CPU fakes; no stage table qualifies."""
    from tilemega.build import identity
    a = args(tmp_path)
    for path, text in ((a.prompt_ids, json.dumps([[1]*64])), (a.policy, '{"guard":true}')):
        path.write_text(text)
    a.model.mkdir(); (a.model / 'config.json').write_text('{"synthetic_fixture":true}')
    a.stage_source_map = tmp_path / 'source_map.json'; a.stage_source_map.write_text('{"sources":[]}')
    a.preview = tmp_path / 'run.mx5_candidates.json'; a.preview.write_text('{"synthetic_fixture":true}')
    a.out = tmp_path / 'output'; a.needs_mib = 20480
    inputs = {}
    for name in ('producer', 'input', 'stage_table'):
        path = tmp_path / name; path.write_text('CPU synthetic input; not qualified stage data'); inputs[name] = m.record(path)
    request = dict(batch=1, threads=576, capacity=1088, past_lo=64, past_hi=1086, runtime_target=str(tmp_path / 'target'))
    candidates = []
    for i in range(4):
        pin = dict(mx_decode_v2=dict(threads=576, purpose='manual', diagnostic=True, final=False))
        path = tmp_path / ('pin%d.json' % i); path.write_text(json.dumps(pin))
        candidates.append(dict(index=i, pin=pin, pin_file=m.record(path), predicted_tpot_ns=100+i,
            predicted_stage_ns={}, ranks=[0]*6, evidence=dict(synthetic_fixture=True)))
    preview = dict(synthetic_fixture=True, request=request, candidates=candidates, **inputs)
    monkeypatch.setattr(m, 'validate_preview', lambda path: preview)
    monkeypatch.setattr(identity, 'verify', lambda path: dict(artifact_id='synthetic-prefill'))
    monkeypatch.setattr(m, 'verify_candidate', lambda so, candidate, preview, sources:
        dict(identity=dict(artifact_id='synthetic-%d' % candidate['index']), synthetic_fixture=True))
    calls = []
    def execute(argv, directory, **kwargs):
        calls.append(argv)
        Path(directory).mkdir(parents=True, exist_ok=True)
        if '--evaluate-configs' in argv:
            so = Path(argv[3]); so.write_text('CPU fixture binary, not executable')
            if resource_failure:
                Path(str(so) + '.resource_failure.json').write_text(json.dumps(dict(violations=['stack frame is 568 bytes'])))
                return dict(argv=argv, pid=100, returncode=1)
        if 'tilemega.serving.measure' in argv:
            guard = Path(argv[3]); guard.mkdir()
            out = Path(argv[argv.index('--out') + 1]); out.mkdir()
            so = Path(argv[argv.index('--decode-so') + 1]); index = int(so.parent.name.removeprefix('candidate'))
            (guard / 'guard_result.json').write_text('{"code":0}')
            (guard / 'pgid').write_text(str(1000+len(calls)))
            report = measurement(); report['execution_identity']['artifact_id'] = 'synthetic-%d' % index
            for run in report['runs']: run['execution_identity']['artifact_id'] = 'synthetic-%d' % index
            report['measurement_policy'] = dict(guard=True)
            (out / 'measurements.json').write_text(json.dumps(report))
        if '--mx-decode-v2-results' in argv:
            Path(argv[3] + '.mx5_selection.json').write_text(json.dumps(dict(kind='mx5_decode_top4_selection', final=False, selected_index=0)))
        return dict(argv=argv, pid=100+len(calls), returncode=0)
    monkeypatch.setattr(m, 'run_command', execute)
    return a, calls


def test_all_four_candidates_get_two_separate_guarded_processes(tmp_path, monkeypatch):
    a, calls = mocked_orchestration(tmp_path, monkeypatch)
    result = m.orchestrate(a)
    measures = [call for call in calls if 'tilemega.serving.measure' in call]
    order = [int(Path(call[call.index('--decode-so')+1]).parent.name.removeprefix('candidate')) for call in measures]
    assert order == [0,1,2,3,1,2,3,0]
    assert len({p['measurement_pid'] for p in result['processes']}) == 8
    assert len(result['resource_attempts']) == 4 and result['status'] == 'measured_candidate_selection'
    assert result['final'] is False and result['selection_applied'] is False
    assert any('--mx-decode-v2-results' in call for call in calls)
    assert m.validate_results(a.out / 'verified_measurements.json', a.preview)['process_tpot_ns'] == [[1e6,1e6]]*4


def test_576_resource_failure_stops_without_pruning_or_fallback(tmp_path, monkeypatch):
    a, calls = mocked_orchestration(tmp_path, monkeypatch, resource_failure=True)
    with pytest.raises(RuntimeError, match='unresolved per-plan resource policy: stack frame is 568 bytes.*D7'):
        m.orchestrate(a)
    assert len([call for call in calls if '--evaluate-configs' in call]) == 1
    assert not any('tilemega.serving.measure' in call for call in calls)
    states = sorted(a.out.glob('state*.json')); final = json.loads(states[-1].read_text())
    assert final['status'] == 'blocked' and final['resource_attempts'][0]['failure']['violations'] == ['stack frame is 568 bytes']
    assert final['resource_attempts'][0]['thread_fallback_performed'] is False


def test_pair_only_result_blob_cannot_claim_measured_selection(tmp_path):
    path = tmp_path / 'pairs.json'; path.write_text(json.dumps(dict(process_tpot_ns=[[1,1]]*4)))
    with pytest.raises(ValueError, match='fresh-process/guard/artifact provenance'):
        m.validate_results(path, tmp_path / 'unused-preview')



def schema3_preview_fixture(tmp_path,status='predicted_top4_resource_review_required',count=4):
    """Temporary synthetic parser fixture, never a real stage measurement/table."""
    def save(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value));return m.record(path)
    scope=dict(model_id='CPU_SYNTHETIC',model=dict(capacity=1088),batch=1,threads=576,grid=104,past_lo=64,past_hi=1086,
               source_sha256={kind:'1'*64 for kind in ('gemm','head','attention')})
    rows=[];evidence={};pin=dict(threads=576,b_min=1,attn_prefetch=1,purpose='manual',diagnostic=True,final=False,solver_selected=False,default_enabled=False)
    for number,cls in enumerate(m.CLASSES):
        evidence[cls]=[]
        for length in ((65,256,512,768,1088) if cls=='attention' else (None,)):
            key=dict(threads=576,b=1)
            if cls=='attention':key.update(length=length,b_min=1,prefetch_kind=1)
            else:
                key.update(tn=16,wn=1,depth=1,prefetch=1,part=1,n=128*(number+1),k=1024)
                pin[cls]=dict(tn=16,wn=1,d=1,prefetch=1,part='kTileRange',r_next=160)
            row=dict(kind='attention' if cls=='attention' else 'head' if cls=='head' else 'gemm',key=key,r_next=160,
                     fixture_provenance='CPU_SYNTHETIC_PARSER_ONLY')
            rows.append(row);evidence[cls].append(row)
    rejections=[dict(fixture_provenance='CPU_SYNTHETIC_REJECTION_PARSER_ONLY')]
    coverage=dict(expected_keys=len(rows)+1,costed_keys=len(rows),rejected_keys=1,missing_keys=0)
    table=dict(schema_version=3,table_kind='mx5_decode_stage_costs',qualified=True,diagnostic_reasons=[],
        qualification_scope='complete_outcomes_single_request_v1',outcome_domain_complete=True,full_table=False,
        execution_512_approved=False,global_dt576_policy_approved=False,target_defaults_changed=False,
        scope=scope,coverage=coverage,resource_rejections=rejections,rows=rows,
        r_next_policy_audit=dict(modern_normal_checked=len(rows)+1,legacy_cost_only_unverifiable=0),
        fixture_provenance='CPU_SYNTHETIC_PARSER_ONLY')
    report={key:copy.deepcopy(table[key]) for key in ('scope','coverage','resource_rejections','r_next_policy_audit')}
    report.update(execution_512_approved=False,final=False,default_enabled=False,
        ranking_claim='exact_top4_of_certified_executable_stage_domain' if count==4 else 'no_top4_claim')
    input_file=tmp_path/'input';input_file.write_text('CPU synthetic input, never executed')
    candidates=[]
    if count==4:
        for i in range(4):
            value=dict(mx_decode_v2=copy.deepcopy(pin));descriptor=save('pin%d.json'%i,value)
            candidates.append(dict(index=i,pin=value,pin_file=descriptor,evidence=copy.deepcopy(evidence),
                diagnostic=True,final=False,solver_selected=False,default_enabled=False,execution_authorized=False))
    preview=dict(schema_version=1,kind='mx5_decode_candidate_preview',diagnostic=True,final=False,solver_selected=False,default_enabled=False,
        stage_table=save('table.json',table),input=m.record(input_file),producer=m.record(input_file),
        request=dict(batch=1,threads=576,grid=104,past_lo=64,past_hi=1086,capacity=1088),
        preview_mode='schema3_qualified_domain_diagnostic',source_schema_version=3,schema3_diagnostic_only=True,
        execution_authorized=False,results_consumption_allowed=False,domain_report=report,status=status,
        available_candidate_count=count,candidates=candidates)
    path=tmp_path/'preview.json';save(path.name,preview);return path,preview,table


@pytest.mark.parametrize('status,count',[('predicted_top4_resource_review_required',4),('no_executable_plan',0),
    ('insufficient_top4',1),('insufficient_top4',2),('insufficient_top4',3)])
def test_complete_schema3_preview_requires_explicit_audit_opt_in(tmp_path,status,count):
    path,preview,_=schema3_preview_fixture(tmp_path,status,count)
    with pytest.raises(ValueError,match='explicit allow_schema3_diagnostic'):m.validate_preview(path)
    assert m.validate_preview(path,allow_schema3_diagnostic=True)==preview
    assert preview['execution_authorized'] is False
    if count<4:assert not preview['candidates']


@pytest.mark.parametrize('change',[
    lambda d:d.pop('preview_mode'),lambda d:d.update(schema3_diagnostic_only=False),
    lambda d:d.update(execution_authorized=True),lambda d:d.update(results_consumption_allowed=True),
    lambda d:d['domain_report']['resource_rejections'].clear(),
    lambda d:d['domain_report']['scope'].update(threads=512),
    lambda d:d['domain_report']['coverage'].update(expected_keys=999),
    lambda d:d['domain_report'].update(default_enabled=True),
    lambda d:d['request'].update(threads=512),
    lambda d:d.update(available_candidate_count=3),
    lambda d:d['candidates'][0].update(execution_authorized=True),
])
def test_schema3_diagnostic_cannot_hide_rejections_or_grant_policy(tmp_path,change):
    path,preview,_=schema3_preview_fixture(tmp_path);change(preview);path.write_text(json.dumps(preview))
    with pytest.raises(ValueError):m.validate_preview(path,allow_schema3_diagnostic=True)


def test_schema3_opt_in_does_not_open_measurement_or_results_consumers(tmp_path):
    path,preview,_=schema3_preview_fixture(tmp_path)
    assert m.validate_preview(path,allow_schema3_diagnostic=True)['available_candidate_count']==4
    a=args(tmp_path);a.preview=path;a.out=tmp_path/'must-not-build'
    with pytest.raises(ValueError,match='explicit allow_schema3_diagnostic'):m.orchestrate(a)
    assert not a.out.exists()
    results=tmp_path/'results.json';results.write_text(json.dumps(dict(execution_manifest={})))
    with pytest.raises(ValueError,match='explicit allow_schema3_diagnostic'):m.validate_results(results,path)


def test_schema3_incomplete_top4_never_accepts_fake_pin(tmp_path):
    path,preview,_=schema3_preview_fixture(tmp_path,'insufficient_top4',1)
    preview['candidates']=[dict(index=0,pin=dict(mx_decode_v2={}))];path.write_text(json.dumps(preview))
    with pytest.raises(ValueError,match='must not serialize candidate pins'):
        m.validate_preview(path,allow_schema3_diagnostic=True)



def test_schema3_self_consistent_manual_pin_still_cannot_enable_default(tmp_path):
    path,preview,_=schema3_preview_fixture(tmp_path)
    candidate=preview['candidates'][0];candidate['pin']['mx_decode_v2']['default_enabled']=True
    pin_file=Path(candidate['pin_file']['path']);pin_file.write_text(json.dumps(candidate['pin']))
    candidate['pin_file']=m.record(pin_file);path.write_text(json.dumps(preview))
    with pytest.raises(ValueError,match='manual pins cannot enable defaults'):
        m.validate_preview(path,allow_schema3_diagnostic=True)
