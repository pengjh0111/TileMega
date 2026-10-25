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
