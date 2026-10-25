"""CPU synthetic receipt fixtures only; no GPU, real timings or compiler queries."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import pytest

ROOT = Path(os.environ.get('TILEMEGA_TEST_ROOT', Path(__file__).resolve().parents[2]))
spec = importlib.util.spec_from_file_location('cpu_plan_ledger_fixture', ROOT/'test/python/test_mx5_plan_resources.py')
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
MODULE = Path(os.environ.get('TILEMEGA_MX5_PLAN_RECEIPT_MODULE', ROOT/'python/tilemega/mx5_plan_receipt.py'))
spec = importlib.util.spec_from_file_location('tilemega.mx5_plan_receipt', MODULE)
m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)


def native(h, ledger, **kwargs):
    path = h.attempt(ledger, **kwargs)
    attempt = json.loads(path.read_text())
    original = json.loads(f.m.check_file(attempt['compile_receipt']).read_text())
    value = dict(original, schema_version=1, kind='mx5_native_compile_receipt',
                 wait_status=original['returncode'] << 8, changed_inputs=[],
                 files={name: attempt[name] for name in (*m.REQUIRED, 'binary')},
                 test_fixture='CPU synthetic data, not a real native/GPU artifact')
    for name in m.REQUIRED:
        value[name+'_sha256'] = attempt[name]['sha256']
    receipt = path.parent/'native.json'; f.save(receipt, value)
    return receipt, attempt


def assemble(h, ledger, receipt, attempt, **kwargs):
    return m.assemble(receipt, ledger, stage_sources=h.sources,
        resources=f.m.check_file(attempt['resources']) if attempt.get('resources') else None,
        occupancy=f.m.check_file(attempt['occupancy']) if attempt.get('occupancy') else None, **kwargs)


@pytest.mark.parametrize('mt,stack,occupancy,classification', [
    (228,0,1,'resource_failure'), (168,4,1,'resource_failure'),
    (168,0,1,'resource_fit'), (168,0,None,'blocked')])
def test_scoped_receipt_reaches_unchanged_real_ledger_gates(tmp_path,monkeypatch,mt,stack,occupancy,classification):
    h=f.Harness(tmp_path,monkeypatch); ledger=h.start()
    receipt, attempt=native(h,ledger,mt=mt,stack=stack,occupancy=occupancy)
    scoped=assemble(h,ledger,receipt,attempt)
    assert scoped['ordinal']==0 and scoped['origin_sha256']==h.origin.sha256
    path=tmp_path/'scoped.json';m.write_exclusive(path,scoped)
    result=ledger.append_attempt(path)
    assert result.attempts[-1].data['verified']['classification']==classification
    assert ledger.attempts==() and ledger.pin==result.pin
    with pytest.raises(FileExistsError):m.write_exclusive(path,scoped)


def test_actual_compiler_failure_missing_binary_is_preserved_and_blocked(tmp_path,monkeypatch):
    h=f.Harness(tmp_path,monkeypatch);ledger=h.start()
    receipt,attempt=native(h,ledger,compiler_rc=7,binary=False,occupancy=None)
    scoped=assemble(h,ledger,receipt,attempt)
    assert scoped['binary'] is None and scoped['resources'] is None
    path=tmp_path/'failure.json';m.write_exclusive(path,scoped)
    result=ledger.append_attempt(path)
    assert result.attempts[-1].data['verified']['classification']=='blocked'
    assert 'compiler failed' in result.attempts[-1].data['verified']['reason']


@pytest.mark.parametrize('change,reason', [
    (lambda v:v.update(returncode=1),'wait status'),
    (lambda v:v.update(changed_inputs=['pin']),'inputs changed'),
    (lambda v:v.update(same_compile_mir='0'*64),'MIR'),
    (lambda v:v['argv'].append('-fake'),'argv differs'),
    (lambda v:v['files'].pop('source_patch'),'required file'),
    (lambda v:v.update(binary_sha256='0'*64),'binary hash'),
    (lambda v:v.update(wait_status=None),'compiler status'),
    (lambda v:v.update(returncode=True),'compiler status'),
])
def test_missing_or_inconsistent_native_observations_stop(tmp_path,monkeypatch,change,reason):
    h=f.Harness(tmp_path,monkeypatch);ledger=h.start();receipt,_=native(h,ledger)
    value=json.loads(receipt.read_text());change(value);f.save(receipt,value)
    with pytest.raises((ValueError,KeyError),match=reason):m.load_native(receipt)


def test_changed_source_file_rejected_instead_of_rehashed(tmp_path,monkeypatch):
    h=f.Harness(tmp_path,monkeypatch);ledger=h.start();receipt,attempt=native(h,ledger)
    f.m.check_file(attempt['generated_source']).write_text('different source\n')
    with pytest.raises(ValueError,match='immutable input changed'):m.load_native(receipt)


def test_successful_device_compile_report_failure_is_not_resource_retry(tmp_path,monkeypatch):
    h=f.Harness(tmp_path,monkeypatch);ledger=h.start();receipt,attempt=native(h,ledger)
    attempt['resources']=None
    with pytest.raises(ValueError,match='report failure stops'):assemble(h,ledger,receipt,attempt)


def test_no_automatic_archive_or_occupancy_evidence(tmp_path,monkeypatch):
    h=f.Harness(tmp_path,monkeypatch);ledger=h.start();receipt,attempt=native(h,ledger,mt=168,occupancy=None)
    with pytest.raises(ValueError,match='archive inventory'):
        m.assemble(receipt,ledger,stage_sources=[],resources=f.m.check_file(attempt['resources']))
    assert m.OCCUPANCY_QUERY_CONTRACT['execution']=='actual_runtime_query_only'
    assert assemble(h,ledger,receipt,attempt)['occupancy'] is None


def test_wrong_pin_or_new_producer_cannot_scope_old_origin(tmp_path,monkeypatch):
    h=f.Harness(tmp_path,monkeypatch);ledger=h.start();receipt,attempt=native(h,ledger)
    wrong=copy.deepcopy(ledger.pin.data);wrong['mx_decode_v2']['qkv']['prefetch']=8
    changed=f.m.replace(ledger,pin=f.m.FrozenJSON.freeze(wrong))
    with pytest.raises(ValueError,match='effective pin'):assemble(h,changed,receipt,attempt)
    # A real producer change invalidates the original native hash before scoping.
    h.producer.write_text('a new producer requires a new native origin')
    with pytest.raises(ValueError,match='immutable input changed'):assemble(h,ledger,receipt,attempt)


def test_actual_native_writer_to_ledger_without_gpu(tmp_path,monkeypatch):
    import shutil
    import subprocess
    driver=os.environ.get('TILEMEGA_MX5_NATIVE_RECEIPT_TEST_DRIVER')
    if not driver:
        pytest.skip('set the CPU native receipt unit-test executable for cross-language synthetic test')
    h=f.Harness(tmp_path,monkeypatch);ledger=h.start()
    template=tmp_path/'synthetic-device-bytes';template.write_bytes(f.elf_fixture())
    # This executable is a CPU file-producing fixture; never cucc or device code.
    h.compiler.write_text('#!/usr/bin/env python3\nimport pathlib,sys\n'
        + 'pathlib.Path(sys.argv[sys.argv.index("-o")+1]).write_bytes(pathlib.Path('
        + repr(str(template))+').read_bytes())\n'
        + 'print('+repr(f.mir_fixture(228,0,0))+',end="")\n')
    h.compiler.chmod(0o700)
    attempt_path=h.attempt(ledger,mt=228,occupancy=None)
    attempt=json.loads(attempt_path.read_text());out=f.m.check_file(attempt['binary'])
    for name,suffix in [('resolved_pin','.mx5_pin.json'),('source_snapshot','.source.json'),
                        ('source_patch','.source.json.patch'),('command','.build_command.txt')]:
        shutil.copyfile(f.m.check_file(attempt[name]),str(out)+suffix)
    out.unlink();f.m.check_file(attempt['log']).unlink()
    result=subprocess.run([driver,'--capture-synthetic',str(out),str(f.m.check_file(attempt['pin'])),str(h.producer)],check=False)
    assert result.returncode==0
    resources=Path(str(out)+'.resources.json');f.save(resources,f.m.inspect(out,Path(str(out)+'.ptxas.log'),576))
    scoped=m.assemble(str(out)+'.compile_receipt.json',ledger,stage_sources=h.sources,resources=resources)
    path=tmp_path/'actual-native-scoped.json';m.write_exclusive(path,scoped)
    actual=ledger.append_attempt(path)
    assert actual.attempts[-1].data['verified']['classification']=='resource_failure'
    assert actual.attempts[-1].data['verified']['kernel']['registers']==228


def test_existing_attempt_requires_explicit_retry_authorization_before_next_receipt(tmp_path,monkeypatch):
    h=f.Harness(tmp_path,monkeypatch);ledger=h.start();receipt,attempt=native(h,ledger)
    path=tmp_path/'attempt-scope.json';m.write_exclusive(path,assemble(h,ledger,receipt,attempt))
    failed=ledger.append_attempt(path)
    with pytest.raises(ValueError,match='another retry authorization'):
        assemble(h,failed,receipt,attempt)
