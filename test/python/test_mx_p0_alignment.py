"""CPU fixtures only: P0 attributes are not executed/timed-body qualification."""
import copy
import ctypes
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

REPO = Path(__file__).resolve().parents[2]
MAIN = Path(os.environ.get('TILEMEGA_TEST_ROOT', REPO))


def module(name):
    spec = importlib.util.spec_from_file_location('p0_cpu_' + name, REPO / 'experiments/mx5' / (name + '.py'))
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


calibration, table, qualification = module('calibration'), module('stage_table'), module('qualify_campaign')


def artifact(directory, pw=0, resources=(168,616,0,1), status=0):
    directory.mkdir()
    binary = directory / 'CPU_fixture.so'; binary.write_bytes(b'CPU synthetic compiler probe fixture ' + str(pw).encode())
    source_name = 'experiments/mx5/stage_bench.cu'; body = b'// CPU fixture, never compiled or executed\n'
    with tarfile.open(directory / 'sources.tar.gz', 'w:gz') as archive:
        member = tarfile.TarInfo(source_name); member.size = len(body); archive.addfile(member, io.BytesIO(body))
    source_sha = hashlib.sha256(source_name.encode()+b'\0'+body+b'\0').hexdigest()
    metadata = dict(body='gemm',bench=True,returncode=0,tn=16,depth=1,threads=576,pw=pw,
        source_sha256=source_sha,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),base_head='CPU_FIXTURE',
        command=['CPU_fixture_compiler','-DMX5_TEST_TN=16','-DMX5_TEST_DEPTH=1','-DMX5_TEST_THREADS=576',
                 '-DMX5_BENCH_PW='+str(pw),source_name])
    binary.with_suffix('.json').write_text(json.dumps(metadata))
    key=dict(n=32768,k=4096,b=1,tn=16,wn=1,depth=1,threads=576,op=0,norm=1,part=1,prefetch=pw)
    bench=calibration.module('gemm');calls=[]
    class Function:
        def __init__(self,call):self.call=call
        def __call__(self,*args):return self.call(*args)
    def query(argument,*args):
        actual=argument._obj;assert actual.iterations==0 and actual.copies==1
        assert (actual.n,actual.k,actual.b)==(32768,4096,1) and args==(1,0,1,1,None)
        calls.append(dict(iterations=actual.iterations,key=key.copy()))
        for index,value in enumerate(resources):actual.resources[index]=value
        return status
    library=SimpleNamespace(mx5_bench=Function(query),mx5_bench_args_size=Function(lambda:ctypes.sizeof(bench.Args)))
    with patch.dict(os.environ,TILEMEGA_GPU_LOCK_HELD='1'),patch.object(calibration.C,'CDLL',return_value=library):
        result=calibration.query_one('gemm',key,binary,{}, {})
    return key,result,binary,library,calls


def probe(directory,**kwargs):
    key,result,binary,library,calls=artifact(directory,**kwargs)
    reference=calibration.compiler_probe.from_query('gemm',key,result)
    return key,result,reference,binary,library,calls


@pytest.fixture(scope='session')
def domain(tmp_path_factory):
    output=tmp_path_factory.mktemp('p0_cpu_domain')/'domain'
    compiler=shutil.which('c++')
    if not compiler:pytest.skip('host C++ compiler unavailable')
    subprocess.run([compiler,'-std=c++17','-O2','-I'+str(MAIN/'include'),str(REPO/'experiments/mx5/domain.cpp'),'-o',str(output)],check=True)
    return output


def test_nonfitting_P0_derives_actual_primary_with_measured_168_and_strict_primary_check(tmp_path,domain):
    key,result,reference,_,_,calls=probe(tmp_path/'p0')
    assert result['fits'] is False and reference['compiler_probe_only'] is True
    assert reference['resources']['mtreg']==168 and reference['resources']['local_bytes']==616
    assert reference['probe']['iterations']==0 and calls[0]['iterations']==0
    original=copy.deepcopy(reference); key.pop('prefetch')
    inspected=dict(kind='gemm',key=key,first_chunks=16,query_only=True,query_status=0,iterations=0,
                   fits=False,library=result['library'],r_next_reference=reference)
    planned,builds,references=calibration.plan([inspected],domain,tmp_path/'derived')
    assert planned[0]['disposition']=='derived' and planned[0]['key']['prefetch']==13
    assert planned[0]['fits'] is False and references==[original] and reference==original
    assert builds[0]['command'][-1]=='13'
    _,_,primary,library,_=artifact(tmp_path/'primary',pw=13,resources=(210,0,0,1))
    row=copy.deepcopy(planned[0]);row['artifacts'][0]['library']=str(primary)
    with patch.dict(os.environ,TILEMEGA_GPU_LOCK_HELD='1'),patch.object(calibration,'initialize_cuda'),patch.object(calibration.C,'CDLL',return_value=library):
        selected=next(calibration.select([row]))
    assert selected['fits'] is True and selected['selected_resource_step']==0
    assert selected['r_next_reference']['resources']['local_bytes']==616
    # A selected prefix with stack is still ineligible, regardless of its P0 marker.
    _,_,bad,bad_library,_=artifact(tmp_path/'bad_primary',pw=13,resources=(210,4,0,1))
    row['artifacts']=[dict(resource_step=0,prefetch=13,library=str(bad))]
    with patch.dict(os.environ,TILEMEGA_GPU_LOCK_HELD='1'),patch.object(calibration,'initialize_cuda'),patch.object(calibration.C,'CDLL',return_value=bad_library):
        failed=next(calibration.select([row]))
    assert failed['fits'] is False and failed['selected_resource_step'] is None
    assert failed['selection_status']=='unresolved_resource_failure'
    assert all(a['key']['threads']==576 for a in failed['resource_attempts'])


@pytest.mark.parametrize('field,value', [('query_only',False),('iterations',None),('iterations',1),('query_status',None),('query_status',7)])
def test_missing_or_failed_query_evidence_stops_before_primary(tmp_path,domain,field,value):
    key,result,reference,_,_,_=probe(tmp_path/'p0');reference['probe'][field]=value
    inspected=dict(kind='gemm',key=key,first_chunks=16,query_only=True,query_status=0,fits=False,r_next_reference=reference)
    with patch.object(calibration.subprocess,'run') as run:
        with pytest.raises(RuntimeError,match='compiler probe'):
            calibration.plan([inspected],domain,tmp_path/'unused')
        run.assert_not_called()
    with pytest.raises(table.TableError,match='query_only/iterations=0/status=0'):
        table.validate_reference(reference)


@pytest.mark.parametrize('change', ['null_attribute','wrong_source','wrong_binary','changed_archive','nonzero_prefetch'])
def test_probe_resource_source_and_binary_proofs_remain_strict(tmp_path,change):
    _,_,reference,_,_,_=probe(tmp_path/'p0')
    if change=='null_attribute':reference['resources']['mtreg']=None
    elif change=='wrong_source':reference['provenance']['source_sha256']='f'*64
    elif change=='wrong_binary':reference['provenance']['binary_sha256']='e'*64
    elif change=='changed_archive':Path(reference['probe']['source_archive']['path']).write_bytes(b'changed CPU fixture')
    else:reference['key']['prefetch']=1
    with pytest.raises(table.TableError):table.validate_reference(reference)


def test_reference_attributes_may_be_nonresident_but_diagnostic_fits_cannot_lie(tmp_path):
    _,_,reference,_,_,_=probe(tmp_path/'p0',resources=(268,616,16,0))
    _,_,_,attributes=table.validate_reference(reference)
    assert attributes==dict(mtreg=268,local_bytes=616,static_smem_bytes=16,resident_ctas=0)
    reference['fits']=True
    with pytest.raises(table.TableError,match='diagnostic fits'):table.validate_reference(reference)


def timing_row(reference,pw=13):
    key=dict(reference['key'],prefetch=pw)
    return dict(key=key,round=1,executor='mx_static_stage_bench',implementation='mx5_gemm_v2',median_ns=100,p90_ns=150,
        iterations=200,copies=8,working_set_bytes=256*1024**2,mtreg=210,local_bytes=0,static_smem_bytes=0,resident_ctas=1,
        threads=576,source_sha256=reference['provenance']['source_sha256'],binary_sha256='b'*64,base_head='CPU_FIXTURE',
        prefetch=pw,r_next_reference=reference,test_fixture='CPU synthetic times, never an experiment result')


def test_aggregate_and_coverage_qualifier_accept_only_actual_fitting_rows(tmp_path):
    _,_,reference,_,_,_=probe(tmp_path/'p0');row=timing_row(reference);runs=[]
    for number in (1,2,3):
        path=tmp_path/('round%d.jsonl'%number);path.write_text(json.dumps(dict(row,round=number))+'\n');runs.append(path)
    document=table.aggregate(runs,card='CPU synthetic fixture; not a device measurement')
    assert document['rows'][0]['r_next']==168
    assert document['rows'][0]['r_next_reference']['resources']['local_bytes']==616
    checked=dict(kind='gemm',key=row['key'],provenance=table.identity(row),resources={f:row[f] for f in table.RESOURCE_FIELDS},
                 fits=True,query_only=True,query_status=0,selected_resource_step=0,r_next_reference=reference)
    assert qualification.checked_identity(checked)==('gemm',row['key'])
    checked['resources']['local_bytes']=4
    with pytest.raises(ValueError,match='resource rejection'):qualification.checked_identity(checked)
    for changes in (dict(local_bytes=4),dict(mtreg=228),dict(static_smem_bytes=16),dict(resident_ctas=0)):
        for number,path in enumerate(runs,1):path.write_text(json.dumps(dict(row,round=number,**changes))+'\n')
        with pytest.raises(table.TableError):table.aggregate(runs,card='CPU fixture')


def test_old_unmarked_nonfitting_reference_is_not_silently_upgraded(tmp_path):
    key,result,reference,_,_,_=probe(tmp_path/'p0')
    for field in ('compiler_probe_only','probe','fits'):reference.pop(field)
    with pytest.raises(table.TableError,match='resource rejection'):table.validate_reference(reference)
    with pytest.raises(RuntimeError,match='legacy reference'):
        calibration.selection_reference('gemm',key,reference)
    with pytest.raises(ValueError,match='query_only/iterations=0/status=0'):
        calibration.compiler_probe.from_query('gemm',key,{k:v for k,v in result.items() if k!='iterations'})


def test_inspection_marks_new_reference_without_rewriting_input_row(tmp_path):
    key,result,_,_,_=artifact(tmp_path/'p0');inventory=dict(kind='gemm',key=dict(key),first_chunks=16,uses=[dict(model='CPU_FIXTURE',stage='qkv')])
    inventory['key'].pop('prefetch');before=copy.deepcopy(inventory)
    with patch.object(calibration,'initialize_cuda'),patch.object(calibration,'baseline_libraries',return_value={calibration.library_key('gemm',key):Path(result['library'])}),patch.object(calibration,'query_one',return_value=result):
        inspected=next(calibration.inspect([inventory],tmp_path))
    assert inventory==before and inspected['fits'] is False and inspected['iterations']==0
    assert inspected['r_next_reference']['compiler_probe_only'] is True
    assert inspected['r_next_reference']['resources']['local_bytes']==616


def test_actual_failed_query_keeps_null_attributes_and_cannot_be_retagged(tmp_path):
    key,result,_,_,_=artifact(tmp_path/'p0',status=7)
    assert result['query_status']==7 and result['fits'] is False and result['resources']['mtreg'] is None
    with pytest.raises(ValueError,match='status=0'):
        calibration.compiler_probe.from_query('gemm',key,result)
