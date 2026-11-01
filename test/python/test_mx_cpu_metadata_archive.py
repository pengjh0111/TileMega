import importlib.util
import json
from pathlib import Path

import pytest

SPEC=importlib.util.spec_from_file_location('cpu_metadata_archive',Path(__file__).resolve().parents[2]/'scripts/mx_cpu_metadata_archive.py')
archive=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(archive)


def fixture(tmp_path):
    repository=tmp_path/'repository';folder=repository/'runs/closed/rank0000001';folder.mkdir(parents=True)
    values={'CPU_RESOURCE_RESULT.json':json.dumps({'status':'resource_rejected'}).encode(),
            'LOSSLESS_CPU_SOLID_ARCHIVE.json':b'{"preserved_previous_index": true}\n',
            'decode/receipt.json':b'original\x00metadata\nwith actual newlines\n'}
    for name,data in values.items():
        p=folder/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    return repository,folder,values


def test_closed_metadata_round_trip_and_live_exclusion(tmp_path):
    repository,folder,values=fixture(tmp_path)
    (folder/'queries').mkdir()
    assert not archive.eligible(folder)
    (folder/'queries').rmdir()
    receipt=archive.pack([folder],repository,tmp_path/'archives')
    assert receipt['roundtrip_verified']
    assert all(not(folder/name).exists()for name in values)
    archive.restore(folder)
    assert all((folder/name).read_bytes()==data for name,data in values.items())


def test_changed_source_is_not_removed(tmp_path,monkeypatch):
    repository,folder,values=fixture(tmp_path)
    original=archive.tarfile.TarFile.addfile;changed=False
    def addfile(self,*args,**kwargs):
        nonlocal changed
        result=original(self,*args,**kwargs)
        if not changed and args[0].name.endswith('/decode/receipt.json'):
            (folder/'decode/receipt.json').write_bytes(b'new concurrent data');changed=True
        return result
    monkeypatch.setattr(archive.tarfile.TarFile,'addfile',addfile)
    with pytest.raises(AssertionError):archive.pack([folder],repository,tmp_path/'archives')
    assert not(folder/archive.MARKER).exists()
    assert (folder/'CPU_RESOURCE_RESULT.json').read_bytes()==values['CPU_RESOURCE_RESULT.json']
    assert (folder/'decode/receipt.json').read_bytes()==b'new concurrent data'


def test_corrupt_archive_refuses_restoration(tmp_path):
    repository,folder,values=fixture(tmp_path)
    receipt=archive.pack([folder],repository,tmp_path/'archives');p=Path(receipt['archive']['path']);p.write_bytes(p.read_bytes()+b'corrupted')
    with pytest.raises(AssertionError):archive.restore(folder)
    assert all(not(folder/name).exists()for name in values)
