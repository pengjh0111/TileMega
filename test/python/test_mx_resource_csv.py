"""Evidence CSV parts preserve fields and refuse damaged or mislabelled records."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'mx_resource_csv', Path(__file__).resolve().parents[2] / 'scripts/mx_resource_csv.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
FIELDS = ['checkpoint', 'record_type', 'model', 'batch', 'program_version', 'purpose',
          'actual_query', 'evidence', 'status', 'resource_scope', 'binary_path',
          'binary_sha256', 'mtreg', 'stack_bytes', 'coordinates_json']


def fixture(path):
    with path.open('w', newline='') as output:
        writer = csv.DictWriter(output, FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerow(dict(checkpoint='CP1', record_type='compiled_plan', model='old'))
    return path.read_bytes()


def rejection(number=0):
    return dict(checkpoint='CP4', record_type='compiled_plan', model='fixture', batch=16,
                program_version=3, purpose='resource_probe', actual_query=False,
                evidence='verified', status='CPU_resource_rejected_never_launched',
                resource_scope='full metadata', binary_path=f'runs/path with spaces,{number}/plan.so',
                binary_sha256=hashlib.sha256(str(number).encode()).hexdigest(), mtreg=255,
                stack_bytes=128 + number, coordinates_json=json.dumps(dict(TN=128, D=4)))


def append(path, index):
    with path.open('a', newline='') as output:
        csv.DictWriter(output, FIELDS, lineterminator='\n').writerow(index)


def test_full_fields_round_trip_and_original_prefix(tmp_path):
    main = tmp_path / 'resources.csv'
    before = fixture(main)
    rows = [rejection(n) for n in range(250)]
    index = MODULE.seal_rejections(main, FIELDS, rows, 'CP4')
    append(main, index)
    assert main.read_bytes().startswith(before)
    expanded = list(MODULE.resource_rows(main))
    assert len(expanded) == 251
    assert expanded[1:] == [{key: str(row[key]) for key in FIELDS} for row in rows]


def test_tampered_part_is_rejected(tmp_path):
    main = tmp_path / 'resources.csv'
    fixture(main)
    index = MODULE.seal_rejections(main, FIELDS, [rejection()], 'CP4')
    append(main, index)
    part = tmp_path / json.loads(index['resource_scope'])['path']
    payload = bytearray(part.read_bytes())
    payload[-5] ^= 1
    part.write_bytes(payload)
    with pytest.raises(ValueError, match='compressed SHA256 differs'):
        list(MODULE.resource_rows(main))


def test_miscounted_part_is_rejected(tmp_path):
    main = tmp_path / 'resources.csv'
    fixture(main)
    index = MODULE.seal_rejections(main, FIELDS, [rejection()], 'CP4')
    metadata = json.loads(index['resource_scope'])
    metadata['rows'] += 1
    index['resource_scope'] = json.dumps(metadata)
    append(main, index)
    with pytest.raises(ValueError, match='row count differs'):
        list(MODULE.resource_rows(main))


def test_queried_row_stays_in_main_file(tmp_path):
    main = tmp_path / 'resources.csv'
    before = fixture(main)
    row = rejection()
    row['actual_query'] = True
    with pytest.raises(ValueError, match='queried/measured'):
        MODULE.seal_rejections(main, FIELDS, [row], 'CP4')
    assert main.read_bytes() == before
    assert not list(tmp_path.glob('*.csv.gz'))


def test_nested_part_is_rejected_before_index_publication(tmp_path):
    main = tmp_path / 'resources.csv'
    before = fixture(main)
    row = rejection()
    row['record_type'] = MODULE.PART_TYPE
    with pytest.raises(ValueError, match='nested resource CSV parts'):
        MODULE.seal_rejections(main, FIELDS, [row], 'CP4')
    assert main.read_bytes() == before
