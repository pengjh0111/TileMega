"""CPU protocol rejection tests; no builds, real model claims or GPU execution."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('mx5_watchdog_pair_runner', ROOT/'experiments/mx5/watchdog_pair_runner.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def protocol(tmp_path):
    domain = tmp_path/'CPU_SYNTHETIC_DOMAIN'
    domain.write_text('not an executable\n')
    p = tmp_path/'protocol.json'
    value = dict(kind='mx5_watchdog_pair_top4_protocol',schema_version=1,
                 domain_tool=m.record(domain),observer_tool=m.record(Path(m.pairs.__file__)),
                 runner_tool=m.record(Path(m.__file__)),plans=[{'candidate_id':i}for i in range(4)])
    return p,domain,value


@pytest.mark.parametrize('mutation,match',[
    ('schema','wrong paired protocol'),('domain','paired domain'),('helper','helper bytes'),
    ('missing','four original IDs'),('duplicate','four original IDs'),('reorder','four original IDs')])
def test_reject_unbound_or_incomplete_protocol_before_native_work(protocol, mutation, match):
    path,domain,value = protocol
    if mutation == 'schema':value['schema_version'] = 2
    elif mutation == 'domain':value['domain_tool']['sha256'] = '0'*64
    elif mutation == 'helper':value['observer_tool']['sha256'] = '0'*64
    elif mutation == 'missing':value['plans'].pop()
    elif mutation == 'duplicate':value['plans'][3]['candidate_id'] = 2
    else:value['plans'][0],value['plans'][1] = value['plans'][1],value['plans'][0]
    path.write_text(json.dumps(value)+'\n')
    with pytest.raises(ValueError,match=match):m.ready(path,domain)
