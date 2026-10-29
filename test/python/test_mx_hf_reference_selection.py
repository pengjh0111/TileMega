"""The user-approved MoE reference override is explicit and reaches HF only."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('experts', [None, 'grouped_mm', 'eager'])
def test_check_modes_reference_override(tmp_path, experts):
    stub = tmp_path / 'python'
    stub.write_text('''#!/usr/bin/env python3
import json,os,sys
with open(os.environ['REFERENCE_ARGV_LOG'],'a') as f:
    f.write(json.dumps(sys.argv[1:])+'\\n')
if sys.argv[2].endswith('hf_check'):
    print(json.dumps(dict(pass_=True, gap_le_0_5_ratio=1, max_gap=0, gap_zero_ratio=1)).replace('pass_', 'pass'))
else:
    print('{}')
''')
    stub.chmod(0o755)
    log = tmp_path / 'arguments.jsonl'
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ['PATH'], REFERENCE_ARGV_LOG=str(log))
    env.pop('MX_MOE_HF_EXPERTS', None)
    if experts:
        env['MX_MOE_HF_EXPERTS'] = experts
    subprocess.run(['bash', str(ROOT / 'scripts/mx_check_modes.sh'), 'model', 'prompt.so', 'decode.so',
        '16', '512', str(tmp_path / 'checks'), 'prompts.json'], cwd=ROOT, env=env, check=True, capture_output=True)
    native, hf = [json.loads(line) for line in log.read_text().splitlines()]
    assert native[1] == 'tilemega.serving.check_modes' and '--experts-implementation' not in native
    assert hf[1] == 'tilemega.serving.hf_check' and '--skip-free-greedy' in hf
    if experts:
        assert hf[hf.index('--experts-implementation') + 1] == experts
    else:
        assert '--experts-implementation' not in hf


@pytest.mark.parametrize('module', ['tilemega.serving.hf_check', 'tilemega.moe.native_routing_check'])
def test_reference_cli_documents_explicit_choices(module):
    result = subprocess.run([sys.executable, '-m', module, '--help'], cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=str(ROOT / 'python'), CUDA_VISIBLE_DEVICES='', MACA_VISIBLE_DEVICES=''),
        capture_output=True, text=True, check=True)
    assert '--experts-implementation {grouped_mm,eager}' in result.stdout
