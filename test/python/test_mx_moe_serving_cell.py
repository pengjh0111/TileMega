"""C550 MoE script routing reaches native control builds without legacy flags."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def environment(tmp_path):
    script = tmp_path / 'scripts/mx_serving_cell.sh'
    script.parent.mkdir()
    shutil.copy2(ROOT / 'scripts/mx_serving_cell.sh', script)
    bridge = tmp_path / 'runs/mx1/export/moe_decode/bridge.json'
    bridge.parent.mkdir(parents=True)
    bridge.write_text('{"expert": true}')
    pin = tmp_path / 'control.json'
    pin.write_text('{"mx_moe_v2":{"purpose":"manual"}}')
    compiler = tmp_path / 'compiler.py'
    compiler.write_text('''#!/usr/bin/env python3
import json,os,pathlib,sys
with open(os.environ['COMMAND_LOG'],'a')as f:f.write(json.dumps(sys.argv[1:])+'\\n')
out=pathlib.Path(sys.argv[3]);out.write_bytes(b'control')
pathlib.Path(str(out)+'.mx5_pin.json').write_text('{}')
''')
    compiler.chmod(0o755)
    env = dict(os.environ, CELLS=str(tmp_path / 'cells'), SOLVE_BIN=str(compiler), PIN=str(pin),
        SELECTION='predicted', COMMAND_LOG=str(tmp_path / 'commands.jsonl'),
        MX5_RESOLVED_V2='', CUDA_VISIBLE_DEVICES='', MACA_VISIBLE_DEVICES='')
    return script, env


def commands(env):
    return [json.loads(line) for line in Path(env['COMMAND_LOG']).read_text().splitlines()]


def test_moe_decode_uses_native_and_builds_a_watchdog_twin(tmp_path):
    script, env = environment(tmp_path)
    p = subprocess.run([str(script), 'solve', 'moe', '16', 'decode'], env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    timed, checked = commands(env)
    for argv in (timed, checked):
        assert argv[1] == 'runs/mx1/export/moe_decode/bridge.json'
        assert argv[argv.index('--frontend') + 1] == 'decoder'
        assert argv[argv.index('--batch') + 1] == '16'
        assert '--solve' not in argv and '--forward-executor' not in argv
    assert timed[timed.index('--watchdog') + 1] == '0'
    assert checked[checked.index('--watchdog') + 1] == '1'
    assert checked[checked.index('--evaluate-configs') + 1].endswith('/decode/plan.so.mx5_pin.json')


def test_moe_bridge_can_come_from_the_campaign_export_root(tmp_path):
    script, env = environment(tmp_path)
    root = tmp_path / 'mx6_export'
    destination = root / 'moe_decode/bridge.json'
    destination.parent.mkdir(parents=True)
    shutil.move(tmp_path / 'runs/mx1/export/moe_decode/bridge.json', destination)
    env['EXPORT_ROOT'] = str(root)
    p = subprocess.run([str(script), 'solve', 'moe', '1', 'decode'], env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert all(args[1] == str(destination) for args in commands(env))


@pytest.mark.parametrize('watchdog,directory', [('0', 'prefill'), ('1', 'prefill_w')])
def test_moe_prompt_uses_decode_bridge_and_prompt64(tmp_path, watchdog, directory):
    script, env = environment(tmp_path)
    env['WATCHDOG'] = watchdog
    p = subprocess.run([str(script), 'solve', 'moe', '1', 'prefill'], env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    (argv,) = commands(env)
    assert argv[1] == 'runs/mx1/export/moe_decode/bridge.json'
    assert argv[2].endswith('/' + directory + '/plan.so')
    for flag, value in [('--serving', 'prefill'), ('--mx-prompt', '1'), ('--prompt-len', '64'), ('--watchdog', watchdog)]:
        assert argv[argv.index(flag) + 1] == value


@pytest.mark.parametrize('change', [dict(PIN=''), dict(SELECTION='measure'), dict(MX5_RESOLVED_V2='dense-protocol')])
def test_unready_moe_solver_never_runs_the_legacy_script_branch(tmp_path, change):
    script, env = environment(tmp_path)
    env.update(change)
    p = subprocess.run([str(script), 'solve', 'moe', '1', 'decode'], env=env, capture_output=True, text=True)
    assert p.returncode == 2 and 'MoE measured top-four selection is pending' in p.stderr
    assert not Path(env['COMMAND_LOG']).exists()
