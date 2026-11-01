"""Preserve guard exit/output while accounting through host clock jumps."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.mark.parametrize('code,wall_times', [(0, [10_000_000_000, 1_000_000_000]),
                                         (7, [1_000_000_000, 10_000_000_000])])
def test_clock_jump_keeps_monotonic_duration_and_child_result(tmp_path, monkeypatch, capfd, code, wall_times):
    path = Path(__file__).resolve().parents[2] / 'scripts/mx_job_duration.py'
    spec = importlib.util.spec_from_file_location('mx_job_duration_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    times = iter(wall_times)
    monkeypatch.setattr(module.time, 'time_ns', lambda: next(times))
    output = tmp_path / 'duration.json'
    command = [sys.executable, '-c', f'import time; time.sleep(.02); print("guard output"); raise SystemExit({code})']
    assert module.run(command, output) == code
    assert capfd.readouterr().out == 'guard output\n'
    record = json.loads(output.read_text())
    assert record['returncode'] == code and record['argv'] == command
    assert record['elapsed_seconds'] >= .02 and record['wall_clock_discontinuity']
    assert record['monotonic_end_ns'] >= record['monotonic_start_ns']
