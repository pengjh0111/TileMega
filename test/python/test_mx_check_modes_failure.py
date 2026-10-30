"""A failed C-2 child must stop the pipeline before HF reference checking."""
import os
from pathlib import Path
import subprocess


def test_c2_exit_status_survives_output_filter(tmp_path):
    root = Path(__file__).resolve().parents[2]
    commands = tmp_path / 'commands'
    commands.mkdir()
    marker = tmp_path / 'unexpected_hf_check'
    python = commands / 'python'
    python.write_text('''#!/bin/bash
case "$*" in
  *tilemega.serving.check_modes*) echo "C2 failed inside the native runtime"; exit 23;;
  *tilemega.serving.hf_check*) touch "$UNEXPECTED_HF_CHECK"; echo '{"pass":true,"gap_le_0_5_ratio":1,"max_gap":0,"gap_zero_ratio":1}';;
  *) exit 99;;
esac
''')
    python.chmod(0o755)
    result = subprocess.run(['bash', str(root / 'scripts/mx_check_modes.sh'),
        'model', 'prefill.so', 'decode.so', '2', '512', str(tmp_path / 'out')],
        cwd=root, env=dict(os.environ, PATH=str(commands) + ':' + os.environ['PATH'],
            UNEXPECTED_HF_CHECK=str(marker)), capture_output=True, text=True)
    assert result.returncode == 23, result.stderr
    assert 'C2 failed inside the native runtime' in result.stdout
    assert not marker.exists()
