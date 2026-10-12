"""Run both numerical suites and compare every partial context and LSE bit."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory() as temporary:
    outputs=[]
    for index,executable in enumerate(sys.argv[1:]):
        path=Path(temporary)/f'{index}.bin'
        subprocess.run([executable],env=dict(os.environ,TILEMEGA_ATTENTION_TEST_DUMP=str(path)),check=True)
        outputs.append(path.read_bytes())
    if len(outputs)!=2 or not outputs[0] or outputs[0]!=outputs[1]:
        raise SystemExit('FAIL pvswap/pvswap2 partial context or LSE differs')
    print('PASS pvswap/pvswap2 bitwise '+hashlib.sha256(outputs[0]).hexdigest())
