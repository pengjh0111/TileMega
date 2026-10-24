#!/usr/bin/env python3
"""Check actual repository exports without CUDA or historical artifact inputs."""
import hashlib
import json
import lzma
import os
from pathlib import Path
import subprocess
import sys
import tempfile

root=Path(__file__).resolve().parents[2]
inputs=root/'configs/serving/sm90d1/inputs'
manifest=json.loads((inputs/'manifest.json').read_text())
scratch=root/'runs/sm90r2/workload-binding-tests'
scratch.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(dir=scratch) as directory:
    out=Path(directory)
    command=[sys.argv[1],str(root/'configs/targets/sm_90_h800.json')]
    for model,batch in [('qwen3_0_6b',16),('llama3_2_1b',1)]:
        packed=inputs/model/'decode.export.json.xz'
        expected=manifest[str(packed.relative_to(root))]
        blob=packed.read_bytes()
        if hashlib.sha256(blob).hexdigest()!=expected['compressed_sha256']:
            raise ValueError('repository packed export changed')
        raw=lzma.decompress(blob)
        if hashlib.sha256(raw).hexdigest()!=expected['raw_sha256']:
            raise ValueError('repository export input changed')
        export=out/f'{model}.json';export.write_bytes(raw)
        coordinates=root/f'configs/serving/sm90r2/development/isolation_abi_v1/{model}_b{batch}.tmexec.json'
        command += [str(export),str(coordinates),str(batch)]
    subprocess.run(command+[str(out)],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
