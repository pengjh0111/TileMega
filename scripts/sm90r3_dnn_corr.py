#!/usr/bin/env python3
"""P-CORR for the native DNN forward (prompt §2.3, §10 N-2).

Runs the DNN numerical gate in N fresh processes on the functional card and
aggregates: every process must pass, in its own child process, against the
same artifact, inputs, export archive and checkpoint, and must exercise both
executor modes. Five processes are used because these programs carry
synchronization mechanisms (the V3 boundary, split-K and row-statistics last
arrivers, the pooling reduction).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/workspace/llm-stack/envs/vllm/bin/python'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    for name in ('library', 'export', 'bridge', 'inputs', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--processes', type=int, default=5)
    p.add_argument('--gpu', type=int, default=5)
    a = p.parse_args()
    if a.gpu == 6:
        raise ValueError('GPU6 is excluded')
    a.out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), TILEMEGA_DEVICE_INDEX=str(a.gpu),
               TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{a.gpu}.lock',
               PYTHONPATH=str(ROOT / 'python'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='8')
    rows = []
    for index in range(a.processes):
        gate = a.out / f'gate-f{index}.json'
        if gate.exists():
            raise FileExistsError('retain prior correctness evidence: ' + str(gate))
        command = ['flock', f'/tmp/tilemega_gpu{a.gpu}.lock', PYTHON, '-m',
                   'tilemega.dnn.check_generated', '--library', str(a.library),
                   '--bridge', str(a.bridge), '--export', str(a.export), '--batch', str(a.batch),
                   '--input-tensors', str(a.inputs), '--out', str(gate)]
        started = time.time()
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        (a.out / f'process-f{index}.log').write_text(result.stdout + result.stderr)
        if result.returncode or not gate.is_file():
            raise RuntimeError(f'process {index} failed; see {a.out}/process-f{index}.log')
        receipt = json.loads(gate.read_text())
        rows.append(dict(process=index, elapsed_s=round(time.time() - started, 3),
                         passed=receipt['passed'], artifact_id=receipt['artifact_id'],
                         gate=str(gate), gate_sha256=sha(gate),
                         modes=sorted({case['mode'] for case in receipt['cases']}),
                         metrics=[dict(mode=case['mode'], name=output['name'],
                                       cosine_min=output['cosine_min'],
                                       max_error=output['max_error'],
                                       bf16_max_error=output['bf16_max_error'])
                                  for case in receipt['cases'] for output in case['outputs']],
                         input_tensors_sha256=receipt['input_tensors_sha256'],
                         archive_sha256=receipt['archive_sha256'],
                         source_checkpoint_sha256=receipt['source_checkpoint_sha256'],
                         bridge_sha256=receipt['bridge_sha256']))
    identity = {key: {row[key] for row in rows} for key in
                ('artifact_id', 'input_tensors_sha256', 'archive_sha256',
                 'source_checkpoint_sha256', 'bridge_sha256')}
    passed = (all(row['passed'] for row in rows) and
              all(len(values) == 1 for values in identity.values()) and
              all({1, 2} <= set(row['modes']) for row in rows) and
              len(rows) == a.processes)
    # Every process must reproduce the same output metrics bit for bit.
    digests = {json.dumps(row['metrics'], sort_keys=True) for row in rows}
    result = dict(schema=1, protocol='P-CORR', scope='native DNN forward gate',
                  model=a.model, batch=a.batch, gpu=a.gpu, processes=a.processes,
                  library=str(a.library), library_sha256=sha(a.library),
                  fresh_processes=len(rows), identical_metrics=len(digests) == 1,
                  passed=passed and len(digests) == 1, rows=rows,
                  driver_sha256=sha(__file__))
    (a.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=1))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
