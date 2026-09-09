#!/usr/bin/env python3
"""A0: reuse the common-FP32 method; GPU capture requires an explicit flag."""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import torch

from export_real import _load_probe, make_stack
from run_depth import digest, noise_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, required=True)
    parser.add_argument('--capture-once', action='store_true')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    work = here/'raw/work/condition9'
    fixture = work/'export/fixture'
    out = here/'condition9_noise'
    out.mkdir(exist_ok=True)
    log = out/'capture.txt'
    dump = work/'dump_A0'
    if args.capture_once:
        if log.exists():
            raise RuntimeError('capture already exists; refusing a second GPU process')
        dump.mkdir(exist_ok=True)
        result = subprocess.run([str(work/'k8'), str(fixture)], capture_output=True,
            text=True, timeout=120, env=dict(os.environ, TILEMEGA_WARMUP='5',
            TILEMEGA_REPEAT='11', TILEMEGA_DUMP_BUFFERS=str(dump)))
        log.write_text(result.stdout+result.stderr)
        (out/'capture_manifest.json').write_text(json.dumps({
            'binary_sha256': digest(work/'k8'), 'returncode': result.returncode,
            'command': [str(work/'k8'), str(fixture)], 'warmup': 5, 'repeat': 11,
            'authorization': 'user approved one original-binary output capture'}, indent=2)+'\n')
    old = (here/'condition9/k8_r0.txt').read_text()
    new = log.read_text()
    for pattern in (r'E2E_HASH .*', r'E2E_DIFF .*'):
        if re.search(pattern, old)[0] != re.search(pattern, new)[0]:
            raise RuntimeError('capture does not reproduce the original failure')
    torch.set_num_threads(args.threads)
    manifest = json.loads((fixture/'manifest.json').read_text())
    program = torch.export.load(work/'export/exported_program.pt2')
    probe = _load_probe(repo/'docs/experiments/V_H/export_probe.py')
    model = make_stack(probe.LlamaLayer, manifest['layers'], manifest['hidden'],
                       manifest['intermediate'], manifest['heads'], manifest['kv_heads'])
    model = model.eval().to(dtype=torch.bfloat16)
    model.load_state_dict(program.state_dict, strict=True)
    def read(path, dtype=torch.bfloat16):
        return torch.frombuffer(bytearray(path.read_bytes()), dtype=dtype)
    inputs = []
    input_hashes = {}
    for spec in program.graph_signature.input_specs:
        if spec.kind.name != 'USER_INPUT': continue
        path = fixture/f'input_{spec.arg.name}.bin'
        shape = ((1,manifest['seq'],manifest['hidden']) if spec.arg.name == 'hidden' else
                 (1,manifest['kv_heads'],manifest['past'],manifest['head_dim']))
        inputs.append(read(path).reshape(shape))
        input_hashes[path.name] = digest(path)
    with torch.no_grad():
        outputs = model(*inputs)
    # Pin the original arithmetic/reference, not merely the same random seed.
    for index, tensor in enumerate(outputs):
        if not torch.equal(tensor.flatten(), read(fixture/f'reference_{index}.bin')):
            raise RuntimeError(f'CPU BF16 golden differs at output {index}; do not change threads to fit')
    reference = copy.deepcopy(model).float()
    with torch.no_grad():
        wide_outputs = reference(*(value.float() for value in inputs))
    buffer = re.search(r'E2E_OUTPUT_DIFF index=0 buffer=(\d+)', new)[1]
    tm = read(dump/f'buffer_{buffer}.bin').double()
    py = outputs[0].flatten().double()
    wide = wide_outputs[0].flatten().double()
    if any(not torch.isfinite(x).all() for x in (tm,py,wide)):
        raise RuntimeError('nonfinite noise-floor input')
    metrics = noise_metrics(py, tm, wide)
    k = metrics[1]['l2_error']/metrics[0]['l2_error']
    result = {'torch_version': torch.__version__, 'threads': torch.get_num_threads(),
              'golden_device': 'cpu', 'original_golden_bitwise': True,
              'elements': tm.numel(), 'input_hashes': input_hashes,
              'output_sha256': digest(dump/f'buffer_{buffer}.bin'),
              'comparisons': metrics, 'k_l2': k,
              'decision': 'criterion_artifact' if k <= 1.003 else
                          'stop_condition9_open' if k > 1.2 else 'unassigned_threshold_gap'}
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
