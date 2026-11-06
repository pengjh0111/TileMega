#!/usr/bin/env python3
"""DNN3 §4 gate 1: build every F3 forward plan on the CPU and check its stages.

Serializes each export once per batch into the bridge the plan reader consumes,
then runs `tilemega compile --dump-plan` for both ABIs. No GPU, no coordinates.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/root/TileMega/runs/r16/env/tensorrt-dnn/bin/python'
EXTRA_PATH = '/root/TileMega/runs/r16/env/dnn-export/lib/python3.10/site-packages'
BATCHES = (1, 2, 4, 8, 16)
# Appendix C: the stage count each model's F3 plan must have.
EXPECTED_STAGES = {'resnet18': 24, 'resnet50': 57, 'vgg16': 22, 'bert': 63}
EXPORTS = {
    'resnet18': Path('/root/TileMega-dnn1/runs/dnn1/exports/resnet18'),
    'bert': Path('/root/TileMega-dnn1/runs/dnn1/exports/bert'),
    'resnet50': ROOT / 'runs/dnn3/exports/resnet50',
    'vgg16': ROOT / 'runs/dnn3/exports/vgg16',
}

BRIDGE_SOURCE = r'''
import json, sys, torch
from pathlib import Path
from tilemega.export_bridge import serialize
archive, batch, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
if 'bert' in archive:
    import transformers.modeling_outputs
program = torch.export.load(archive)
symbols = {str(node.meta['val'].shape[0]): batch for node in program.graph.nodes
           if node.op == 'placeholder' and isinstance(node.meta.get('val'), torch.Tensor)
           and node.meta['val'].ndim and not isinstance(node.meta['val'].shape[0], int)}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(serialize(program, shape_bindings=symbols), indent=1) + '\n')
print(out)
'''


def bridge(model, batch, folder):
    path = folder / f'{model}_b{batch}.bridge.json'
    if path.is_file():
        return path
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='',
                       PYTHONPATH=os.pathsep.join([str(ROOT / 'python'), EXTRA_PATH]),
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    archive = EXPORTS[model].resolve() / 'exported_program.pt2'
    done = subprocess.run([PYTHON, '-c', BRIDGE_SOURCE, str(archive), str(batch), str(path)],
                          capture_output=True, text=True, env=environment, cwd=ROOT)
    if done.returncode:
        raise RuntimeError(f'bridge failed for {model} b{batch}: {done.stderr[-2000:]}')
    return path


def dump(binary, source, model, batch, abi, folder):
    out = folder / f'{model}_b{batch}.{abi}.plan.json'
    done = subprocess.run([str(binary), 'compile', str(source), '/dev/null',
                           '--emit', 'serving', '--serving', 'forward', '--frontend', 'dnn',
                           '--batch', str(batch), '--abi', abi, '--dump-plan', str(out)],
                          capture_output=True, text=True, cwd=ROOT,
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    return out, done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path,
                        default=ROOT / 'runs/dnn3/build/main/tools/tilemega')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/dnn3/plan_gate')
    parser.add_argument('--models', default=','.join(EXPECTED_STAGES))
    parser.add_argument('--abis', default='f3')
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=True)
    rows, failures = [], 0
    for model in options.models.split(','):
        for batch in BATCHES:
            source = bridge(model, batch, options.out)
            for abi in options.abis.split(','):
                path, done = dump(options.binary, source, model, batch, abi, options.out)
                row = dict(model=model, batch=batch, abi=abi, status='pass')
                if done.returncode or not path.is_file():
                    row.update(status='fail', error=(done.stderr or done.stdout)[-700:])
                    failures += 1
                else:
                    plan = json.loads(path.read_text())
                    row.update(stages=plan['stages'], gemms=plan['gemms'],
                               activation_gib=round(plan['activation_bytes'] / 2**30, 3),
                               weight_gib=round(plan['weight_bytes'] / 2**30, 3),
                               modes={m: sum(1 for s in plan['stage_rows'] if s['mode'] == m)
                                      for m in ('normal', 'gather', 'swapab')},
                               bodies={b: sum(1 for s in plan['stage_rows'] if s['body'] == b)
                                       for b in sorted({s['body'] for s in plan['stage_rows']})})
                    want = EXPECTED_STAGES.get(model)
                    if abi == 'f3' and want is not None and plan['stages'] != want:
                        row.update(status='stage_count_differs', expected=want)
                        failures += 1
                    # U6: a plan whose activations exceed 60 GiB stops the round.
                    if plan['activation_bytes'] > 60 * 2**30:
                        row.update(status='activation_budget_exceeded')
                        failures += 1
                rows.append(row)
                print(json.dumps(row), flush=True)
    (options.out / 'gate.json').write_text(json.dumps(
        dict(schema=1, failures=failures, rows=rows), indent=2) + '\n')
    print(json.dumps(dict(cells=len(rows), failures=failures)))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
