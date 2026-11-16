#!/usr/bin/env python3
"""Write the dnn3-v1 TRT/torch.compile queue specs and queues; launches nothing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / 'python')]
import dnn1_baselines as base
from dnn1_aitemplate_cell import AFFINITY

ROOT = base.ROOT
TVM_ROOT = Path('/root/TileMega-dnn1/runs/dnn1/resume_tvm/typed_runner_repair_v1/tvm')
# One model per card keeps each model's batch scaling on a single device.
DEFAULT_SPLIT = {2: 'resnet50', 3: 'vgg16'}


def spec_for(workload, gpu, model, exports):
    return dict(schema=1, workload_id=workload['workload_id'], models=workload['models'],
        batches=workload['batches'], arms=workload['arms'], device_file=workload['device_file'],
        output_stem=workload['output_stem'], correctness_policy=workload['correctness_policy'],
        gpu=gpu, cells=[dict(model=model, batch=batch) for batch in workload['batches']],
        systems=['tensorrt', 'torchcompile'], parallel=True, cpu_affinity=AFFINITY[gpu],
        python=str(ROOT / 'scripts/dnn1_supervised_python.py'), exports_root=str(exports),
        onnx_preparation=str(exports / 'preparation_dnn3_onnx_v1.json'),
        tvm_root=str(TVM_ROOT), tvm_selfcheck=str(TVM_ROOT / 'selfcheck.json'),
        inductor_cache=str(ROOT / 'runs/dnn3/inductor'), env=dict(TILEMEGA_DNN1_TVM_ROOT=str(TVM_ROOT)),
        libraries={}, logical_timeout_s=3600, cell_timeout_s=86400, needs_free_mib=8192,
        scheduling_amendment='parent 2026-11-07: four H800 cards; Phase B on cards 2 and 3 beside kernel work on 0 and 1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workload', type=Path, default=ROOT / 'configs/dnn/dnn3_workload.json')
    parser.add_argument('--exports', type=Path, default=ROOT / 'runs/dnn3/exports')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/dnn3/baselines')
    parser.add_argument('--refresh-prepared', action='store_true',
                        help='archive and refresh queues that have never started')
    options = parser.parse_args()
    workload = base.read(options.workload)
    for gpu, model in DEFAULT_SPLIT.items():
        folder = options.out.resolve() / f'gpu{gpu}_{model}'
        spec_path = folder / 'input.json'
        base.write(spec_path, spec_for(workload, gpu, model, options.exports.resolve()))
        base.queue(argparse.Namespace(spec=spec_path, out=folder / 'run',
                                      refresh_prepared=options.refresh_prepared))
        base.preflight(argparse.Namespace(run_root=folder / 'run'))
    print(json.dumps(dict(status='prepared_only', executed_gpu=False)))


if __name__ == '__main__':
    raise SystemExit(main())
