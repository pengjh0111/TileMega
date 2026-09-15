#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Solve the chain plans from measured sm_120 geometry, never from sm_89 dumps.

`tilemega-place-chain` reads each cell's grid, SM count and CTAs per SM out of a
control run's `E2E_RESOURCE` line, and the plan it emits binds (worker, slot) to
that resident grid.  The dumps `run.sh` uses are a 4090's -- grid 256 over 128
SMs -- so on a Blackwell part the probe has to be taken again here and the
manifest pointed at the local copies (H9).  Round one learned this the expensive
way: EFT tables copied across devices were refused by the host's plan guard.

No trace is available on the target, so `trace_dir` is `-` and the simulator
reports modulo worker-SM assignment rather than borrowing sm_89 observations.
The emitted plan is priced by the cost model on both machines either way; §6.2's
traced weights are a comparison, not the basis of what is emitted.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--realwidth', choices=('0', '1'), default='0')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    out = args.out.resolve()
    old = repo / 'docs/experiments/CHAIN/raw'
    if out == old or old in out.parents:
        parser.error('sm_89 output directories are read-only inputs')
    build = Path(os.environ.get('BUILD_DIR', repo / 'build-portable'))
    nvcc = os.environ.get('CUDACXX', '/usr/local/cuda/bin/nvcc')
    for folder in ('plan', 'log', 'prepare'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    # Mode 5 is the control the probe is taken from, matching how the sm_89
    # dumps were produced; the chain arm's own macro is 0 and is applied by
    # run.sh when it compiles the emitted plan, not here.
    common = ['-std=c++17', '-O2', '-arch=sm_120', '-lineinfo',
              '-DTILEMEGA_EVENT_KAPPA=1', '-DTILEMEGA_PLACEMENT=5',
              '-Xptxas=-v,--warn-on-spills']
    common += [f'-I{repo / path}' for path in (
        'include', 'third_party/cutlass/include',
        'third_party/cutlass/tools/util/include', 'third_party/cutlass/test')]
    rows = []
    models = ['gqa2', 'mha4'] + (['real'] if args.realwidth == '1' else [])
    for seq in (4, 128):
        for model in models:
            if model == 'real':
                work = repo / f'docs/experiments/REALMODEL/raw/work/r2sim_s{seq}'
                source, export, fixture = (work / 'model.cu', work / 'model.json',
                                           work / 'export/fixture')
            else:
                source = repo / f'docs/experiments/PLAN_CONTRACT/legacy_identity/plan/{model}.cu'
                export = repo / f'docs/experiments/SEQSCAN/raw/export/{model}.json'
                fixture = repo / f'docs/experiments/SEQSCAN/raw/fixture/{model}_s{seq}_p3'
            prefix = out / 'prepare' / model
            binary = out / 'prepare' / f'{model}_s{seq}_p5'
            command = [nvcc, *common, str(source), str(build / 'libtilemega.a'),
                       f'-L{Path(nvcc).parent.parent / "lib64"}', '-lcudart',
                       '-o', str(binary)]
            with (out / 'log' / f'prepare_{model}_s{seq}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=log, check=True)
            result = subprocess.run([str(binary), str(fixture)],
                env=dict(os.environ, TILEMEGA_PLACEMENT_BASE_DUMP='1'),
                capture_output=True, text=True,
                timeout=900 if model == 'real' else 300)
            text = result.stdout + result.stderr
            (out / 'prepare' / f'{model}_p5_s{seq}.out').write_text(text)
            if result.returncode or 'RESULT status=PASS' not in text:
                raise RuntimeError(f'control probe failed: {model} seq={seq}')
            if 'E2E_RESOURCE ' not in text or 'E2E_PLACE_BASE ' not in text:
                raise RuntimeError('control probe lacks resource/task graph evidence')
            rows.append([model, seq, 3, source, prefix, '-', export])
    manifest = out / 'manifest.tsv'
    with manifest.open('w') as stream:
        writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
        writer.writerow(['model', 'seq', 'past', 'generated_cu', 'out_prefix',
                         'trace_dir', 'export_json'])
        writer.writerows(rows)
    target = Path(os.environ.get('TARGET_JSON', out / 'sm_120.json'))
    if not target.exists():
        command = [str(build / 'tools/tilemega-calibrate'), '--dtype', 'bf16',
                   '--base', str(repo / 'configs/targets/sm_120.json'),
                   '--repeats', '41', '--out', str(target)]
        with (out / 'log/calibration.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=log, check=True)
    profile = json.loads(target.read_text())
    if profile.get('arch_tag') != 'sm_120':
        raise RuntimeError('target profile must describe sm_120')
    if not profile.get('calibration_by_dtype', {}).get('bf16', {}).get('calibrated'):
        raise RuntimeError('sm120 BF16 target calibration was not accepted')
    # The hop curve is the target's own: on sm_120 it is a ~400 ns floor with no
    # backoff term to remove, which is why chaining is the lever there at all.
    command = [str(build / 'tools/tilemega-place-chain'), str(repo), str(manifest),
               str(out), '--target', str(target),
               '--hop', str(repo / 'docs/experiments/SIMULATOR/raw_sm120/hop_ns.tsv')]
    with (out / 'log/prepare_solver.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=log, check=True)


if __name__ == '__main__':
    main()
