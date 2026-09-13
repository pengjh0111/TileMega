#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Bind EFT plans to measured sm_120 control geometry, not sm_89 tables."""
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
    old = repo / 'docs/experiments/PLACE_EFT/raw'
    if out == old or old in out.parents:
        parser.error('sm_89 output directories are read-only inputs')
    build = Path(os.environ.get('BUILD_DIR', repo / 'build-portable'))
    nvcc = os.environ.get('CUDACXX', '/usr/local/cuda/bin/nvcc')
    for folder in ('plan', 'log', 'prepare'):
        (out / folder).mkdir(parents=True, exist_ok=True)
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
                capture_output=True, text=True, timeout=300)
            text = result.stdout + result.stderr
            (out / 'prepare' / f'{model}_p5_s{seq}.out').write_text(text)
            if result.returncode or 'RESULT status=PASS' not in text:
                raise RuntimeError(f'control probe failed: {model} seq={seq}')
            if 'E2E_RESOURCE ' not in text or 'E2E_PLACE_BASE ' not in text:
                raise RuntimeError('control probe lacks resource/task graph evidence')
            # Without a trace, the simulator explicitly reports modulo worker-SM
            # assignment; it must not borrow sm_89 hardware placement observations.
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
    command = [str(build / 'tools/tilemega-place-eft'), str(repo), str(manifest),
               str(out), '--target', str(target),
               '--hop', str(repo / 'docs/experiments/SIMULATOR/raw_sm120/hop_ns.tsv')]
    with (out / 'log/prepare_solver.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=log, check=True)


if __name__ == '__main__':
    main()
