#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Solve every candidate's plan from measured sm_120 geometry, never from sm_89.

Both placement tools bind `(worker, slot)` to the resident grid they were solved
against, and the dumps `run.sh` generates from are a 4090's -- grid 256 over 128
SMs.  Round one copied EFT tables across devices and the host's plan guard
refused them (H9), so the geometry is probed again here, on the target, and the
manifest points at those local probes.

`run.sh` is then driven with SKIP_GENERATE=1, so everything its generate phase
would have done has to happen here instead: both tools, the byte-identity check
on the three candidates they share, and the one-at-a-time merge into `plan/`.

No trace exists on the target, so `trace_dir` is `-` and worker-SM assignment is
reported modulo rather than borrowed from sm_89 observations.
"""
import argparse
import csv
import filecmp
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--realwidth', choices=('0', '1'), default='1')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    out = args.out.resolve()
    old = repo / 'docs/experiments/PLACE_EFT2/raw'
    if out == old or old in out.parents:
        parser.error('sm_89 output directories are read-only inputs')
    build = Path(os.environ.get('BUILD_DIR', repo / 'build-portable'))
    nvcc = os.environ.get('CUDACXX', '/usr/local/cuda/bin/nvcc')
    for folder in ('plan', 'log', 'prepare'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    # Mode 5 is the control the probe is taken from, matching how the sm_89
    # dumps were produced.  Each candidate's own macro is applied by run.sh when
    # it compiles that arm, not here.
    common = ['-std=c++17', '-O2', '-arch=sm_120', '-lineinfo',
              '-DTILEMEGA_EVENT_KAPPA=1', '-DTILEMEGA_PLACEMENT=5',
              '-Xptxas=-v,--warn-on-spills']
    common += [f'-I{repo / path}' for path in (
        'include', 'third_party/cutlass/include',
        'third_party/cutlass/tools/util/include', 'third_party/cutlass/test')]
    control = repo / 'docs/experiments/PLAN_CONTRACT/legacy_identity/plan'
    rows = []
    models = ['gqa2', 'mha4'] + (['real'] if args.realwidth == '1' else [])
    for seq in (4, 128):
        for model in models:
            if model == 'real':
                work = repo / f'docs/experiments/REALMODEL/raw/work/r2sim_s{seq}'
                source, export, fixture = (work / 'model.cu', work / 'model.json',
                                           work / 'export/fixture')
            else:
                source = control / f'{model}.cu'
                export = repo / f'docs/experiments/SEQSCAN/raw/export/{model}.json'
                fixture = repo / f'docs/experiments/SEQSCAN/raw/fixture/{model}_s{seq}_p3'
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
            rows.append([model, seq, 3, source, out / 'prepare' / model, '-', export])
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
    hop = repo / 'docs/experiments/SIMULATOR/raw_sm120/hop_ns.tsv'
    for tool in ('eft', 'chain'):
        # The tool writes into `plan/` but does not create it.
        (out / f'gen_{tool}/plan').mkdir(parents=True, exist_ok=True)
        command = [str(build / f'tools/tilemega-place-{tool}'), str(repo),
                   str(manifest), str(out / f'gen_{tool}'),
                   '--target', str(target), '--hop', str(hop)]
        with (out / f'log/place_{tool}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=log, check=True)
    # Both tools reach legacy_grid_stride, band and wavefront through their own
    # import and codegen path, so a difference there means one of them is not
    # emitting the placement it names.  Checked before the merge, because the
    # merge is what would hide it.
    agreement = out / 'tool_agreement.tsv'
    with agreement.open('w') as stream:
        stream.write('file\tidentical_between_tools\n')
        for emitted in sorted((out / 'gen_eft/plan').glob('*.cu')):
            other = out / 'gen_chain/plan' / emitted.name
            if not other.exists():
                continue
            same = filecmp.cmp(emitted, other, shallow=False)
            stream.write(f'{emitted.name}\t{1 if same else 0}\n')
            if not same:
                raise RuntimeError(
                    f'{emitted.name} differs between the two placement tools')
    # One copy pass per tool: the second rewrites the three shared files with
    # content the check above just proved byte-identical.
    for tool in ('eft', 'chain'):
        for emitted in sorted((out / f'gen_{tool}/plan').glob('*.cu')):
            (out / 'plan' / emitted.name).write_bytes(emitted.read_bytes())


if __name__ == '__main__':
    main()
