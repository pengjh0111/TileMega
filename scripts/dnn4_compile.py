#!/usr/bin/env python3
"""Compile one identified F3 source with active Hopper register reallocation."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from tilemega.build.identity import generate, sha, snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='source artifact prefix (.so)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--trace-level', type=int, choices=(0, 1, 2, 3))
    parser.add_argument('--prefetch-distance', type=int, choices=(1, 2))
    parser.add_argument('--cpu-set', default='136-143')
    parser.add_argument('--build-lock', default='/tmp/tilemega_dnn4_native.lock')
    parser.add_argument('--gather-reference', action='store_true',
                        help='DT-STEM control: retain the96-lane producer with otherwise identical source')
    args = parser.parse_args()
    args.input = args.input.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    library = args.out / 'model.so'
    for suffix in ('.cu', '.plan.json', '.tmexec.mlir'):
        source = Path(str(args.input) + suffix)
        if suffix != '.tmexec.mlir' or source.exists():
            shutil.copyfile(source, Path(str(library) + suffix))
    manifest_path = Path(str(library) + '.plan.json')
    manifest = json.loads(manifest_path.read_text())
    coordinates = manifest['native_forward']
    if coordinates['schema'] == 3:
        if any(s['body']=='encoder_attention' for s in coordinates['stages']):
            raise ValueError('regenerate BERT sources: schema4 changes query maps and statistics storage')
        coordinates['schema'] = 4
        for index, stage in enumerate(coordinates['stages']):
            stage.update(tk=64, store='reg', mc=0)
            manifest['stages'][index]['coordinate'] = dict(stage)
        manifest['register_contract'].update(threads=384, callee_limit=224)
        source_path = Path(str(library) + '.cu')
        source = source_path.read_text()
        identity = json.dumps(json.dumps(coordinates, separators=(',', ':')))
        source, count = re.subn(r'(kF3ForwardIdentity\[\]\s*=\s*)"(?:\\.|[^"\\])*";',
                               lambda m: m[1] + identity + ';', source)
        if count != 1:
            raise ValueError('expected exactly one embedded coordinate identity')
        source = re.sub(r'(using Body_\d+=Sm90F3Gemm(?:SwapAB)?<[^;]+)>;',
                        lambda m: m[1] + ',64>;', source)
        source_path.write_text(source)
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    if args.gather_reference:
        source_path = Path(str(library) + '.cu')
        source = source_path.read_text()
        if 'Sm90F3Mode::kTmaStem' not in source:
            raise ValueError('DT-STEM control needs a TMA stem source')
        source_path.write_text(source.replace('Sm90F3Mode::kTmaStem','Sm90F3Mode::kGather'))
    if args.trace_level is not None:
        source_path = Path(str(library) + '.cu')
        source, count = re.subn(r'#define TILEMEGA_TRACE_STAGE [0-3]\b',
                               '#define TILEMEGA_TRACE_STAGE ' + str(args.trace_level),
                               source_path.read_text())
        if count != 1:
            raise ValueError('expected one explicit trace level')
        source_path.write_text(source)
    if args.prefetch_distance is not None:
        source_path = Path(str(library) + '.cu')
        source = source_path.read_text()
        if re.search(r'^#define TILEMEGA_F3_PREFETCH_DISTANCE ', source, re.M):
            source = re.sub(r'^#define TILEMEGA_F3_PREFETCH_DISTANCE [12]$',
                            '#define TILEMEGA_F3_PREFETCH_DISTANCE ' +
                            str(args.prefetch_distance), source, flags=re.M)
        else:
            source = '#define TILEMEGA_F3_PREFETCH_DISTANCE ' + str(args.prefetch_distance) + '\n' + source
        source_path.write_text(source)
    receipt = args.out / 'source.json'
    snapshot(ROOT, receipt)
    command = ['/usr/local/cuda/bin/nvcc', '-std=c++17', '-O3',
               '-gencode', 'arch=compute_90a,code=sm_90a', '--expt-relaxed-constexpr',
               '-shared', '-Xcompiler', '-fPIC', '-cudart', 'shared', '-Xptxas', '-v',
               '--split-compile', '1', '-I' + str(ROOT / 'include'),
               '-I' + str(ROOT / 'third_party/cutlass/include'),
               str(library) + '.cu', '-o', str(library)]
    # Separate device compilation disables setmaxnreg (ptxas C7504).
    Path(str(library) + '.build_command.txt').write_text(shlex.join(command) + '\n')
    log = Path(str(library) + '.ptxas.log')
    with log.open('w') as stream:
        done = subprocess.run(['taskset', '-c', args.cpu_set, sys.executable,
            str(ROOT / 'scripts/run_sm90r1_build.py'), '--receipt',
            str(library) + '.build_limits.json', '--max-gib', '48', '--timeout', '1800',
            '--lock', args.build_lock, '--', *command], cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdout=stream,
            stderr=subprocess.STDOUT)
    text = log.read_text()
    spills = list(map(int, re.findall(r'(\d+) bytes spill (?:stores|loads)', text)))
    if done.returncode or not spills or any(spills) or re.search(r'C750[47]', text):
        raise RuntimeError('compile, spill or register reallocation gate failed: ' + str(log))
    sass = args.out / 'model.sass'
    with sass.open('w') as stream:
        subprocess.run(['cuobjdump', '--dump-sass', str(library)], stdout=stream, check=True)
    code = sass.read_text()
    if 'USETMAXREG.TRY_ALLOC.CTAPOOL' not in code or 'USETMAXREG.DEALLOC.CTAPOOL' not in code:
        raise RuntimeError('linked kernel lacks register reallocation')
    identity = generate(library, ROOT, receipt)
    result = dict(status='passed', evidence='verified', library=str(library),
                  artifact_id=identity['artifact_id'], source_snapshot_sha256=sha(receipt),
                  sass_sha256=sha(sass),
                  sass_register_max=max(map(int, re.findall(r'\bR(\d+)\b', code))))
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
