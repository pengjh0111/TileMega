#!/usr/bin/env python3
"""Freeze graph-generated DNN/MoE correctness checks with canonical identities.

Build steps hold the shared GPU lock. Numerical steps run in fresh processes
under the bench scheduler/guard. None of these results are latency samples.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import traceback


DEBUG_ACCESSOR = r'''
// Test-only intermediate access; this library is ineligible for timing.
extern "C" void* tm_dm_debug_buffer(void* handle, unsigned index,
                                    unsigned long long* metadata) {
  auto* plan=static_cast<tilemega::codegen::serving::Plan*>(handle);
  if(!plan || index>=plan->model.spec->buffer_count || !metadata) return nullptr;
  auto const& layout=plan->model.spec->buffers[index].layout;
  metadata[0]=static_cast<unsigned>(layout.kind); metadata[1]=layout.rank;
  for(unsigned i=0;i<4;++i) {
    metadata[2+i]=layout.logical[i]; metadata[6+i]=layout.physical[i];
    metadata[10+i]=layout.strides[i];
  }
  metadata[14]=layout.halo_top; metadata[15]=layout.halo_bottom;
  metadata[16]=layout.halo_left; metadata[17]=layout.halo_right;
  return plan->model.buffers[index];
}
'''
SUPPORT = ['lib/Target/TargetSpec.cpp', 'lib/Support/Json.cpp',
           'lib/Codegen/RuntimeTaskGraph.cpp', 'lib/Solver/PlanMaterialize.cpp',
           'lib/Dialect/CouplingGraph/PlacementPlan.cpp',
           'lib/Solver/BalancedPlacement.cpp', 'lib/Solver/ListScheduler.cpp']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def execute(command, cwd, folder, label, timeout=1800):
    (folder / (label + '.command.json')).write_text(json.dumps(command) + '\n')
    with (folder / (label + '.log')).open('w') as stream:
        process = subprocess.Popen(command, cwd=cwd, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            from tilemega.bench.gpu_guard import stop
            stop(process)
            raise
    if code:
        raise RuntimeError(f'{label} exited {code}; see {folder}/{label}.log')


def prepare(args):
    root, out, inputs = args.root.resolve(), args.out.resolve(), args.inputs.resolve()
    if out.exists():
        raise FileExistsError('generated-model checkpoint already exists')
    compiler = args.compiler.resolve()
    # Import the implementation from the source tree actually being compiled.
    sys.path.insert(0, str(root / 'python'))
    from tilemega.build.identity import snapshot
    out.mkdir(parents=True)
    shutil.copy2(__file__, out / 'runner.py')
    snapshot(root, out / 'source.json', compiler)
    shutil.copy2(args.target, out / 'target.json')
    cases = []
    for model in ('resnet18', 'mbv1', 'mbv2', 'bert', 'nafnet'):
        export = args.dnn_exports.resolve() / model
        cases.append(dict(label=model, kind='dnn', bridge=str(inputs / (model + '.json')),
                          export=str(export), batch=2, pg='off', processes=1,
                          files={str(export / name): sha(export / name) for name in
                                 ('exported_program.pt2', 'manifest.json')}))
    if args.pipelines is not None:
        for export in sorted(args.pipelines.resolve().iterdir()):
            if not (export / 'bridge.json').is_file():
                continue
            cases.append(dict(label='fixture_' + export.name, kind='dnn',
                bridge=str(export / 'bridge.json'), export=str(export), batch=2,
                pg='off', processes=5, elementwise=True,
                files={str(export / name):sha(export / name) for name in
                       ('exported_program.pt2', 'manifest.json')}))
    for tokens in (1, 17, 513):
        cases.append(dict(label=f'moe_T{tokens}', kind='moe',
                          bridge=str(inputs / 'region/bridge.json'), tokens=tokens,
                          batch=1, pg='l2', processes=5, gemv=0, files={}))
    cases.append(dict(label='moe_T1_gemv', kind='moe',
                      bridge=str(inputs / 'region/bridge.json'), tokens=1,
                      batch=1, pg='l2', processes=5, gemv=1, files={}))
    for phase in ('decode', 'prefill'):
        for pg in ('l2', 'pages'):
            cases.append(dict(label=f'decoder_{phase}_{pg}', kind='decoder', phase=phase,
                              bridge=str(inputs / phase / 'bridge.json'),
                              config=str(inputs / 'decoder_config.json'),
                              batch=1, pg=pg, processes=5, files={}))
    if args.labels:
        selected = set(args.labels.split(','))
        cases = [case for case in cases if case['label'] in selected]
        if {case['label'] for case in cases} != selected:
            raise ValueError('unknown generated checkpoint labels')
    for case in cases:
        folder = out / case['label']
        folder.mkdir()
        shutil.copy2(case['bridge'], folder / 'bridge.json')
        case['bridge'] = str(folder / 'bridge.json')
        case['files'][case['bridge']] = sha(case['bridge'])
        if 'config' in case:
            shutil.copy2(case['config'], folder / 'config.json')
            case['config'] = str(folder / 'config.json')
            case['files'][case['config']] = sha(case['config'])
    record = dict(root=str(root), compiler=str(compiler), compiler_sha256=sha(compiler),
                  source_sha256=sha(out / 'source.json'), runner_sha256=sha(out / 'runner.py'),
                  target_sha256=sha(out / 'target.json'), cases=cases, timing_eligible=False,
                  nvcc_timeout_s=args.nvcc_timeout_s,
                  nvcc_split_compile=args.nvcc_split_compile,
                  nvcc_diagnostics=args.nvcc_diagnostics)
    (out / 'inputs.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'Prepared {len(cases)} independently queued generated-model checks')


def validate(out, label):
    record = json.loads((out / 'inputs.json').read_text())
    for name, expected in [('runner.py', record['runner_sha256']),
                           ('source.json', record['source_sha256']),
                           ('target.json', record['target_sha256'])]:
        if sha(out / name) != expected:
            raise ValueError('frozen input changed: ' + name)
    if sha(record['compiler']) != record['compiler_sha256']:
        raise ValueError('frozen compiler changed')
    root = Path(record['root'])
    sys.path.insert(0, str(root / 'python'))
    from tilemega.build.identity import validate_snapshot_inputs
    validate_snapshot_inputs(root, json.loads((out / 'source.json').read_text()))
    case = next(row for row in record['cases'] if row['label'] == label)
    for path, expected in case['files'].items():
        if sha(path) != expected:
            raise ValueError('frozen reference changed: ' + path)
    return record, case


def reuse_inputs(args):
    previous, out = args.reuse_inputs.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError('generated-model checkpoint already exists')
    record = copy.deepcopy(json.loads((previous / 'inputs.json').read_text()))
    selected = set(args.labels.split(',')) if args.labels else {
        case['label'] for case in record['cases']}
    cases = [case for case in record['cases'] if case['label'] in selected]
    if not selected or {case['label'] for case in cases} != selected:
        raise ValueError('unknown generated checkpoint labels')
    for case in cases:
        validate(previous, case['label'])
    out.mkdir(parents=True)
    for name in ('source.json', 'source.json.patch', 'target.json'):
        if (previous / name).exists():
            shutil.copy2(previous / name, out / name)
    shutil.copy2(__file__, out / 'runner.py')
    for case in cases:
        folder = out / case['label']
        folder.mkdir()
        for key in ('bridge', 'config'):
            if key not in case:
                continue
            original = case[key]
            replacement = folder / (key + '.json')
            shutil.copy2(original, replacement)
            case[key] = str(replacement)
            case['files'][str(replacement)] = case['files'].pop(original)
    record.update(cases=cases, runner_sha256=sha(out / 'runner.py'),
                  nvcc_timeout_s=args.nvcc_timeout_s,
                  nvcc_split_compile=args.nvcc_split_compile,
                  nvcc_diagnostics=args.nvcc_diagnostics,
                  reused_inputs=str(previous), reused_inputs_sha256=sha(previous / 'inputs.json'))
    (out / 'inputs.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'Reused {len(cases)} source-bound generated-model inputs')


def compiler_options(record, folder):
    threads = record.get('nvcc_split_compile', 1)
    if not isinstance(threads, int) or threads < 1:
        raise ValueError('nvcc split-compile threads must be positive')
    options = []
    if threads > 1:
        options.append(f'-Xptxas=--split-compile={threads}')
    if record.get('nvcc_diagnostics', False):
        keep = folder / 'nvcc_intermediates'
        keep.mkdir(exist_ok=True)
        options += ['--keep', '--keep-dir', str(keep), '--time', str(folder / 'nvcc_phases.csv')]
    return options


def build(out, label):
    record, case = validate(out, label)
    root, folder = Path(record['root']), out / label
    so = folder / 'plan.so'
    phase = case.get('phase', 'forward')
    command = [record['compiler'], 'compile', case['bridge'], str(so) + '.cu',
               '--emit', 'serving', '--serving', phase, '--frontend',
               'dnn' if case['kind'] == 'dnn' else 'decoder',
               '--runtime-target', str(out / 'target.json'), '--pg', case['pg'],
               '--pdl', 'off', '--watchdog', '1', '--selection', 'predicted',
               '--batch', str(case['batch']), '--nonpaged-la', '1',
               '--moe-gemv', str(case.get('gemv', 0)), '--serve-kv-block', '64',
               '--serve-query-rows', '8' if phase == 'decode' else '16',
               '--capacity', '192', '--past-range', '3:3' if phase == 'decode' else '0:0']
    if case['kind'] == 'moe':
        command += ['--seq', str(case['tokens'])]
    with Path(os.environ['TILEMEGA_GPU_LOCK']).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        execute(command, root, folder, 'generate', 1800)
        with Path(str(so) + '.cu').open('a') as source:
            source.write(DEBUG_ACCESSOR)
        options = shlex.split(Path(str(so) + '.cu.runtime_flags.txt').read_text())
        nvcc = [os.getenv('NVCC', '/usr/local/cuda/bin/nvcc'), '-std=c++17', '-O3',
                '-arch=sm_89', '-DTILEMEGA_ARCH_ID=890', '--shared', '-Xcompiler=-fPIC',
                '--expt-relaxed-constexpr', '-Xptxas=-v',
                '-I' + str(root / 'include'), '-I' + str(root / 'third_party/cutlass/include'),
                '-I' + str(root / 'third_party/cutlass/tools/util/include'),
                *options, *compiler_options(record, folder), '-x', 'cu', str(so) + '.cu',
                *[str(root / p) for p in SUPPORT], '-o', str(so)]
        execute(nvcc, root, folder, 'nvcc', record.get('nvcc_timeout_s', 3600))
        shutil.copy2(folder / 'nvcc.log', str(so) + '.ptxas.log')
        shutil.copy2(str(so) + '.cu.plan.json', str(so) + '.plan.json')
        Path(str(so) + '.build_command.txt').write_text(shlex.join(nvcc) + '\n')
        from tilemega.build.identity import generate
        identity = generate(so, root, out / 'source.json')
    validate(out, label)
    (folder / 'build.json').write_text(json.dumps(dict(artifact_id=identity['artifact_id'],
        source=identity['source']['head'], timing_eligible=False), indent=2) + '\n')
    print('PASS generated and identity-bound ' + label, flush=True)


def check_one(out, label, process):
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise RuntimeError('generated numerical checks require the bench GPU guard')
    _, case = validate(out, label)
    folder, so = out / label, out / label / 'plan.so'
    from tilemega.build.identity import verify
    verify(so)
    if case['kind'] == 'dnn':
        from tilemega.dnn.check_generated import check
        result = check(so, case['export'], case['bridge'], case['batch'],
                       synthetic_weights=True, elementwise=case.get('elementwise', False),
                       diagnostics=folder / f'intermediates_{process}.json')
    elif case['kind'] == 'moe':
        from tilemega.moe.check_generated import check
        result = check(so, case['bridge'])
    else:
        from tilemega.moe.check_decoder import check
        result = check(so, case['config'], case['bridge'])
    (folder / f'numeric_{process}.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PASS generated numerical oracle and C-2: ' + label, flush=True)


def check(out, label):
    _, case = validate(out, label)
    folder = out / label
    failures = []
    for process in range(case['processes']):
        try:
            execute([sys.executable, str(out / 'runner.py'), 'check-one', '--out', str(out),
                     '--case', label, '--process', str(process)], folder, folder,
                    f'numeric_{process}', 1800)
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            failures.append(dict(process=process, error=str(error)))
    (folder / 'numeric.json').write_text(json.dumps(dict(label=label,
        passed=case['processes']-len(failures), processes=case['processes'], failures=failures,
        timing_eligible=False), indent=2) + '\n')
    if failures:
        raise RuntimeError(f'{label}: {len(failures)} numerical processes failed')
    print(f'PASS {label}: {case["processes"]}/{case["processes"]} fresh processes', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'build', 'check', 'check-one'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--case')
    parser.add_argument('--process', type=int, default=0)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--compiler', type=Path)
    parser.add_argument('--target', type=Path)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--dnn-exports', type=Path)
    parser.add_argument('--pipelines', type=Path)
    parser.add_argument('--labels', help='prepare only these comma-separated cases')
    parser.add_argument('--reuse-inputs', type=Path,
                        help='copy validated frozen inputs into a new checkpoint')
    parser.add_argument('--nvcc-timeout-s', type=int, default=10800,
                        help='frozen compilation budget; does not change optimization flags')
    parser.add_argument('--nvcc-split-compile', type=int, default=1,
                        help='ptxas optimization threads; recorded separately for correctness retries')
    parser.add_argument('--nvcc-diagnostics', action='store_true',
                        help='retain compiler intermediates and per-phase elapsed times')
    args = parser.parse_args()
    if args.nvcc_timeout_s <= 0:
        parser.error('--nvcc-timeout-s must be positive')
    if args.nvcc_split_compile < 1:
        parser.error('--nvcc-split-compile must be positive')
    if args.action == 'prepare':
        if args.reuse_inputs:
            reuse_inputs(args)
            return
        if any(getattr(args, key) is None for key in
               ('root', 'compiler', 'target', 'inputs', 'dnn_exports')):
            parser.error('prepare requires root, compiler, target, inputs and dnn-exports')
        prepare(args)
    else:
        if args.case is None:
            parser.error('build/check require --case')
        try:
            {'build': build, 'check': check, 'check-one': lambda o, c: check_one(o, c, args.process)
             }[args.action](args.out.resolve(), args.case)
        except BaseException as error:
            failure = args.out.resolve() / args.case / (args.action + '_failure.json')
            failure.write_text(json.dumps(dict(error=str(error), traceback=traceback.format_exc()),
                                          indent=2) + '\n')
            raise


if __name__ == '__main__':
    main()
