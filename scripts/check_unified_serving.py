#!/usr/bin/env python3
"""Regenerate G-REG-U plans with two compilers, then compare real-weight tokens.

Prepare is CPU-only. Build is a queued CPU step holding TILEMEGA_GPU_LOCK;
run is a correctness-only GPU step under the bench scheduler/guard.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(command, cwd, log, timeout):
    from tilemega.bench.gpu_guard import stop
    with Path(log).open('w') as stream:
        process = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            stop(process)
            raise
    if code:
        raise RuntimeError(f'exit {code}: {log}')


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def prepare(root, out, work):
    if (out / 'inputs.json').exists():
        raise FileExistsError('reference inputs already frozen')
    out.mkdir(parents=True, exist_ok=True)
    records = {}
    for label, head in [('reference', '97f7a2f1d'), ('candidate', git(root, 'rev-parse', 'HEAD'))]:
        tree = work / label
        subprocess.run(['git', '-C', str(root), 'worktree', 'add', '--detach', '--no-checkout',
                        str(tree), head], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(['git', '-C', str(tree), 'sparse-checkout', 'init', '--cone'],
                       check=True, stdout=subprocess.DEVNULL)
        subprocess.run(['git', '-C', str(tree), 'sparse-checkout', 'set',
                        'include', 'lib', 'tools', 'cmake', 'python', 'scripts', 'test', 'configs'],
                       check=True, stdout=subprocess.DEVNULL)
        subprocess.run(['git', '-C', str(tree), 'read-tree', '-mu', 'HEAD'],
                       check=True, stdout=subprocess.DEVNULL)
        # Old compilers include these data files in their source fingerprint.
        # Materialize their exact committed versions, without taking the
        # experiment tree or changing the historical fingerprint algorithm.
        for path in ('docs/experiments/SERVING_R10/calibration/target_serving.json',
                     'docs/experiments/SIMULATOR/hop_ns.tsv'):
            frozen = subprocess.check_output(['git', '-C', str(tree), 'show', head + ':' + path])
            destination = tree / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(frozen)
        if not (tree / 'CMakeLists.txt').is_file():
            raise FileNotFoundError('sparse compiler checkout was not materialized')
        # CUTLASS and the polyhedral libraries are unchanged pinned dependencies.
        # Sources and headers under include/lib/tools remain independent checkouts.
        third_party = tree / 'third_party'
        if third_party.is_dir() and not any(third_party.iterdir()):
            third_party.rmdir()
        if third_party.exists():
            raise FileExistsError('sparse reference unexpectedly contains third_party')
        third_party.symlink_to(root / 'third_party', target_is_directory=True)
        records[label] = dict(root=str(tree), head=git(tree, 'rev-parse', 'HEAD'),
                              compiler=str(tree / 'build-r15/tools/tilemega'))
    a = json.loads((root / 'docs/experiments/SERVING_R14/phase_a_builds.json').read_text())
    b = json.loads((root / 'docs/experiments/SERVING_R14/phase_b_builds.json').read_text())
    jobs = [r for r in a if r['label'] == 'R13D_prime' or
            (r['cell'] == 'llama_B1' and r['label'] == 'N1_prime')]
    jobs += [r for r in b if r['label'] == 'AT_pv' or
             (r['cell'] == 'qwen3_B1' and r['label'] == 'AT_ec128')]
    cases = []
    for job in jobs:
        folder = out / 'inputs' / job['cell'] / job['label']
        folder.mkdir(parents=True)
        old = Path(job['out'])
        files = {}
        for source, name in [(old / 'selected.mlir', 'selected.mlir'),
                             (old / 'plan.so.plan.json', 'plan.json'),
                             (old / 'plan.so.build_command.txt', 'nvcc.txt'),
                             (Path(job['target']), 'target.json')]:
            shutil.copy2(source, folder / name)
            files[str(folder / name)] = sha(folder / name)
        cases.append(dict(cell=job['cell'], label=job['label'], batch=job['batch'],
                          input=str(folder), files=files, overrides=job['overrides']))
    arms = json.loads((root / 'runs/r15/rf1/arms.json').read_text())
    numeric = {}
    llama_plans = json.loads((root / 'runs/r12b-llama/plans.json').read_text())
    for cell in sorted({case['cell'] for case in cases}):
        model, batch = cell.split('_B')
        if model == 'qwen3':
            spec = arms['cells'][cell]
            prefill = Path(spec['arms'][0]['prefill'])
            model_dir, prompts = spec['model_path'], Path(spec['prompt_ids'])
        else:
            prefill = Path(llama_plans[batch]['prefill'])
            model_dir, prompts = '/root/models/llama3_2_1b', root / 'configs/prompts/llama3_isl64.json'
        copied = out / 'inputs' / (cell + '_prompts.json')
        shutil.copy2(prompts, copied)
        numeric[cell] = dict(model=model_dir, prefill=str(prefill), batch=int(batch),
                             prompts=str(copied), files={str(copied): sha(copied),
                             str(prefill): sha(prefill), str(prefill)+'.plan.json': sha(str(prefill)+'.plan.json')})
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    record = dict(trees=records, cases=cases, runner_sha256=sha(out / 'runner.py'),
                  root=str(root), cutlass_head=git(root / 'third_party/cutlass', 'rev-parse', 'HEAD'),
                  numeric_cases=numeric, rf1_arms_snapshot=arms)
    (out / 'inputs.json').write_text(json.dumps(record, indent=2) + '\n')
    print('Prepared two compiler worktrees and ' + str(len(cases)) + ' regression plans')


def reuse_inputs(previous, out, nvcc_timeout_s=10800):
    record = validate(previous)
    if out.exists():
        raise FileExistsError('regression checkpoint already exists')
    if nvcc_timeout_s <= 0:
        raise ValueError('nvcc timeout must be positive')
    out.mkdir(parents=True)
    shutil.copy2(__file__, out / 'runner.py')
    record['runner_sha256'] = sha(out / 'runner.py')
    record['runner_head'] = git(Path(record['root']), 'rev-parse', 'HEAD')
    record['reuses_inputs'] = str(previous)
    record['nvcc_timeout_s'] = nvcc_timeout_s
    (out / 'inputs.json').write_text(json.dumps(record, indent=2) + '\n')
    print('Prepared new runner with unchanged, validated compiler inputs')


def prepare_prefill(previous, out, nvcc_timeout_s):
    record = validate(previous)
    if set(record['numeric_cases']) != {'llama_B1', 'llama_B16', 'qwen3_B1', 'qwen3_B16'}:
        raise ValueError('prefill regression requires all four default cells')
    if out.exists():
        raise FileExistsError('prefill checkpoint already exists')
    if nvcc_timeout_s <= 0:
        raise ValueError('nvcc timeout must be positive')
    out.mkdir(parents=True)
    root = Path(record['root'])
    registry = root / 'docs/experiments/SERVING_R14/phase_a_arms.json'
    arms = json.loads(registry.read_text())
    shutil.copy2(registry, out / 'reference_arms.json')
    cases = []
    for cell, numeric in record['numeric_cases'].items():
        matches = [arm for arm in arms[cell] if arm['label'] == 'R13D_prime']
        if len(matches) != 1:
            raise ValueError('expected one R14 default prefill for ' + cell)
        so = Path(matches[0]['prefill'])
        manifest = Path(str(so) + '.plan.json')
        data = json.loads(manifest.read_text())
        if data.get('phase') != 'prefill' or data.get('seq') != 64 or data.get('pg') != 'l2':
            raise ValueError('unexpected R14 prefill contract: ' + str(manifest))
        source = next(case for case in record['cases']
                      if case['cell'] == cell and case['label'] == 'R13D_prime')
        folder = out / 'inputs' / cell
        folder.mkdir(parents=True)
        files = {}
        for origin, name in ((so.parent / 'selected.mlir', 'selected.mlir'),
                (manifest, 'plan.json'), (Path(str(so)+'.build_command.txt'), 'nvcc.txt'),
                (Path(source['input']) / 'target.json', 'target.json')):
            shutil.copy2(origin, folder / name)
            files[str(folder / name)] = sha(folder / name)
        overrides = dict(watchdog=int(data.get('watchdog', 0)),
                         pdl=data.get('pdl', 'auto'), sync=data.get('sync', 'calibrated'))
        cases.append(dict(cell=cell, label='R14_prefill', phase='prefill', batch=numeric['batch'],
                          input=str(folder), files=files, overrides=overrides))
        numeric['prefill'] = str(so)
        numeric['files'].update({str(so):sha(so), str(manifest):sha(manifest)})
    record.update(cases=cases, decode_checkpoint=str(previous), nvcc_timeout_s=nvcc_timeout_s,
                  runner_head=git(root, 'rev-parse', 'HEAD'),
                  reference_registry=dict(path=str(out/'reference_arms.json'),
                                          sha256=sha(out/'reference_arms.json')))
    shutil.copy2(__file__, out / 'runner.py')
    record['runner_sha256'] = sha(out / 'runner.py')
    (out / 'inputs.json').write_text(json.dumps(record, indent=2) + '\n')
    print('Prepared four R14 prefill comparisons with unchanged compiler worktrees')


def validate(out):
    record = json.loads((out / 'inputs.json').read_text())
    if sha(out / 'runner.py') != record['runner_sha256']:
        raise ValueError('regression runner changed')
    for row in record['trees'].values():
        tree = Path(row['root'])
        if git(tree, 'rev-parse', 'HEAD') != row['head']:
            raise ValueError('reference worktree HEAD changed')
        if git(tree, 'diff', '--name-only', 'HEAD', '--', 'include', 'lib', 'tools', 'python', 'cmake', 'CMakeLists.txt'):
            raise ValueError('reference source modified')
    for case in record['cases']:
        for path, expected in case['files'].items():
            if sha(path) != expected:
                raise ValueError('frozen regression input changed: ' + path)
    for case in record['numeric_cases'].values():
        for path, expected in case['files'].items():
            if sha(path) != expected:
                raise ValueError('numeric reference input changed: ' + path)
    registry = record.get('reference_registry')
    if registry and sha(registry['path']) != registry['sha256']:
        raise ValueError('prefill registry changed')
    root = Path(record['root'])
    if git(root / 'third_party/cutlass', 'rev-parse', 'HEAD') != record['cutlass_head']:
        raise ValueError('CUTLASS revision changed')
    return record


@contextlib.contextmanager
def locked():
    with Path(os.environ['TILEMEGA_GPU_LOCK']).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def build(out):
    record = validate(out)
    root = Path(record['root'])
    built = []
    with locked():
        for label, item in record['trees'].items():
            tree = Path(item['root'])
            build_dir = tree / 'build-r15'
            command = ['cmake', '-S', str(tree), '-B', str(build_dir), '-G', 'Ninja',
                '-DCMAKE_BUILD_TYPE=Release', '-DTILEMEGA_TARGET_ARCH=sm_89',
                '-DTILEMEGA_BUILD_TESTS=OFF', '-DTILEMEGA_BUILD_VERIFY=OFF',
                '-DTILEMEGA_BUILD_EXPERIMENTAL=OFF',
                '-DMLIR_DIR=/root/toolchains/mlir-23a60f15/lib/cmake/mlir']
            for dep in ('ISL', 'POLYLIB', 'BARVINOK'):
                command.append('-DTILEMEGA_' + dep + '_BUILD_DIR=' + str(root / ('build-' + dep.lower())))
            execute(command, tree, out / (label + '_configure.log'), 600)
            execute(['cmake', '--build', str(build_dir), '--target', 'tilemega', '-j', '6'],
                    tree, out / (label + '_compiler.log'), 7200)
            item['compiler_sha256'] = sha(item['compiler'])
        for case in record['cases']:
            input_dir = Path(case['input'])
            data = json.loads((input_dir / 'plan.json').read_text())
            opts = dict(case['overrides'])
            opts.update(pg=data['pg'], page_bytes=(data.get('pages') or {}).get('page_bytes', 16384),
                        weight_layout='tiled' if data['pg'] == 'pages' else 'row')
            # selected.mlir already contains the proved/elided reductions.
            # Lower those decisions; running selection again is rejected by
            # both compilers. This flag only skips the repeated transform.
            if data['pg'] == 'pages':
                opts['paged_la'] = 0
            for label, item in record['trees'].items():
                tree = Path(item['root'])
                folder = out / 'builds' / case['cell'] / case['label'] / label
                folder.mkdir(parents=True)
                so = folder / 'plan.so'
                phase = case.get('phase', 'decode')
                command = [item['compiler'], 'compile', str(input_dir / 'selected.mlir'), str(so) + '.cu',
                    '--serving', phase, '--emit', 'serving', '--batch', str(case['batch']),
                    '--past-range', '0:0' if phase == 'prefill' else '64:1086', '--capacity', '1088',
                    '--runtime-target', str(input_dir / 'target.json')]
                for key, value in opts.items():
                    command += ['--' + key.replace('_', '-'), str(value)]
                (folder / 'generate_command.json').write_text(json.dumps(command) + '\n')
                execute(command, tree, folder / 'generate.log', 600)
                nvcc = shlex.split((input_dir / 'nvcc.txt').read_text())
                for i, token in enumerate(nvcc):
                    if token.endswith('.cu') and not token.startswith('-'):
                        nvcc[i] = str(so) + '.cu'
                    elif token == '-o':
                        nvcc[i+1] = str(so)
                    elif token.startswith('-I') and str(root / 'include') in token:
                        nvcc[i] = token.replace(str(root / 'include'), str(tree / 'include'))
                    elif token.startswith(str(root / 'lib') + '/'):
                        nvcc[i] = str(tree / Path(token).relative_to(root))
                (folder / 'nvcc_command.json').write_text(json.dumps(nvcc) + '\n')
                execute(nvcc, tree, Path(str(so) + '.ptxas.log'), record.get('nvcc_timeout_s', 1800))
                shutil.copy2(input_dir / 'plan.json', str(so) + '.plan.json')
                built.append(dict(cell=case['cell'], label=case['label'], variant=label,
                                  so=str(so), so_sha256=sha(so), cu_sha256=sha(str(so) + '.cu')))
            paths = {row['variant']: row['so'] for row in built
                     if row['cell'] == case['cell'] and row['label'] == case['label']}
            comparison = out / 'comparisons' / case['cell'] / case['label']
            command = [sys.executable, '-m', 'tilemega.bench.compare_artifacts',
                '--reference', paths['reference'], '--candidate', paths['candidate'], '--out', str(comparison)]
            env = dict(os.environ, PYTHONPATH=record['trees']['candidate']['root'] + '/python')
            comparison.parent.mkdir(parents=True, exist_ok=True)
            with (comparison.parent / (case['label'] + '.log')).open('w') as log:
                # The comparator writes full per-function differences itself.
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
            if result.returncode:
                raise RuntimeError('G-REG-U artifact mismatch: ' + str(comparison))
    validate(out)
    (out / 'build.json').write_text(json.dumps(dict(trees=record['trees'], plans=built), indent=2) + '\n')
    print('G-REG-U generated source, resources and SASS PASS')


def execution_pair(record, case, row):
    if case.get('phase', 'decode') != 'prefill':
        return row['so'], None
    checkpoint = Path(record['decode_checkpoint'])
    built = json.loads((checkpoint / 'build.json').read_text())
    matches = [plan for plan in built['plans'] if plan['cell'] == row['cell']
               and plan['label'] == 'R13D_prime' and plan['variant'] == row['variant']]
    if len(matches) != 1 or sha(matches[0]['so']) != matches[0]['so_sha256']:
        raise ValueError('prefill comparison needs an unchanged matching decode artifact')
    return matches[0]['so'], row['so']


def run(out):
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise RuntimeError('numerical regression requires the bench guard')
    record = validate(out)
    if record.get('decode_checkpoint'):
        validate(Path(record['decode_checkpoint']))
    built = json.loads((out / 'build.json').read_text())
    reports = []
    for case in record['cases']:
        outputs = []
        for row in built['plans']:
            if row['cell'] != case['cell'] or row['label'] != case['label']:
                continue
            if sha(row['so']) != row['so_sha256']:
                raise ValueError('regression binary changed')
            folder = out / 'numeric' / row['cell'] / row['label'] / row['variant']
            folder.mkdir(parents=True)
            env = dict(os.environ, PYTHONPATH=record['trees']['candidate']['root'] + '/python')
            decode, prefill = execution_pair(record, case, row)
            command = [sys.executable, str(out / 'runner.py'), 'run-one', '--out', str(out),
                       '--cell', row['cell'], '--so', decode, '--tokens', str(folder / 'tokens.json')]
            if prefill:
                command += ['--prefill-so', prefill]
            with (folder / 'run.log').open('w') as log:
                process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
            if process.returncode:
                raise RuntimeError('real-weight token generation failed: ' + str(folder))
            outputs.append(folder / 'tokens.json')
        if len(outputs) != 2 or outputs[0].read_bytes() != outputs[1].read_bytes():
            raise RuntimeError('cross-build token mismatch: ' + case['cell'] + '/' + case['label'])
        reports.append(dict(cell=case['cell'], label=case['label'], pass_tokens=True,
                            tokens_sha256=sha(outputs[0]), generated_tokens=64))
    validate(out)
    (out / 'numeric.json').write_text(json.dumps(dict(cases=reports, timing_eligible=False,
        scope='G-REG-U ' + ('prefill' if record.get('decode_checkpoint') else 'decode') +
              ' cases; full G-DM-U remains separate'), indent=2) + '\n')
    print('G-REG-U real-weight 64-token comparison PASS')


def run_one(out, cell, so, tokens, prefill_so=None):
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise RuntimeError('numerical regression requires the bench guard')
    import torch
    from tilemega.serving.engine import ServingEngine
    spec = json.loads((out / 'inputs.json').read_text())['numeric_cases'][cell]
    prompts = torch.tensor(json.loads(Path(spec['prompts']).read_text())[:spec['batch']], dtype=torch.int32)
    with ServingEngine(spec['model'], prefill_so or spec['prefill'], so, spec['batch'],
                       max_new_tokens=64, mode='L1', prefill_mode='L1',
                       decode_loop=False, step_events=False) as engine:
        try:
            result = engine.generate(prompts, 64)
            tokens.write_text(json.dumps(result.tokens.tolist(), separators=(',', ':')) + '\n')
        except BaseException:
            tokens.with_name('watchdog.json').write_text(json.dumps(engine.decode.watchdog()) + '\n')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'build', 'run', 'run-one'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--work', type=Path, default=Path('/root/r15_work/unification'))
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--reuse-inputs', type=Path)
    inputs.add_argument('--prefill-from', type=Path)
    parser.add_argument('--nvcc-timeout-s', type=int, default=10800)
    parser.add_argument('--cell')
    parser.add_argument('--so', type=Path)
    parser.add_argument('--prefill-so', type=Path)
    parser.add_argument('--tokens', type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.action == 'prepare':
        if args.prefill_from is not None:
            prepare_prefill(args.prefill_from.resolve(), out, args.nvcc_timeout_s)
        elif args.reuse_inputs is not None:
            reuse_inputs(args.reuse_inputs.resolve(), out, args.nvcc_timeout_s)
        else:
            prepare(Path(__file__).resolve().parents[1], out, args.work)
    elif args.action == 'build':
        build(out)
    elif args.action == 'run':
        run(out)
    else:
        if not args.cell or not args.so or not args.tokens:
            parser.error('run-one requires --cell, --so and --tokens')
        run_one(out, args.cell, args.so, args.tokens, args.prefill_so)


if __name__ == '__main__':
    main()
