#!/usr/bin/env python3
"""Archive E0 acceptance and append fresh calibration/fixed-build dependencies."""
import argparse
import concurrent.futures
import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from bootstrap import HERE, ROOT, FRAME, WORK, BUILD, TMP, PY, sha, write, run


def audit():
    destination = HERE / 'raw/acceptance_01'
    destination.mkdir(parents=True, exist_ok=True)
    state = json.loads((WORK / 'scheduler/state.json').read_text())
    selected = {k: v for k, v in state.items() if k.endswith('_r2')}
    if any(v['status'] in ('pending', 'running') for v in selected.values()):
        raise RuntimeError('r2 has not terminated')
    statuses = {}
    for row in selected.values():
        statuses[row['status']] = statuses.get(row['status'], 0) + 1
    tests = []
    for case in ET.parse(HERE / 'raw/E0_units/ctest.xml').getroot().iter('testcase'):
        failure = case.find('failure')
        tests.append(dict(name=case.attrib['name'], status=case.attrib.get('status'),
                          failure=failure.attrib if failure is not None else None))
    native = []
    for model in ('llama', 'qwen3'):
        target = next((TMP / 'cache' / model / 'targets').glob('*.json'))
        data = json.loads(target.read_text())
        body = data['calibration_by_dtype']['bf16']['task_body']
        native.append(dict(model=model, path=str(target), sha256=sha(target),
                           sections=list(data['calibration_sections']),
                           native_serving_kinds=len(body['serving']),
                           paged_and_loader_entries=len(body['serving_paged']),
                           legacy_phase_samples=body['samples'],
                           command_exit=json.loads((HERE / f'raw/E1_{model}/commands/calibrate/command.json').read_text())['returncode']))
    write(destination / 'acceptance.json', dict(utc=time.time(), state=selected,
          statuses=statuses, ctest=tests, native_calibrations=native,
          python_tests_passed=31, ctest_passed=84, ctest_total=97,
          accepted_phase=False, E2a_executed=False,
          reason='native profile admission rejects optional missing legacy fits'))
    folders = ['E0_core_r2', 'E0_environment', 'E0_environment_r2', 'E0_units',
               'E0_units_r2', 'E0_test_build', 'E0_model_llama', 'E0_model_qwen3',
               'E1_llama', 'E1_qwen3', 'E1_llama_r2', 'E1_qwen3_r2']
    subprocess.run(['tar', '-cJf', str(destination / 'e0_e1_original_evidence.tar.xz'),
                    '-C', str(HERE / 'raw'), *folders], check=True)
    write(destination / 'archive.json', dict(path=str(destination / 'e0_e1_original_evidence.tar.xz'),
          sha256=sha(destination / 'e0_e1_original_evidence.tar.xz')))
    print(json.dumps(statuses), 'CTest 84/97; Python 31/31; E2a not run', flush=True)
    return 0


def rebuild():
    folder = HERE / 'raw/E0_native_build_r3'
    code = run(['cmake', '--build', BUILD, '--target', 'tilemega', 'tilemega-unit',
                'tilemega-opt', '--parallel', '3'], folder / 'build.log')
    if code:
        return code
    return run([sys.executable, ROOT / 'python/tilemega/fingerprint.py', '--check',
                BUILD / 'tools/tilemega'], folder / 'fingerprint.log')


def contracts():
    folder = HERE / 'raw/E0_contracts_r3'
    expression = '^(target_spec|frontend_import|plan_skeleton|operator_classes|coupling_cache|semantic_lifting|embedding_plan|wiring_coupling|pipeline_sigma)$'
    codes = [run(['ctest', '--test-dir', BUILD, '-R', expression, '--output-on-failure',
                  '--timeout', '900', '--output-junit', folder / 'ctest.xml'], folder / 'ctest.log')]
    codes.append(run([PY, ROOT / 'test/python/serving_calibration_contract.py'],
                     folder / 'native_contract.log'))
    codes.append(run([PY, '-m', 'unittest', 'discover', '-s', FRAME, '-p', 'test_*.py'],
                     folder / 'r13_contracts.log'))
    write(folder / 'status.json', dict(ctest_exit=codes[0], native_exit=codes[1],
          r13_python_exit=codes[2], replaces_gpu_unit_failures=False))
    return int(any(codes))


def register():
    env = json.loads((HERE / 'launch_env.json').read_text())
    env['TILEMEGA_E1_ATTEMPT'] = '_r3'
    env['TILEMEGA_R13_COMPILER_COMMIT'] = subprocess.check_output(
        ['git', 'rev-parse', '17860b184'], text=True).strip()
    write(HERE / 'launch_env_r3.json', env)
    fixture = ROOT / 'docs/experiments/E2E_GEN/raw/export_bridge.json'
    write(HERE / 'raw/acceptance_01/fixture_recovery.json', dict(
          source='/root/shared-nvme/junhuipeng/TileMega/docs/experiments/E2E_GEN/raw/export_bridge.json',
          destination=str(fixture), sha256=sha(fixture), changes_expected_values=False))
    for model in ('llama', 'qwen3'):
        config = json.loads((HERE / f'{model}_r13_sm120.json').read_text())
        config['device']['cache_dir'] = str(TMP / 'cache' / (model + '-r3'))
        config['output']['dir'] = str(HERE / 'raw' / ('E1_' + model + '_r3'))
        write(HERE / f'{model}_r13_sm120_r3.json', config)
    steps = []

    def add(name, command, gpu=False, after=(), after_any=(), timeout=14400):
        command = list(map(str, command))
        if not gpu:
            command = ['flock', str(WORK / 'gpu.lock'), 'env', 'TILEMEGA_GPU_LOCK_HELD=1'] + command
        steps.append(dict(name=name, command=command, env=env, cwd=str(ROOT), gpu=gpu,
                          after=list(after), after_any=list(after_any), timeout_s=timeout,
                          needs_free_mib=12288, priority=len(steps), out=str(HERE / 'raw' / name)))

    script = HERE / 'advance.py'
    add('E0_native_build_r3', [sys.executable, script, 'rebuild'], timeout=7200)
    add('E0_contracts_r3', [sys.executable, script, 'contracts'], after=['E0_native_build_r3'], timeout=7200)
    for model in ('llama', 'qwen3'):
        add('E1_' + model + '_r3', [PY, '-m', 'tilemega', 'calibrate', '--config',
            HERE / f'{model}_r13_sm120_r3.json'], gpu=True, after=['E0_native_build_r3'],
            after_any=['E0_contracts_r3'], timeout=21600)
    add('E1_TL2_r3', [PY, HERE / 'bootstrap.py', 'ceiling'], gpu=True,
        after=['E1_llama_r3', 'E1_qwen3_r3'], timeout=14400)
    micros = []
    for suite in 'bcdef':
        name = 'E2a_MB-1' + suite + '_r3'
        micros.append(name)
        folder = HERE / 'raw' / name
        folder.mkdir(exist_ok=True)
        add(name, [BUILD / 'tilemega-loadbench', '--suite', suite, '--out', folder / 'loadbench.json'],
            gpu=True, after=['E1_TL2_r3'], timeout=7200)
    add('E3_exports_r3', [PY, script, 'exports'], after=micros, timeout=14400)
    add('E3_fixed_r3', [PY, script, 'fixed'], after=['E3_exports_r3'], timeout=43200)
    filename = 'queue_acceptance_r3.json'
    write(HERE / filename, steps)
    dest = WORK / 'queue' / filename
    if dest.exists():
        raise RuntimeError('queue already published')
    temp = dest.with_suffix('.tmp')
    write(temp, steps)
    temp.replace(dest)
    print('published', len(steps), 'r3 steps without resetting history', flush=True)
    return 0


def exports():
    from tilemega.cli import Run, read_config
    sources = {}
    for model in ('llama', 'qwen3'):
        config = read_config(HERE / f'{model}_r13_sm120_r3.json')
        config['output']['dir'] = str(HERE / 'raw/E3_exports_r3' / model)
        driver = Run(config, os.environ['TILEMEGA_BIN'])
        sources[model] = {}
        for phase in ('prefill', 'decode'):
            key, directory = driver.export(phase)
            sources[model][phase] = dict(key=key, bridge=str(directory / 'bridge.json'),
                                       sha256=sha(directory / 'bridge.json'))
    write(HERE / 'exports_r3.json', sources)
    return 0


def make_jobs():
    sources = json.loads((HERE / 'exports_r3.json').read_text())
    jobs = []
    for model in ('llama', 'qwen3'):
        for batch in (1, 16):
            cell = f'{model}_B{batch}'
            inputs = HERE / 'inputs' / cell
            labels = [('N-R12b-120', 'N-R12b-noWD', 'off', 'identity', None),
                      ('N-R12b-120-pdl0', 'N-R12b-noWD', 'auto', 'identity', None),
                      ('P-R12bN-120', 'N-R12b-noWD', 'off', 'paged', None),
                      ('P-R12bN-120-pdl', 'N-R12b-noWD', 'auto', 'paged', None),
                      ('PS-120', 'PS', 'off', 'identity', 0),
                      ('PSA-120', 'PS', 'off', 'identity', 1),
                      ('PF-120', 'PF-R10-noWD', 'off', 'identity', None)]
            if cell != 'qwen3_B16':
                labels.append(('N-R12bh-120', 'N-R12bh-noWD', 'off', 'head', None))
            for label, donor, pdl, projection, la in labels:
                phase = 'prefill' if label == 'PF-120' else 'decode'
                paged = projection == 'paged' or donor == 'PS'
                options = dict(watchdog=0, handoff='off', pg='pages' if paged else 'l2',
                    weight_layout='tiled' if paged else 'row', sync='calibrated',
                    lookahead_bytes=0, pdl=pdl, l2_slim=0, page_loop_split=0,
                    nonpaged_weight_layout='row', evict_first=0, evict_last=1)
                if paged:
                    options['page_bytes'] = 16384
                if la is not None:
                    options['paged_la_splitk'] = la
                jobs.append(dict(cell=cell, model=model, label=label, phase=phase, batch=batch,
                    manifest=str(inputs / donor / 'plan.json'), classes=str(inputs / donor / 'classes.tsv'),
                    export=sources[model][phase]['bridge'], projection=projection, overrides=options,
                    out=str(TMP / 'fixed_r3' / cell / label)))
    return jobs


def clone(row, label, defines, pages=False):
    sys.path.insert(0, str(FRAME))
    from fixed_builds import trace_command
    folder = Path(row['out']).parent / label
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / 'plan.so'
    if pages:
        cmd = trace_command(Path(row['so'] + '.build_command.txt').read_text(), dest)
    else:
        cmd = shlex.split(Path(row['so'] + '.build_command.txt').read_text())
        cmd[cmd.index('-o') + 1] = str(dest)
    cmd += ['-D' + definition for definition in defines]
    record = dict(row, label=label, so=str(dest), out=str(folder), exit_code=1,
                  source_binary_sha256=row['sha256'], added_defines=defines)
    record['exit_code'] = run(cmd, folder / 'build.log')
    if record['exit_code'] == 0:
        record['sha256'] = sha(dest)
        for suffix in ('.plan.json', '.cu', '.classes.tsv'):
            source = Path(row['so'] + suffix)
            if source.exists():
                subprocess.run(['cp', str(source), str(dest) + suffix], check=True)
        Path(str(dest) + '.build_command.txt').write_text(shlex.join(cmd) + '\n')
    write(folder / 'record.json', record)
    write(HERE / 'raw/E3_fixed_r3' / row['cell'] / label / 'record.json', record)
    return record


def fixed():
    sys.path.insert(0, str(FRAME))
    import builds_r13 as builder
    builder.TARGET = HERE / 'target_sm120.json'
    old_project = builder.project

    def project(data, kind, page_bytes=16384):
        result = old_project(data, kind, page_bytes)
        if kind == 'paged':
            result['gemms'][-1].update(tile_n=128, tile_k=64, stages=2, split_k=1)
        return result

    builder.project = project
    jobs = make_jobs()
    write(HERE / 'jobs_sm120_r3.json', jobs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(builder.one, jobs))
    for row in rows:
        write(HERE / 'raw/E3_fixed_r3' / row['cell'] / row['label'] / 'record.json', row)
    write(HERE / 'builds_sm120_r3.json', rows)
    additional = []
    for row in list(rows):
        if row['exit_code'] or row['phase'] != 'decode':
            continue
        if row['label'] == 'N-R12b-120-pdl0':
            pdl1 = clone(row, 'N-R12b-120-pdl1', ['TILEMEGA_PDL_TRIGGER=1'])
            additional.append(pdl1)
            if not pdl1['exit_code']:
                additional.append(clone(pdl1, 'N-R12b-120-pdl1-step', ['TILEMEGA_TRACE_STEP=1']))
        elif row['label'] == 'N-R12b-120':
            additional.append(clone(row, row['label'] + '-trace', ['TILEMEGA_TRACE_STAGE=1', 'TILEMEGA_TRACE_STEP=1']))
            additional.append(clone(row, row['label'] + '-v2', ['TILEMEGA_TRACE_V2=1']))
        elif row['label'] == 'P-R12bN-120':
            additional.append(clone(row, row['label'] + '-trace', ['TILEMEGA_TRACE_STAGE=1', 'TILEMEGA_TRACE_STEP=1']))
            additional.append(clone(row, row['label'] + '-pages', ['TILEMEGA_TRACE_STEP=1'], pages=True))
        elif row['label'] in ('PS-120', 'PSA-120'):
            additional.append(clone(row, row['label'] + '-pages', ['TILEMEGA_TRACE_STEP=1'], pages=True))
    rows += additional
    write(HERE / 'builds_sm120_r3.json', rows)
    print('fixed builds', sum(r['exit_code'] == 0 for r in rows), '/', len(rows), flush=True)
    # Rejections are retained per arm, not converted to passed tests.
    return 0 if any(r['exit_code'] == 0 for r in rows) else 3


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('audit', 'rebuild', 'contracts', 'register', 'exports', 'fixed'))
    a = p.parse_args()
    return {'audit': audit, 'rebuild': rebuild, 'contracts': contracts,
            'register': register, 'exports': exports, 'fixed': fixed}[a.action]()


if __name__ == '__main__':
    raise SystemExit(main())
