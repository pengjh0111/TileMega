#!/usr/bin/env python3
"""Reconcile the final upstream baseline without rebuilding measured binaries."""
import json
from pathlib import Path
import subprocess
import sys
import time

from bootstrap import HERE, ROOT, FRAME, WORK, BUILD, PY, sha, write, run

FINAL = '9aebaf655'
CHECKPOINT = '36f17e6ee'
FOLDER = HERE / 'raw/acceptance_03'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def configuration_matches(local, upstream):
    # Paths and already selected sm89 binaries are not inputs to the sm120 search.
    solver = lambda data: {key: value for key, value in data['solver'].items() if key != 'prefill_pins'}
    return local['features'] == upstream['features'] and solver(local) == solver(upstream)


def align():
    final = git('rev-parse', FINAL)
    subprocess.run(['git', 'merge-base', '--is-ancestor', final, 'HEAD'], cwd=ROOT, check=True)
    changed = git('diff', '--name-only', CHECKPOINT, final, '--',
                  'include', 'lib', 'tools', 'python', 'CMakeLists.txt').splitlines()
    if changed:
        raise RuntimeError('final upstream device/compiler sources changed: ' + str(changed))
    code = run([PY, ROOT/'python/tilemega/fingerprint.py', '--check', BUILD/'tools/tilemega'],
               FOLDER/'source_fingerprint.log')
    if code:
        return code
    from tilemega.fingerprint import calibration_stamps
    target = json.loads((HERE/'target_sm120.json').read_text())
    stamps = {key: row['stamp'] for key, row in target['calibration_sections'].items()}
    if stamps != calibration_stamps():
        raise RuntimeError('calibration section stamps changed after upstream alignment')
    configs = {}
    for model in ('llama', 'qwen3'):
        local = HERE/f'{model}_r13_sm120_r3.json'
        upstream = ROOT/f'configs/e2e/{model}_r13_final.json'
        configs[model] = dict(local_sha256=sha(local), upstream_sha256=sha(upstream),
                             features_and_solver_equal=configuration_matches(
                                 json.loads(local.read_text()), json.loads(upstream.read_text())))
    if not all(row['features_and_solver_equal'] for row in configs.values()):
        raise RuntimeError('SL-5 inputs differ from the final R13 retained features')
    port_diff = git('diff', '--name-only', final, 'HEAD', '--', 'include', 'lib', 'tools', 'python').splitlines()
    if set(port_diff)-{'lib/Target/TargetSpec.cpp', 'python/tilemega/cli.py'}:
        raise RuntimeError('unexpected code change beyond the recorded native-profile port')
    catalog=json.loads((HERE/'catalog.json').read_text())
    binaries={entry['path']:entry['sha256'] for rows in catalog.values() for arm in rows
              for entry in arm.get('binaries',{}).values()}
    for path,expected in binaries.items():
        if sha(path)!=expected:raise RuntimeError('measured binary changed: '+path)
    record = dict(evidence='verified CPU source/configuration/fingerprint checks', time=time.time(),
                  initial_checkpoint=git('rev-parse', CHECKPOINT), sm89_final_head=final,
                  local_alignment_head=git('rev-parse', 'HEAD'), final_baseline_merged=True,
                  compiler_runtime_aligned=True, upstream_core_changed_since_checkpoint=changed,
                  allowed_native_profile_port_files=port_diff, calibration_stamps_match=True,
                  target_sha256=sha(HERE/'target_sm120.json'), configurations=configs,
                  measured_binaries_rebuilt=False, existing_binary_hashes_preserved=True,
                  preserved_binary_count=len(binaries),
                  cross_architecture_model_hashes_verified=False,
                  model_qualification='sm89 config/weight SHA256 not present in final portable evidence; not assumed equal',
                  vllm_versions=dict(sm120='0.29.0', sm89='0.30.0'),
                  sm89_qualification='Final original three rounds retain two canary flags; cancelled reruns not substituted',
                  no_solver_or_cost_model_changes=True, no_push=True)
    write(HERE/'baseline_alignment.json', record)
    write(FOLDER/'baseline_alignment.json', record)
    return 0


def replay():
    sys.path.insert(0, str(FRAME))
    from ptx_pdl_check import check
    results = []
    old = json.loads((HERE/'raw/E3_codegen_r5/audit.json').read_text())
    for row in old:
        if 'pdl_position_exit' not in row:
            continue
        path = HERE/'raw/E3_codegen_r5'/row['cell']/row['label']/'native.ptx'
        trigger = int(row['label'].endswith('pdl1'))
        try:
            report = dict(pass_=True, entries=check(path.read_text(), trigger))
        except (ValueError, OSError) as error:
            report = dict(pass_=False, error=str(error))
        results.append(dict(cell=row['cell'], label=row['label'], trigger=trigger,
                            ptx_sha256=sha(path) if path.exists() else None,
                            original_position_exit=row['pdl_position_exit'], report=report,
                            check_kind='CPU replay of existing PTX, not a new GPU launch'))
    write(FOLDER/'pdl_replay.json', results)
    print('existing sm120 PTX replay', sum(row['report']['pass_'] for row in results), '/', len(results))
    return int(not results or not all(row['report']['pass_'] for row in results))


def anchor_replay():
    sys.path.insert(0,str(FRAME))
    from anchor import replay_vllm_record
    replacements={}
    records=[]
    for source in sorted((HERE/'raw').glob('E4a_*_r5/*/round*.json')):
        data=json.loads(source.read_text())
        if data.get('invalidated'):continue
        changed=False
        for label,row in data['arms'].items():
            recovered=replay_vllm_record(row)
            if recovered==row:continue
            data['arms'][label]=recovered
            changed=True
            records.append(dict(source=str(source),original_sha256=sha(source),label=label,
                                original_exit_code=row['exit_code'],replayed_exit_code=recovered['exit_code']))
        if changed:
            path=FOLDER/'rounds'/source.parent.parent.name/source.parent.name/source.name
            write(path,data)
            key=f"E4a:{data['cell']}:{data['round']}"
            replacements[key]=dict(path=str(path),original_sha256=sha(source),original_path=str(source))
    write(FOLDER/'anchor_replacements.json',replacements)
    write(FOLDER/'anchor_replay.json',dict(records=records,gpu_remeasurements=0,
          original_files_overwritten=False,qualification='Only the erroneous TM-only loop postcondition is replayed; original command, metrics and clean guard required'))
    print('clean existing vLLM wrapper records replayed',len(records))
    return 0


def inspect():
    state = json.loads((WORK/'scheduler/state.json').read_text())
    failures = []
    for row in json.loads((HERE/'builds_sm120_r3.json').read_text()):
        if not row['exit_code']:
            continue
        log = Path(row['out'])/'build.log'
        failures.append(dict(cell=row['cell'], label=row['label'], exit_code=row['exit_code'],
                             path=str(log), log_sha256=sha(log), diagnostic=log.read_text()[-4096:],
                             classification='solver/cost-model rejection; no solver modification permitted'))
    joints = []
    for model in ('llama', 'qwen3'):
        folder = HERE/f'raw/E3_R13F_{model}_r4'
        log = folder/'run/commands/build-decode-pages-B1/stderr.txt'
        joints.append(dict(model=model, duration=json.loads((folder/'duration.json').read_text()),
                           diagnostic=log.read_text(), stderr_sha256=sha(log),
                           final_config_inputs_unchanged=True,
                           disposition='unavailable; no unchanged-input retry or solver workaround'))
    write(FOLDER/'snapshot.json', dict(time=time.time(), state=state, fixed_failures=failures,
          joint_failures=joints, performance_queue_not_finished=True,
          solver_location='lib/Solver/StageFlowModel.cpp:205, MainStart: DRAM traffic minus averaged prefetched traffic is negative',
          root_cause_qualification='located rejection, not yet a verified explanation for the model imbalance',
          no_device_reset=True, no_push=True))
    return 0


def tests():
    results=[]
    for label,path in (('sm120',HERE),('r13',FRAME)):
        code=run(['env','PYTHONPATH='+str(HERE)+':'+str(ROOT/'python'),PY,'-m','unittest','discover','-s',path,'-p','test_*.py'],FOLDER/(label+'_host_tests.log'))
        results.append(dict(suite=label,exit_code=code,path=str(FOLDER/(label+'_host_tests.log'))))
    write(FOLDER/'host_tests.json',results)
    return int(any(row['exit_code'] for row in results))


if __name__ == '__main__':
    action = sys.argv[1]
    raise SystemExit({'align': align, 'replay': replay, 'inspect': inspect, 'anchor-replay': anchor_replay, 'tests': tests}[action]())
