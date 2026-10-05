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


def recover_anchor():
    from pipeline import runnable
    cell='qwen3_B16'
    folder=HERE/'raw/E4a_qwen3_B16_r1_recovery_r6'
    arms={cell:[runnable(cell,label) for label in ('vllm','B0','R13F')]}
    for arm in arms[cell]:
        if arm['kind']=='tm':arm['step_events']=1
    write(folder/'arms.json',arms)
    code=run([PY,FRAME/'anchor.py','--arms',folder/'arms.json','--cell',cell,'--round','1','--out',folder],folder/'anchor.log')
    if code:return code
    path=folder/cell/'round1.json'
    data=json.loads(path.read_text())
    if any(row.get('exit_code') or 'e2e_seconds' not in row for row in data['arms'].values()):return 1
    write(FOLDER/'anchor_execution_replacements.json',{'E4a:qwen3_B16:1':dict(path=str(path),
          guard=str(folder/'guard_result.json'),reason='Original wrapper SyntaxError before any model measurement; one missing round, not fastest-value selection')})
    return 0


def register_recovery():
    state=json.loads((WORK/'scheduler/state.json').read_text())
    failed='E4a_qwen3_B16_r1_r5'
    source=HERE/f'raw/{failed}'
    if state[failed]['status']!='failed' or list(source.glob('*/round*.json')):
        raise RuntimeError('recovery requires the documented premeasurement failure, not a measured round')
    if 'SyntaxError' not in (source/'anchor.log').read_text():raise RuntimeError('unexpected failure')
    name='E4a_qwen3_B16_r1_recovery_r6'
    path=WORK/'queue/queue_complete_r5.json'
    rows=json.loads(path.read_text())
    original=next(row for row in rows if row['name']==failed)
    node=dict(original,name=name,command=[PY,str(HERE/'acceptance.py'),'recover-anchor'],
              after=['E3_catalog_r5'],after_any=[failed],priority=39,out=str(HERE/'raw'/name))
    for row in rows:
        if row['name'].startswith('E4b_') and state[row['name']]['status']=='pending':
            row['after_any'].append(name)
    destination=WORK/'queue/queue_anchor_recovery_r6.json'
    if destination.exists():raise RuntimeError('refuse to publish the recovery twice')
    # Gate pending consumers before publishing the producer; both writes are atomic.
    write(FOLDER/'r5_dependency_update.json',rows)
    temporary=path.with_suffix('.pending')
    write(temporary,rows);temporary.replace(path)
    write(HERE/'queue_anchor_recovery_r6.json',[node])
    temporary=destination.with_suffix('.pending')
    write(temporary,[node]);temporary.replace(destination)
    write(FOLDER/'recovery_publication.json',dict(time=time.time(),node=name,original_state=state[failed],
          original_failure_sha256=sha(source/'anchor.log'),original_failure_preserved=True,
          prior_model_measurements=0,expected_measured_round_count_unchanged=True))
    return 0


if __name__ == '__main__':
    action = sys.argv[1]
    raise SystemExit({'align': align, 'replay': replay, 'inspect': inspect, 'anchor-replay': anchor_replay,
                     'tests': tests, 'register-recovery': register_recovery, 'recover-anchor': recover_anchor}[action]())
