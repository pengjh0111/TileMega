#!/usr/bin/env python3
"""Resume persisted R13 evidence after loss of the ephemeral runtime tree."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from bootstrap import HERE, ROOT, FRAME, WORK, BUILD, TMP, PY, write, sha, run

PERSIST = Path('/root/shared-nvme/junhuipeng/TileMega_R13_SM120')


def section_stamps_match(data, expected):
    actual={name:row.get('stamp') for name,row in data.get('calibration_sections',{}).items()}
    return actual==expected


def prepare():
    for directory in (TMP/'models', TMP/'tmp', TMP/'hf', WORK/'queue'):
        directory.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(FRAME))
    from gpu_guard import gpu, idle
    old = json.loads((HERE/'guard_policy.json').read_text())
    samples = []
    for index in range(6):
        row = gpu()
        if not idle(row, old, 12288):
            raise RuntimeError('post-recovery GPU is not idle under the unchanged guard')
        samples.append(row)
        if index < 5:
            time.sleep(old['interval_s'])
    policy = dict(old, idle_power_w=statistics.median(r['power_w'] for r in samples))
    write(HERE/'raw/recovery_03/idle_power.json', samples)
    write(HERE/'guard_policy.json', policy)
    measurement = json.loads((HERE/'measurement_policy.json').read_text())
    measurement['idle_power_w'] = policy['idle_power_w']
    write(HERE/'measurement_policy.json', measurement)
    state = json.loads((WORK/'scheduler/state.json').read_text())
    if any(row['status']=='running' for row in state.values()):
        raise RuntimeError('a running historical node needs explicit child reconciliation')
    code = run([sys.executable, ROOT/'python/tilemega/fingerprint.py', '--check',
                BUILD/'tools/tilemega'], HERE/'raw/recovery_03/fingerprint.log')
    if code:
        return code
    write(HERE/'raw/recovery_03/start.json', dict(
        time=time.time(), head=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        device_recovery='performed by user; no agent reset', historical_state=state,
        ephemeral_tree_lost=True, native_target_sha256=sha(HERE/'target_sm120.json'),
        original_loadbench_sha256=sha(BUILD/'tilemega-loadbench'),
        exclusions=['MB-1c TN128/TK64 method5 only; known fatal fault, root cause unresolved'],
        exclusion_is_not_kernel_fix=True, solver_changes=False, no_push=True))
    print('post-recovery idle median',policy['idle_power_w'],flush=True)
    return 0


def restore():
    from tilemega.fingerprint import calibration_stamps
    from tilemega.cli import missing_serving_fit_sections
    target = HERE/'target_sm120.json'
    data = json.loads(target.read_text())
    if (not section_stamps_match(data, calibration_stamps()) or missing_serving_fit_sections(data)
            or not data['calibration_by_dtype']['bf16']['calibrated']):
        raise RuntimeError('the archived native target is not valid for the unchanged compiler sources')
    records = []
    for model in ('llama','qwen3'):
        doctor = json.loads((HERE/f'raw/E1_{model}_r3/doctor.json').read_text())
        destination = Path(doctor['target'])
        if destination.exists():
            raise RuntimeError('refuse to overwrite an existing calibration cache')
        destination.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['cp',str(target),str(destination)],check=True)
        subprocess.run(['cp',str(target)+'.calibration.json',str(destination)+'.calibration.json'],check=True)
        records.append(dict(model=model,path=str(destination),sha256=sha(destination),
                            source=str(target),source_is_shared_gpu_snapshot=True))
    # Rebuild only a host-side runner; the original kernels and binary stay intact.
    binary = TMP/'safe-shapes'
    command = ['/usr/local/cuda-12.8/bin/nvcc','-O3','-DNDEBUG','--expt-relaxed-constexpr',
               '--generate-code=arch=compute_120,code=[compute_120,sm_120]',
               '-D_GNU_SOURCE','-D_GLIBCXX_USE_CXX11_ABI=1',
               '-I'+str(BUILD/'include'),'-I'+str(ROOT/'include'),
               '-I'+str(ROOT/'third_party/cutlass/include'),
               HERE/'loadbench_safe_shapes.cu','-o',binary]
    code = run(command,HERE/'raw/recovery_03/safe_shapes_build.log')
    write(HERE/'raw/recovery_03/target_restore.json',records)
    if not code:
        write(HERE/'raw/recovery_03/safe_shapes_binary.json',dict(path=str(binary),sha256=sha(binary),
              kernel_source=str(ROOT/'tools/experimental/loadbench/kernels.cuh'),
              kernel_sha256=sha(ROOT/'tools/experimental/loadbench/kernels.cuh'),
              kernel_modified=False,quarantined_points=12,supported_points=204))
    return code


def model(name):
    from huggingface_hub import snapshot_download
    source=json.loads((HERE/f'raw/E0_model_{name}/source.json').read_text())
    folder='llama3_2_1b' if name=='llama' else 'qwen3_1_7b'
    persistent=PERSIST/'models'/folder
    snapshot_download(repo_id=source['source'],revision=source['revision'],local_dir=persistent,
                      allow_patterns=['*.json','*.safetensors','tokenizer.model','*.tiktoken','*.txt'],max_workers=3)
    weights={p.name:sha(p) for p in sorted(persistent.glob('*.safetensors'))}
    if weights!=source['safetensors_sha256'] or sha(persistent/'config.json')!=source['config_sha256']:
        raise RuntimeError('restored model does not match the recorded fixed model hashes')
    destination=TMP/'models'/folder
    if destination.exists() or destination.is_symlink():
        raise RuntimeError('refuse to replace an existing runtime model path')
    destination.symlink_to(persistent,target_is_directory=True)
    write(HERE/f'raw/E0_model_{name}_r4/source.json',dict(source, persistent_path=str(persistent),
          destination=str(destination),config_sha256=sha(persistent/'config.json'),
          safetensors_sha256=weights,matches_pre_recovery=True))
    return 0


def joint(name):
    destination=HERE/f'raw/E3_R13F_{name}_r4'
    start=time.monotonic()
    code=run([PY,'-m','tilemega','build','--config',HERE/f'{name}_r13_sm120_r3.json',
              '--run-dir',destination/'run'],destination/'build.log')
    elapsed=time.monotonic()-start
    write(destination/'duration.json',dict(seconds=elapsed,configured_time_budget_s=1800,
          exceeds_twice_budget=elapsed>3600,aborted_at_twice_budget=False,exit_code=code))
    return code


def smoke():
    rows=json.loads((HERE/'builds_sm120_r3.json').read_text())
    for name in ('llama','qwen3'):
        path=HERE/f'raw/E3_R13F_{name}_r4/run/plans.json'
        if path.exists():
            for batch,pair in json.loads(path.read_text()).items():
                rows.append(dict(cell=f'{name}_B{batch}',model=name,batch=int(batch),phase='decode',
                                 label='R13F-120',so=pair['decode'],exit_code=0))
    results=[]
    for row in rows:
        if row['phase']!='decode' or row['exit_code']:
            continue
        folder=HERE/'raw/E2b_smoke_r4'/row['cell']/row['label']
        model_path=json.loads((HERE/f"{row['model']}_r13_sm120_r3.json").read_text())['model']['path']
        command=[PY,'-m','tilemega.serving.smoke','--so',row['so'],'--model',model_path,
                 '--batch',str(row['batch']),'--steps','64','--out',folder]
        code=run(['timeout','--kill-after=15s','300s',*map(str,command)],folder/'stdout.log')
        report=folder/'smoke.json'
        results.append(dict(cell=row['cell'],label=row['label'],so=row['so'],sha256=sha(row['so']),
                            exit_code=code,report=json.loads(report.read_text()) if report.exists() else None))
        write(HERE/'raw/E2b_smoke_r4/results.json',results)
        sys.path.insert(0,str(FRAME))
        from gpu_guard import gpu
        status=gpu()
        if status['utilization_pct']>5 and not status['owners']:
            raise RuntimeError('ownerless GPU saturation after smoke; preserve evidence and stop')
    write(HERE/'raw/E2b_smoke_r4/summary.json',dict(attempted=len(results),
          passed=sum(r['exit_code']==0 for r in results),failed=sum(r['exit_code']!=0 for r in results),
          all_passed=bool(results) and all(r['exit_code']==0 for r in results),
          note='Queue completion is not a claim that every arm passed'))
    return 0 if results else 3


def register():
    env=json.loads((HERE/'launch_env_r3.json').read_text())
    rows=[]
    def add(name,command,gpu=False,after=(),after_any=(),priority=20,timeout=14400):
        command=list(map(str,command))
        if not gpu:
            command=['flock',str(WORK/'gpu.lock'),'env','TILEMEGA_GPU_LOCK_HELD=1',*command]
        rows.append(dict(name=name,command=command,env=env,cwd=str(ROOT),gpu=gpu,
                         after=list(after),after_any=list(after_any),priority=priority,
                         timeout_s=timeout,needs_free_mib=12288,out=str(HERE/'raw'/name)))
    script=HERE/'resume.py'
    add('E0_restore_r4',[PY,script,'restore'],priority=-100,timeout=7200)
    for index,name in enumerate(('llama','qwen3')):
        add(f'E0_model_{name}_r4',[PY,script,'model','--model',name],
            after=['E0_restore_r4'],priority=-99+index,timeout=14400)
    add('E2a_MB-1c_r4',[TMP/'safe-shapes','--out',HERE/'raw/E2a_MB-1c_r4/loadbench.json'],
        gpu=True,after=['E0_restore_r4'],priority=8,timeout=7200)
    micro=['E2a_MB-1c_r4']+[f'E2a_MB-1{x}_r3' for x in 'def']
    models=[f'E0_model_{name}_r4' for name in ('llama','qwen3')]
    add('E3_exports_r4',[PY,HERE/'advance.py','exports'],after=models,after_any=micro,priority=20)
    add('E3_fixed_r4',[PY,HERE/'advance.py','fixed'],after=['E3_exports_r4'],priority=21,timeout=86400)
    for index,name in enumerate(('llama','qwen3')):
        add(f'E3_R13F_{name}_r4',[PY,script,'joint','--model',name],gpu=True,
            after=['E3_fixed_r4'],priority=22+index,timeout=345600)
    add('E2b_smoke_r4',[PY,script,'smoke'],gpu=True,after=['E3_fixed_r4'],
        after_any=[f'E3_R13F_{name}_r4' for name in ('llama','qwen3')],priority=25,timeout=86400)
    write(HERE/'queue_resume_r4.json',rows)
    destination=WORK/'queue/queue_resume_r4.json'
    if destination.exists():
        raise RuntimeError('recovery queue already published')
    temporary=destination.with_suffix('.tmp')
    write(temporary,rows)
    temporary.replace(destination)
    print('published',len(rows),'new steps; original failures and skipped nodes retained')
    return 0


def launch():
    pidfile=WORK/'scheduler/scheduler.pid'
    if pidfile.exists():
        old_pid=int(pidfile.read_text())
        command_line=Path('/proc')/str(old_pid)/'cmdline'
        if command_line.exists() and b'scheduler.py' in command_line.read_bytes():
            raise RuntimeError('refuse to launch a duplicate scheduler')
    code=subprocess.run(['flock','--nonblock',str(pidfile),'true']).returncode
    if code:
        raise RuntimeError('scheduler singleton lock is still held')
    environment=dict(os.environ,**json.loads((HERE/'launch_env_r3.json').read_text()))
    command=[sys.executable,FRAME/'scheduler.py','--queue-dir',WORK/'queue',
             '--out',WORK/'scheduler','--policy',HERE/'guard_policy.json','--deadline-hours','96']
    with (WORK/'scheduler.log').open('a') as log:
        process=subprocess.Popen(list(map(str,command)),cwd=ROOT,env=environment,
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(HERE/'raw/recovery_03/launch.json',dict(pid=process.pid,command=list(map(str,command)),
          time=time.time(),preserved_scheduler_state=True))
    print('resumed scheduler',process.pid,flush=True)
    return 0


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','restore','model','joint','smoke','register','launch'))
    parser.add_argument('--model',choices=('llama','qwen3'))
    args=parser.parse_args()
    return {'prepare':prepare,'restore':restore,'model':lambda:model(args.model),
            'joint':lambda:joint(args.model),'smoke':smoke,'register':register,'launch':launch}[args.action]()


if __name__=='__main__':
    raise SystemExit(main())
