#!/usr/bin/env python3
"""Reuse compiled plans for the user-requested E2E-only queue with GPU guards."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def gpu():
    raw = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.used,power.draw,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True).strip()
    memory, power, util = map(float, raw.split(','))
    apps = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid,used_memory',
                                    '--format=csv,noheader,nounits'], text=True)
    owners = {int(row.split(',')[0]): int(row.split(',')[1]) for row in apps.splitlines() if row.strip()}
    return dict(time=time.time(), used_mib=memory, power_w=power, utilization_pct=util,
                owners=owners, hidden_mib=max(0, memory-sum(owners.values())))


def descendants(pid):
    own = {pid}
    parents = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            parents[int(path.parent.name)] = int(path.read_text().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            pass
    while True:
        added = {child for child, parent in parents.items() if parent in own} - own
        if not added:
            return own
        own.update(added)


def idle(row, threshold):
    return not row['owners'] and row['used_mib'] <= 256 and row['power_w'] <= threshold and row['utilization_pct'] <= 5


def external(row, own):
    return bool(set(row['owners'])-own) or row['hidden_mib'] > 256


def stop(process, own):
    # Stop only this benchmark's descendants, never another user's workload.
    own = descendants(process.pid)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for pid in own:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        for pid in own:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=('llama','qwen3'),required=True)
    args=parser.parse_args();out=ROOT/'runs'/f'r12b-priority-{args.model}';out.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/f'configs/e2e/{args.model}_r12b.json').read_text())
    config['test'].update(hf_check=False,mode_check=False,vllm=True,warmup=1,repeats=3)
    config['workload']['batch']=[1,16];config['solver']['mode']='L2';config['output']['dir']=str(out)
    config_path=HERE/f'{args.model}_priority.json';config_path.write_text(json.dumps(config,indent=2)+'\n')
    plans=ROOT/f'runs/r12b-{args.model}/plans.json'
    data=json.loads(plans.read_text())
    for batch in ('1','16'):
        for phase in ('prefill','decode'):
            if not Path(data[batch][phase]).is_file():raise FileNotFoundError(data[batch][phase])
    (out/'plans.json').write_text(plans.read_text())
    policy=json.loads((ROOT/config['test']['policy_file']).read_text())
    threshold=policy['idle_power_w']+policy['power_margin_w']
    acceptance=out/'bench_acceptance.json'
    acceptance.write_text(json.dumps(dict(accepted=False,reason='awaiting guarded full benchmark'))+'\n')
    def record(row, **extra):
        with (out/'occupancy_guard.jsonl').open('a') as f:f.write(json.dumps(dict(row,**extra))+'\n')
    lock=Path(os.getenv('TILEMEGA_GPU_LOCK','/root/r12_work/serving_gpu.lock'))
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        for index in range(6):
            row=gpu();accepted=idle(row,threshold);record(row,phase='preflight',accepted=accepted,power_threshold_w=threshold)
            if not accepted:
                print('GPU occupied: defer this model (exit 75)',flush=True);return 75
            time.sleep(5)
    command=[sys.executable,'-m','tilemega','bench','--config',str(config_path),'--run-dir',str(out)]
    with (out/'bench.log').open('a') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        begin=time.monotonic();violations=0;own={process.pid}
        while True:
            try:
                code=process.wait(timeout=5);break
            except subprocess.TimeoutExpired:
                own.update(descendants(process.pid));row=gpu();bad=external(row,own)
                violations=violations+1 if bad else 0
                record(row,phase='during',external=bad,consecutive_violations=violations)
                if set(row['owners'])-own or violations>=3:
                    stop(process,own);print('External GPU activity detected: reject and retry (75)',flush=True);return 75
                if time.monotonic()-begin>10800:
                    stop(process,own);print('Benchmark timeout (124)',flush=True);return 124
    row=gpu();record(row,phase='after-process',exit_code=code)
    if external(row,own):
        print('External GPU activity at exit: reject and retry (75)',flush=True);return 75
    if code:
        errors='\n'.join(p.read_text(errors='replace')[-6000:] for p in (out/'commands').glob('bench-*/stderr.txt'))
        if code==75 or external(row,own) or 'out of memory' in errors.lower():
            print('Occupancy/initialization OOM: retry without accepting metrics (75)',flush=True);return 75
        return code
    code=subprocess.run([sys.executable,'-m','tilemega','report','--config',str(config_path),
                         '--run-dir',str(out)],cwd=ROOT).returncode
    if code==0:
        acceptance.write_text(json.dumps(dict(accepted=True,cells=[1,16],paired_rounds=3,
            power_threshold_w=threshold,preflight_samples=6,monitor_interval_seconds=5,
            invisible_allocation_limit_mib=256))+'\n')
    return code


if __name__=='__main__':raise SystemExit(main())
