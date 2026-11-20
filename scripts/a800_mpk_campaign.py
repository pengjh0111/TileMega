"""Drive finite single-A800 jobs from the login node, gating expansion on verified outputs."""
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time

WORK = Path(__file__).resolve().parents[2]
RUN = WORK/'TileMega-a800/runs/a800'
LOG = WORK/'logs'
TARGET = 'configs/targets/sm_80_a800_sxm4_80gb.json'
ORDER = ('llama3_2_1b', 'qwen3_0_6b', 'qwen3_1_7b', 'llama3_2_3b', 'llama3_1_8b')
CAP = 3


def put(path, value):
    tmp = path.with_suffix('.incoming')
    tmp.write_text(value)
    tmp.replace(path)


def run(args, timeout=30):
    return subprocess.run(args, cwd=WORK, text=True, capture_output=True, timeout=timeout)


def cohort(key):
    model, batch, repeat = key
    return f"mpk-v3-{model.replace('_', '-')}-b{batch}-r{repeat}"


def submit(key):
    model, batch, repeat = key
    tag = cohort(key)
    with (RUN/'collected/A800_baselines.csv').open() as f:
        hashes = {r['checkpoint_sha256'] for r in csv.DictReader(f)
                  if r['model'] == model and r['status'] == 'measured' and r['framework'] != 'mpk'}
    if len(hashes) != 1 or not next(iter(hashes)):
        raise RuntimeError(f'expected a shared baseline checkpoint for {model}')
    large = model == 'llama3_1_8b'
    seconds = 1800 if large else 1140
    prior = LOG/f'{tag}.guard-retry.json'
    if prior.exists():
        last = json.loads(prior.read_text())
        if time.time() < last['retry_after']:
            return False
    options = ['--cohort', tag, '--engine', 'mpk', '--model', model, '--batch', str(batch),
               '--round', str(repeat), '--expected-target', TARGET, '--expected-checkpoint-sha256', hashes.pop(),
               '--process-timeout-s', str(seconds), '--heartbeat-timeout-s', '420']
    plan = run(['./run-env.sh', 'tilemega', 'python', 'scripts/a800_baselines.py', 'plan',
                *options, '--budget-s', str(seconds+200)])
    plan.check_returncode()
    if json.loads(plan.stdout)['steps'] != [f'{model}_mpk_b{batch:02d}_r{repeat}']:
        raise RuntimeError('invalid MPK job scope')
    put(LOG/f'{tag}.plan.json', plan.stdout)
    duration = 35 if large else 24
    placement = [f'--time=00:{duration:02}:00']
    if repeat:
        pilot = RUN/'cohorts'/cohort((model, batch, 0))
        if not (pilot/'mpk-accepted.json').is_file():
            raise RuntimeError(f'missing validated timing pilot for {tag}')
        step = json.loads((pilot/'baseline/state.json').read_text())[f'{model}_mpk_b{batch:02d}_r0']
        # Keep twice the observed guarded runtime plus the wrapper and fresh
        # device-probe allowances. Slurm may choose the full requested limit.
        minutes = math.ceil((2*step['run_s'] + 356 + 75)/60)
        if minutes < duration:
            placement.append(f'--time-min=00:{minutes:02}:00')
    intent = LOG/f'{tag}.intent'
    with intent.open('x') as f:
        f.write(str(time.time()))
    reply = run(['sbatch', '--parsable', f'--job-name={tag}', *placement,
                 f'--output=logs/{tag}-%j.out', 'baseline-a800.sbatch', *options])
    if reply.returncode:
        if 'AssocMaxSubmitJobLimit' in reply.stderr or 'QOSMaxSubmitJobPerUserLimit' in reply.stderr:
            intent.unlink()
            return False
        raise RuntimeError(reply.stderr.strip())
    job = reply.stdout.strip().split(';')[0]
    if not job.isdecimal():
        raise RuntimeError(f'check Slurm accounting for submission {tag}')
    put(LOG/f'{tag}.id', job+'\n')
    intent.unlink()
    print('submitted', job, tag, flush=True)
    return True


def outcome(key, active):
    tag = cohort(key)
    folder = RUN/'cohorts'/tag
    if (folder/'mpk-accepted.json').is_file():
        return 'accepted'
    if (LOG/f'{tag}.rejected').is_file():
        return 'rejected'
    path = LOG/f'{tag}.id'
    if not path.is_file() or not path.read_text().strip():
        if (LOG/f'{tag}.intent').exists():
            raise RuntimeError(f'reconcile interrupted submission {tag} with Slurm before restarting')
        return 'new'
    job = path.read_text().strip()
    if job in active:
        return 'active'
    result = run(['sacct', '-XnP', '-j', job, '-o', 'State,ExitCode'])
    result.check_returncode()
    state = result.stdout.strip()
    if not state or state.split('|')[0] in ('PENDING','RUNNING','COMPLETING','CONFIGURING'):
        return 'active'
    if state != 'COMPLETED|0:0':
        scheduler = folder/'baseline/state.json'
        info = json.loads(scheduler.read_text()) if scheduler.exists() else {}
        model, batch, repeat = key
        names = ('a800_device_identity', f'{model}_mpk_b{batch:02d}_r{repeat}')
        records = [info.get(name, {}) for name in names]
        guards = []
        for row in records:
            if row.get('out'):
                where = Path(row['out']).relative_to('/work')
                proof = WORK/where/'guard_result.json'
                if proof.is_file():
                    guards.append(json.loads(proof.read_text()))
        retry = LOG/f'{tag}.guard-retry.json'
        history = json.loads(retry.read_text())['jobs'] if retry.exists() else []
        # Parent scheduler records 75 for a guard refusal. Backend timeouts or
        # application exceptions require review and cannot trigger a retry here.
        temporary = [g for g in guards if g.get('code') == 75]
        if temporary and not any(r.get('status')=='failed' for r in records) and len(history)<2:
            if job not in history:
                history.append(job)
            after = max([time.time()+45] + [r.get('retry_at',0) for r in records])
            put(retry, json.dumps(dict(jobs=history, retry_after=after, guard=temporary[-1]))+'\n')
            path.rename(LOG/f'{tag}.guard-{job}.id')
            print('allocation released; retry scheduled', job, tag, flush=True)
            return 'new'
    if state == 'COMPLETED|0:0':
        check = run(['./run-env.sh', 'tilemega', 'python', 'scripts/a800_mpk_verify.py', tag], 120)
        if not check.returncode:
            print('verified', job, tag, check.stdout.strip(), flush=True)
            return 'accepted'
        state += '\n' + check.stderr
    put(LOG/f'{tag}.rejected', f'{job} {state}\n')
    print('requires review', job, tag, state.splitlines()[-1], flush=True)
    return 'rejected'


def dependencies(key):
    model, batch, repeat = key
    family = 'qwen3_0_6b' if model.startswith('qwen') else 'llama3_2_1b'
    return {(family, 1, 0), (model, 1, 0), (model, batch, 0)} - {key}


def main():
    preflight = json.loads((RUN/'mpk/preflight.json').read_text())
    supported = {(r['model'], r['batch']) for r in preflight['cells'] if r['status']=='requires_gpu'}
    cells = [(m,b,r) for b in (1,2,4,8,16) for r in range(3) for m in ORDER if (m,b) in supported]
    if len(cells) != 3*len(supported):
        raise ValueError('unsupported model in MPK campaign')
    end = time.monotonic()+48*3600
    before = {}
    refresh = None
    refresh_pending = False
    with (LOG/'mpk-campaign.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        while time.monotonic() < end:
            queue = run(['squeue', '-h', '-u', os.environ['USER'], '-o', '%A'])
            queue.check_returncode()
            active = set(queue.stdout.split())
            state = {key: outcome(key,active) for key in cells}
            if state != before:
                refresh_pending = True
                before = state.copy()
            if refresh is not None and refresh.poll() is not None:
                if refresh.returncode:
                    refresh_pending = True
                    if refresh.returncode != 75:
                        print('collector exit', refresh.returncode, flush=True)
                refresh = None
            if refresh_pending and refresh is None:
                # Collection can spend minutes validating unrelated native
                # receipts on the shared filesystem. It holds no GPU.
                with (LOG/'mpk-refresh.log').open('a') as report:
                    refresh = subprocess.Popen(['timeout','--kill-after=10s','900s',
                        './refresh-results-a800.sh','--if-idle'], cwd=WORK,
                        stdout=report, stderr=subprocess.STDOUT, start_new_session=True)
                refresh_pending = False
            inflight = sum(v == 'active' for v in state.values())
            ready = [k for k in cells if state[k]=='new' and
                     all(state[dep]=='accepted' for dep in dependencies(k))]
            if not ready and not inflight and not refresh_pending and refresh is None:
                print('finished runnable work', {v:list(state.values()).count(v) for v in set(state.values())}, flush=True)
                return
            ready = [k for k in ready if not (LOG/f'{cohort(k)}.guard-retry.json').exists() or
                     time.time() >= json.loads((LOG/f'{cohort(k)}.guard-retry.json').read_text())['retry_after']]
            for key in ready[:max(0,CAP-inflight)]:
                if not submit(key):
                    break
            time.sleep(45)


if __name__ == '__main__':
    main()
