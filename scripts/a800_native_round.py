"""Resume one guarded correctness process and one timing process per A800 artifact."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

WORK = Path(__file__).resolve().parents[2]
RUN = WORK / 'TileMega-a800/runs/a800/native'
IDENTITY = ('decode_sha256', 'prefill_sha256', 'target_sha256')
LIVE = {'PENDING', 'CONFIGURING', 'RUNNING', 'COMPLETING', 'REQUEUED', 'SUSPENDED'}


def command(args, timeout=30):
    return subprocess.run(args, cwd=WORK, text=True, capture_output=True, timeout=timeout)


def save(path, text):
    temporary = path.with_name(path.name + '.next')
    temporary.write_text(text)
    temporary.replace(path)


def accounting(job):
    result = command(['sacct', '--array', '-XnP', '-j', job, '-o', 'State,ExitCode'])
    result.check_returncode()
    return [line.split('|')[0] for line in result.stdout.splitlines() if line]


def wait_for_search(job, deadline):
    while time.monotonic() < deadline:
        states = accounting(job)
        if states == ['COMPLETED']:
            return
        if states and not any(s in LIVE for s in states):
            raise RuntimeError(f'search {job} ended with {states}')
        time.sleep(40)
    raise TimeoutError(f'search {job} exceeded the controller deadline')


def accepts(plan, stage):
    pattern = f"*/{plan['model']}/B{plan['batch']}/*/{stage}-*.json"
    rows = []
    for path in RUN.glob(pattern):
        row = json.loads(path.read_text())
        if row.get('pass') and all(row.get(key) == plan[key] for key in IDENTITY):
            rows.append(row)
    if stage == 'correctness':
        if len({r['tokens_sha256'] for r in rows}) > 1:
            raise RuntimeError('correctness processes produced different tokens')
        return any(row.get('hf_pass') for row in rows)
    return bool(rows)


def run_phase(label, stage, common, seconds, wall, script, deadline):
    prefix = WORK/'logs'/f'parallel-native-{label}-{stage}'
    def path(suffix):
        return Path(str(prefix) + suffix)

    result = command(['./run-env.sh','tilemega','python','scripts/a800_native.py','plan',
                      '--stage',stage,'--round','0','--budget-s',str(seconds),*common], 180)
    result.check_returncode()
    plan = json.loads(result.stdout)
    planfile = path('.plan.json')
    if planfile.exists():
        old = json.loads(planfile.read_text())
        if any(plan[k] != old[k] for k in (*IDENTITY,'batch','model','stage')):
            raise ValueError(f'planned identity changed for {prefix}')
    else:
        save(planfile, result.stdout)
    existing = set()
    if path('.id').is_file():
        existing.add(path('.id').read_text().strip())
    if path('.rounds.json').is_file():
        for jobs in json.loads(path('.rounds.json').read_text()).values():
            existing.update(jobs)
    ticket = path('.single.id')
    if ticket.exists():
        existing.add(ticket.read_text().strip())
    intent = path('.single.intent')
    while time.monotonic() < deadline:
        if accepts(plan, stage):
            return
        if intent.exists():
            raise RuntimeError(f'reconcile submission recorded at {intent}')
        states = [accounting(job) for job in existing]
        if any(not ss or any(s in LIVE for s in ss) for ss in states):
            time.sleep(45)
            continue
        if ticket.exists():
            raise RuntimeError(f'{ticket.read_text().strip()} ended without an accepted {stage} receipt: {states}')
        with intent.open('x') as f:
            f.write(str(time.time()))
        submitted = command(['sbatch','--parsable',f'--time={wall}','--array=0',
                             f'--output={prefix}-%A_%a.out',script,'--stage',stage,*common])
        if submitted.returncode:
            if any(reason in submitted.stderr for reason in ('AssocMaxSubmitJobLimit','QOSMaxSubmitJobPerUserLimit')):
                intent.unlink()
                time.sleep(35)
                continue
            raise RuntimeError(submitted.stderr)
        job = submitted.stdout.split(';')[0].strip()
        if not job.isdecimal():
            raise RuntimeError(f'check scheduler submission {intent}')
        save(ticket,job+'\n')
        if not path('.id').exists():
            save(path('.id'),job+'\n')
        existing.add(job)
        intent.unlink()
        print(f'{label} {stage} round 0 submitted as {job}', flush=True)
        time.sleep(30)
    raise TimeoutError(f'{prefix} exceeded the login controller deadline')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('search')
    parser.add_argument('model')
    parser.add_argument('batch',type=int)
    parser.add_argument('artifacts',nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not re.fullmatch('[a-z0-9-]+',args.label) or not args.search.isdecimal():
        parser.error('invalid label or search ID')
    large = args.model == 'qwen3_30b_a3b'
    small = args.model in ('llama3_2_1b','llama3_2_3b','qwen3_0_6b','qwen3_1_7b')
    timeout = 1080 if large else 420 if small else 570
    common = ['--model',args.model,'--batch',str(args.batch),'--timeout-s',str(timeout),*args.artifacts]
    script = 'qwen-hf-replicates-a800.sbatch' if large else 'native-replicates-a800.sbatch'
    timing = os.environ.get('TILEMEGA_TIMING_STAGE','measure')
    if timing not in ('measure','candidate-timing'):
        parser.error('unknown timing stage')
    stages = ['correctness']
    if os.environ.get('TILEMEGA_VALIDATION_ONLY') != '1':
        stages.append(timing)
    deadline = time.monotonic() + 24*3600
    with (WORK/'logs'/f'parallel-native-{args.label}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        wait_for_search(args.search,deadline)
        for stage in stages:
            run_phase(args.label,stage,common,1230 if large else 750,
                      '00:22:30' if large else '00:14:00',script,deadline)
    collected = command(['./refresh-results-a800.sh'], 1200)
    collected.check_returncode()


if __name__ == '__main__':
    main()
