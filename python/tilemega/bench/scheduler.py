"""Serial dynamic queue with strict GPU guarding and a six-hour idle limit."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .gpu_guard import gpu, idle, stop

TERMINAL = {'done', 'failed', 'skipped', 'not_run', 'cancelled'}


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def ready_steps(steps, state, now):
    ready, skipped = [], []
    for step in steps.values():
        record = state[step['name']]
        if record['status'] != 'pending':
            continue
        required = [state.get(dep, {}).get('status') for dep in step.get('after', [])]
        if any(status in TERMINAL - {'done'} for status in required):
            skipped.append(step['name'])
        elif (all(status == 'done' for status in required)
              and all(state.get(dep, {}).get('status') in TERMINAL for dep in step.get('after_any', []))
              and record.get('retry_at', 0) <= now):
            ready.append(step)
    return sorted(ready, key=lambda s: (s.get('priority', 100), s['name'])), skipped


def timing_pause(control, waiting, available, now, limit):
    if control.get('timing_paused'):
        return
    if not waiting or available:
        control.pop('idle_wait_since', None)
    else:
        since = control.setdefault('idle_wait_since', now)
        if now - since >= limit:
            control.update(timing_paused=True, paused_at=now,
                           reason='no idle timing window for six hours')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--deadline-hours', type=float, default=96)
    parser.add_argument('--stay-open', action='store_true')
    parser.add_argument('--resume-timing', action='store_true')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    args.queue_dir.mkdir(parents=True, exist_ok=True)
    singleton = (args.out / 'scheduler.pid').open('a+')
    fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
    singleton.seek(0)
    singleton.truncate()
    singleton.write(str(os.getpid()) + '\n')
    singleton.flush()
    policy = json.loads(args.policy.read_text())
    state_path, control_path = args.out / 'state.json', args.out / 'control.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    control = json.loads(control_path.read_text()) if control_path.exists() else {}
    if args.resume_timing:
        control = {}
    deadline = time.time() + args.deadline_hours * 3600
    observed = None
    os.environ.setdefault('TILEMEGA_GPU_LOCK', str(Path.home() / '.cache/tilemega/gpu.lock'))

    def persist():
        write_json(state_path, state)
        write_json(control_path, control)

    def record(name, last):
        row = state[name]
        with (args.out / 'progress.tsv').open('a') as stream:
            stream.write('\t'.join(map(str, [name, row['status'], row.get('attempts', 0),
                row.get('wait_s', 0), row.get('run_s', 0), last.replace('\t', ' ')[:500]])) + '\n')
        if row['status'] in TERMINAL:
            write_json(args.out / (name + '.' + row['status']), row)
        persist()

    # Replaying an interrupted measurement without collecting its descendant
    # process identities would allow concurrent runs. Require explicit recovery.
    if any(row['status'] == 'running' for row in state.values()):
        raise RuntimeError('interrupted running step: inspect its processes before resetting state')
    while time.time() < deadline:
        steps = {}
        for path in sorted(args.queue_dir.glob('queue_*.json')):
            for step in json.loads(path.read_text()):
                name = step['name']
                if name in steps:
                    raise ValueError('duplicate step ' + name)
                if step.get('timing', step.get('gpu', False)) and not step.get('gpu'):
                    raise ValueError('timing steps must use the GPU guard')
                steps[name] = step
                state.setdefault(name, dict(status='pending', attempts=0, enqueued=time.time(), retry_at=0))
        ready, skipped = ready_steps(steps, state, time.time())
        for name in skipped:
            state[name]['status'] = 'skipped'
            record(name, 'required dependency failed')
        gpu_steps = [s for s in ready if s.get('gpu')]
        timing_steps = [s for s in gpu_steps if s.get('timing', True)]
        available, row = False, {}
        if gpu_steps:
            try:
                row = gpu()
                available = any(idle(row, policy, s.get('needs_free_mib', 12288)) for s in timing_steps)
            except Exception as error:
                row = dict(error=str(error))
            signature = (available, tuple(row.get('owners', {})),
                         row.get('hidden_mib', 0) > policy['max_hidden_mib'], row.get('error'))
            if signature != observed:
                with (args.out / 'occupancy.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(row, available=available)) + '\n')
                observed = signature
        pending_timing = any(state[s['name']]['status'] == 'pending'
            and s.get('timing', s.get('gpu', False))
            and all(state.get(dep, {}).get('status') == 'done' for dep in s.get('after', []))
            and all(state.get(dep, {}).get('status') in TERMINAL for dep in s.get('after_any', []))
            for s in steps.values())
        # A quick idle sample is not an accepted window: preflight or a later
        # interference check can still reject it. Only a completed clean run
        # resets the six-hour budget, including cooldown between attempts.
        timing_pause(control, pending_timing, False, time.time(), policy.get('max_idle_wait_s', 21600))
        chosen = next((s for s in gpu_steps
            if not (control.get('timing_paused') and s.get('timing', True))
            and 'error' not in row and idle(row, policy, s.get('needs_free_mib', 12288))), None)
        if chosen is None:
            chosen = next((s for s in ready if not s.get('gpu')), None)
        if chosen is None:
            persist()
            if steps and all(state[name]['status'] in TERMINAL for name in steps) and not args.stay_open:
                return
            if control.get('timing_paused') and not any(not s.get('timing', s.get('gpu', False)) for s in ready) and not args.stay_open:
                print('timing paused after six hours without an idle window', flush=True)
                return
            time.sleep(60)
            continue
        name = chosen['name']
        rec = state[name]
        rec['attempts'] += 1
        folder = args.out.parent / 'raw' / name / f"attempt{rec['attempts']}"
        folder.mkdir(parents=True, exist_ok=True)
        cmd = [str(value).replace('{attempt_dir}', str(folder)) for value in chosen['command']]
        if rec['attempts'] > 1:
            cmd += chosen.get('retry_args', [])
        env = dict(os.environ, **chosen.get('env', {}))
        env['TILEMEGA_GPU_LOCK'] = os.environ['TILEMEGA_GPU_LOCK']
        # A stale inherited marker must never let a CPU step bypass the lock.
        env.pop('TILEMEGA_GPU_LOCK_HELD', None)
        if chosen.get('gpu'):
            cmd = [sys.executable, '-m', 'tilemega.bench.gpu_guard', '--policy', str(args.policy),
                   '--out', str(folder), '--needs-free-mib', str(chosen.get('needs_free_mib', 12288)),
                   '--timeout-s', str(chosen['timeout_s']), '--', *cmd]
        rec.update(status='running', wait_s=time.time() - rec['enqueued'], out=str(folder), started=time.time())
        persist()
        begin = time.monotonic()
        log = args.out / (name + '.log')
        with log.open('a') as stream:
            stream.write(f"\nATTEMPT {rec['attempts']} {rec['started']}\n")
            stream.flush()
            proc = subprocess.Popen(cmd, cwd=chosen.get('cwd'), env=env,
                                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            rec['pid'] = proc.pid
            persist()
            try:
                code = proc.wait(timeout=chosen['timeout_s'] + (600 if chosen.get('gpu') else 0))
            except subprocess.TimeoutExpired:
                if chosen.get('gpu') and (folder / 'pgid').exists():
                    child_pid = int((folder / 'pgid').read_text())
                    # The recorded guard child remains in its own session.
                    subprocess.run(['pkill', '-KILL', '-s', str(child_pid)], check=False)
                stop(proc)
                code = 124
        rec.update(run_s=time.monotonic() - begin, exit_code=code, ended=time.time())
        if code == 75:
            result_path = folder / 'guard_result.json'
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            cooldown = 300 if result.get('reason', '').startswith('preflight') else 600
            rec.update(status='pending', retry_at=time.time() + cooldown)
        else:
            rec['status'] = 'done' if code == 0 else 'failed'
            if code == 0 and chosen.get('timing', chosen.get('gpu', False)):
                control.pop('idle_wait_since', None)
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size - 2048))
            lines = stream.read().decode(errors='replace').strip().splitlines()
        record(name, lines[-1] if lines else '')
    for name, rec in state.items():
        if rec['status'] == 'pending':
            rec['status'] = 'not_run'
            record(name, 'deadline reached')


if __name__ == '__main__':
    main()
