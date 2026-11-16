#!/usr/bin/env python3
"""Four-card, process-sized jobs with guarded, independently retryable attempts."""
import copy
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback

from sm90d1_common import ROOT, PYTHON, read, sha, write
from sm90r3_llm import environment, job_result, PCI
from sm90r4_inputs import model_dir
import sm90r4_dispatch as dispatch


class GPUItems:
    def __init__(self, directory):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.items = read(self.directory / 'items.json') if (self.directory / 'items.json').exists() else {}
        if any(v['status'] == 'running' for v in self.items.values()):
            raise ValueError('inspect interrupted running items before resuming')
        self.policy = read(dispatch.POLICY)
        self.stopping = threading.Event()
        self.threads = []
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                            '--format=csv,noheader'], text=True)
        actual = {int(line.split(',')[0]): line.split(',')[1].strip() for line in inventory.splitlines()}
        expected = self.policy['idle_power_mapping_provenance']['uuid_by_gpu']
        if any(actual.get(gpu) != expected[str(gpu)] for gpu in range(4)):
            raise ValueError('GPU 0-3 UUID mapping changed')
        write(self.directory / 'inventory.json', dict(gpus=actual, source_sha256=sha(__file__)))

    def save(self):
        write(self.directory / 'items.json', self.items)

    def add(self, job, imported=None):
        with self.lock:
            if job['id'] in self.items:
                if self.items[job['id']]['job'] != job:
                    raise ValueError('queued item definition changed: ' + job['id'])
                return
            self.items[job['id']] = dict(job=job, status='done' if imported else 'pending',
                attempts=[], result=imported, enqueued=time.time(), retry_at=0)
            self.save()

    def get(self, name):
        with self.lock:
            return copy.deepcopy(self.items.get(name))

    def start(self):
        for gpu in range(4):
            thread = threading.Thread(target=self.worker, args=(gpu,), daemon=True)
            thread.start()
            self.threads.append(thread)

    def ready(self):
        return [(name, value) for name, value in self.items.items()
                if value['status'] == 'pending' and value['retry_at'] <= time.time()
                and all(self.items.get(dep, {}).get('status') == 'done'
                        for dep in value['job'].get('after', []))]

    def claim(self, gpu):
        # Claims are held only by an active process-sized item, never during
        # cooldown or while waiting for another card. The guard owns the GPU lock.
        with dispatch.MUTEX.open('a') as mutex:
            fcntl.flock(mutex, fcntl.LOCK_EX)
            if dispatch.machine_waiting():
                return None
            if not dispatch.compatible('trial', gpu, dispatch.live_claims()):
                return None
            if not dispatch.idle_card(gpu, self.policy):
                return None
            used = float(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
            if used > 16:  # Allow only driver bookkeeping on an otherwise empty card.
                return None
            handle = Path(f'/tmp/tilemega_claim{gpu}.lock').open('a')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                return None
            return handle

    def command(self, job, gpu, attempt):
        env = dict(os.environ, **environment(gpu, 'trial'), MKL_NUM_THREADS='8',
                   TILEMEGA_TIMING='0' if job['kind'] != 'llm' or job['spec']['kind'] == 'corr' else '1',
                   TILEMEGA_PREFLIGHT_LOCK_TIMEOUT_S='1')
        for key in ('TILEMEGA_GPU_LOCK_HELD', 'TILEMEGA_DEVICE_GROUP', 'TILEMEGA_RUN_NONCE'):
            env.pop(key, None)
        if job['kind'] == 'llm':
            spec = copy.deepcopy(job['spec'])
            spec.update(gpu=gpu, concurrency='trial')
            for arm in spec['arms'].values():
                arm.update(decode_sha256=sha(arm['decode']), prefill_sha256='prompt mode')
            write(attempt / 'spec.json', spec)
            command = [PYTHON, str(ROOT / 'scripts/run_r16_with_device_identity.py'),
                '--guard-policy', str(dispatch.POLICY), '--out', str(attempt / 'job'),
                '--expected-uuid', self.policy['idle_power_mapping_provenance']['uuid_by_gpu'][str(gpu)],
                '--expected-pci', f'0000:{PCI[gpu]}:00.0', '--numa-node', str(dispatch.numa(gpu)), '--',
                PYTHON, str(ROOT / 'scripts/sm90r3_llm.py'), 'measure', '--spec', str(attempt / 'spec.json'),
                '--arm', job['arm'], '--fresh', str(job['slot']), '--out', str(attempt / 'job')]
            timeout, needs = (5400 if spec['kind'] == 'corr' else 3600), 70000
        elif job['kind'] == 'c2':
            check = [PYTHON, str(ROOT / 'scripts/sm90r6_realize_check.py'), '--reference', job['reference'],
                '--candidate', job['candidate'], '--prompts', job['prompts'], '--batch', str(job['batch']),
                '--steps', '1024', '--out', str(attempt / 'checks')]
            command = [PYTHON, str(ROOT / 'python/tilemega/serving/random_checkpoint.py'),
                '--config', str(model_dir(job['model'])), '--out', str(attempt / 'checkpoint'),
                '--seed', '0', '--max-gib', '64', '--', *check]
            timeout, needs = 7200, 70000
        elif job['kind'] == 'occupancy':
            prepared = copy.deepcopy(job['prepared'])
            for row in prepared:
                folder = attempt / f'candidate-{row["rank"]}'
                folder.mkdir()
                row['directory'] = str(folder)
            write(attempt / 'prepared.json', prepared)
            command = [PYTHON, str(ROOT / 'scripts/sm90r6_probe_native.py'), '--worker',
                       str(attempt / 'prepared.json'), '--out', str(attempt / 'occupancy.json')]
            timeout, needs = 300, 2048
        else:
            raise ValueError('unknown GPU item kind')
        command = [PYTHON, str(ROOT / 'scripts/sm90r6_guard.py'), '--policy', str(dispatch.POLICY),
            '--out', str(attempt), '--needs-free-mib', str(needs), '--timeout-s', str(timeout), '--',
            'numactl', f'--cpunodebind={dispatch.numa(gpu)}', *command]
        return command, env, timeout

    def execute(self, job, gpu, attempt):
        command, env, timeout = self.command(job, gpu, attempt)
        write(attempt / 'command.json', dict(command=command, gpu=gpu, job=job))
        with (attempt / 'run.log').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            write(attempt / 'process.json', dict(pid=process.pid, started=time.time()))
            try:
                code = process.wait(timeout=timeout + 300)
            except subprocess.TimeoutExpired:
                pgid = attempt / 'pgid'
                if pgid.exists():
                    subprocess.run(['pkill', '-TERM', '-s', pgid.read_text().strip()], check=False)
                process.terminate()
                process.wait(timeout=10)
                raise RuntimeError('guard exceeded the item timeout')
        guard = read(attempt / 'guard_result.json')
        if code or guard['code']:
            return code or guard['code'], dict(reason=guard['reason'])
        if job['kind'] == 'llm':
            result = job_result({job['id']: dict(status='done', out=str(attempt))}, job['id'])
            result.update(arm=job['arm'], raw=str(attempt))
            if job['spec']['kind'] == 'corr' and not result.get('c1', {}).get('pass'):
                raise ValueError('C1 numerical check failed')
        elif job['kind'] == 'c2':
            checks = read(attempt / 'checks/result.json')
            if not checks['passed']:
                raise ValueError('C2 numerical check failed')
            result = dict(directory=str(attempt), gpu=gpu, checkpoint=checks['checkpoint'],
                          tokens_sha256=checks['checks'][0]['tokens_sha256'])
        else:
            result = read(attempt / 'occupancy.json')
        result.update(gpu=gpu, origin='dynamic', item=job['id'])
        return 0, result

    def worker(self, gpu):
        while not self.stopping.is_set():
            handle = None
            selected = None
            try:
                with self.lock:
                    ready = self.ready()
                if not ready:
                    self.stopping.wait(2)
                    continue
                handle = self.claim(gpu)
                if handle is None:
                    self.stopping.wait(2)
                    continue
                with self.lock:
                    ready = self.ready()
                    if not ready:
                        continue
                    selected, record = min(ready, key=lambda x: (
                        x[1]['job'].get('priority', 10),
                        x[1]['attempts'][-1]['started'] if x[1]['attempts'] else 0,
                        x[1]['job'].get('slot', 0), x[0]))
                    attempt = self.directory / selected / f'attempt-{len(record["attempts"])}'
                    attempt.mkdir(parents=True)
                    record.update(status='running', gpu=gpu)
                    record['attempts'].append(dict(directory=str(attempt), gpu=gpu, started=time.time()))
                    job = copy.deepcopy(record['job'])
                    self.save()
                write(Path(f'/tmp/tilemega_claim{gpu}.json'), dict(pid=os.getpid(),
                      **{'class': 'trial'}, unit=selected, since=time.time()))
                code, result = self.execute(job, gpu, attempt)
                with self.lock:
                    record = self.items[selected]
                    record['attempts'][-1].update(ended=time.time(), code=code, result=result)
                    record.update(status='done' if code == 0 else 'pending' if code == 75 else 'failed',
                                  result=result if code == 0 else None, retry_at=time.time() + 2)
                    self.save()
                print(f'{selected} GPU{gpu} code={code}', flush=True)
            except Exception as error:
                with self.lock:
                    if selected:
                        self.items[selected].update(status='failed', error=str(error), traceback=traceback.format_exc())
                        self.save()
                print(f'GPU{gpu} item failure: {error}', flush=True)
                self.stopping.wait(2)
            finally:
                if handle is not None:
                    handle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--requests', type=Path, required=True)
    args = parser.parse_args()
    queue = GPUItems(args.out)
    queue.start()
    while True:
        for path in sorted(args.requests.glob('*.json')):
            request = read(path)
            queue.add(request['job'], request.get('imported'))
        if (args.requests / 'CLOSED').exists():
            with queue.lock:
                if all(v['status'] in ('done', 'failed') for v in queue.items.values()):
                    break
        time.sleep(2)
    queue.stopping.set()
    for thread in queue.threads:
        thread.join()
