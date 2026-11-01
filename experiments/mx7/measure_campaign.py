"""Schedule bounded single-card cost cohorts and seal the aggregation request."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--builds', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--chunk', type=int, default=64)
    parser.add_argument('--cards', default='7,6,5,4')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    cards = list(map(int, args.cards.split(',')))
    if not cards or len(set(cards)) != len(cards) or not set(cards) <= {4, 5, 6, 7}:
        parser.error('unique physical cards 4–7 are required')
    if args.chunk < 1:
        parser.error('positive key chunk required')
    builds, out = args.builds.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    keys = json.loads((builds / 'KEYS_CPU.json').read_text())
    driver = builds / 'frozen/experiments/mx7/measure_costs.py'
    source = json.loads((builds / 'SOURCE.json').read_text())
    expected = source['files']['experiments/mx7/measure_costs.py']
    if ref(driver)['sha256'] != expected:
        raise ValueError('measurement driver differs from the frozen CPU build')
    jobs = []
    for number, begin in enumerate(range(0, len(keys), args.chunk)):
        folder = out / f'job{number:04d}'
        folder.mkdir(exist_ok=True)
        path = folder / 'keys.json'
        if not path.exists():
            save(path, keys[begin:begin + args.chunk])
        if json.loads(path.read_text()) != keys[begin:begin + args.chunk]:
            raise ValueError('existing cost chunk differs; use a fresh campaign directory')
        jobs.append((number, folder, path))
    plan = dict(source=ref(builds / 'SOURCE.json'), all_keys=ref(builds / 'KEYS_CPU.json'),
                driver=ref(driver), domain=ref(builds / 'DOMAIN.json'),
                CPU_resource_rejections=json.loads((builds / 'REJECTIONS_CPU.json').read_text()),
                jobs=[dict(keys=ref(path)) for _, _, path in jobs])
    plan_path = out / 'PLAN.json'
    if not plan_path.exists():
        save(plan_path, plan)
    if json.loads(plan_path.read_text()) != plan:
        raise ValueError('campaign inputs changed')
    if args.prepare_only:
        print('prepared', len(jobs), 'chunks', len(keys), 'keys', flush=True)
        return
    environment = dict(os.environ, PYTHONPATH=str(ROOT / 'python'), NEEDS_MIB='12288')

    def card_jobs(card, assigned):
        completed = []
        for number, folder, path in assigned:
            records = []
            for round_number in (1, 2, 3):
                run = folder / f'round{round_number}'
                manifest = run / 'PROCESS.json'
                raw, guard = run / 'raw.jsonl', run / 'guard/guard_result.json'
                command = list(map(str, [ROOT / 'scripts/mx_gpu_job.sh', card, '900', run / 'guard',
                                        sys.executable, driver, '--keys', path, '--out', raw,
                                        '--round', round_number]))
                if manifest.exists():
                    prior = json.loads(manifest.read_text())
                    if prior['returncode'] or prior['argv'] != command or not raw.exists():
                        raise RuntimeError('failed/changed cost job requires an explicit remedy: ' + str(run))
                else:
                    run.mkdir(exist_ok=False)
                    start = time.time()
                    with (run / 'stdout.log').open('x') as stream:
                        child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                                 stdout=stream, stderr=subprocess.STDOUT)
                        status = child.wait()
                    save(manifest, dict(argv=command, card=card, PID=child.pid, start=start,
                                        end=time.time(), returncode=status))
                    if status:
                        raise RuntimeError('cost guard/tool failure; retain ' + str(run))
                records.append(dict(round=round_number, raw=ref(raw), process=ref(manifest), guard=ref(guard)))
            completed.append((number, dict(keys=ref(path), records=records)))
            print('completed cost chunk', number, 'card', card, flush=True)
        return completed

    completed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cards)) as pool:
        tasks = [pool.submit(card_jobs, card, jobs[index::len(cards)]) for index, card in enumerate(cards)]
        for task in tasks:
            completed += task.result()
    request = dict(plan, jobs=[row for _, row in sorted(completed)])
    save(out / 'REQUEST.json', request)
    print('complete cost cohorts', len(completed) * 3, flush=True)


if __name__ == '__main__':
    main()
