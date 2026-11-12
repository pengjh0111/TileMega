"""Build the two selected-geometry arms and enqueue qualification then timing."""
import argparse
from pathlib import Path
import sys
sys.path = [p for p in sys.path if Path(p or '.').resolve() != Path(__file__).resolve().parent]
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
spec = importlib.util.spec_from_file_location('mx8_queue', ROOT / 'experiments/mx8/queue.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def prepare(cell_path, mode):
    cell = queue.read(cell_path)
    directory = ROOT / 'runs/mx8/ablation_build' / f'{cell["model"]}_B{cell["batch"]}' / mode
    binary = directory / 'plan.so'
    if not Path(str(binary) + '.identity.json').exists():
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'BUILD.log').open('a') as log:
            subprocess.run([sys.executable, str(ROOT / 'experiments/mx8/build_sequence.py'),
                '--source', cell['generated_source']['path'], '--out', str(directory), '--mode', mode],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    worker = ROOT / 'experiments/mx8/run_sequence.py'
    inputs = [queue.ref(p) for p in (cell_path, binary, worker, str(binary) + '.identity.json', cell['prompt'])]
    inputs += [cell['prefill'], cell['tokens']]
    base = f'ab4c_{cell["model"]}_B{cell["batch"]}_{mode}'
    qualifications = []
    for process in range(5):
        name = base + f'_check{process}'
        command = [sys.executable, str(worker), '--cell', str(cell_path), '--decode', str(binary),
            '--out', '{out}/result', '--task', 'qualify']
        if process == 0:
            command.append('--hf')
        queue.submit(ROOT / 'runs/mx8/queue', name, dict(kind='ablation_qualify', command=command,
            inputs=inputs, priority=0, needs_mib=40960 if cell['moe'] else 16384,
            timeout_s=1800, after=['graph_p0', 'graph_p1', 'graph_p2', 'graph_p3', 'graph_p4']))
        qualifications.append(name)
    for process in range(3):
        queue.submit(ROOT / 'runs/mx8/queue', base + f'_timing{process}', dict(kind='ablation_timing',
            command=[sys.executable, str(worker), '--cell', str(cell_path), '--decode', str(binary),
                '--out', '{out}/result', '--task', 'measure'], inputs=inputs, priority=0,
            needs_mib=40960 if cell['moe'] else 16384, timeout_s=900, after=qualifications))
    print('queued', base, '5 checks + 3 timing processes', flush=True)
    return dict(cell=queue.ref(cell_path), mode=mode, binary=queue.ref(binary), qualifications=qualifications)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-workers', type=int, default=2)
    args = parser.parse_args()
    tasks = [(p, mode) for p in sorted((ROOT / 'runs/mx8/cells').glob('*.json'))
        if queue.read(p)['batch'] in (1, 4, 16) for mode in ('graph', 'hostlaunch')]
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=args.build_workers) as executor:
        futures = {executor.submit(prepare, *task): task for task in tasks}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                task = futures[future]
                errors.append(dict(cell=str(task[0]), mode=task[1], error=str(error)))
                print('BUILD FAILED', task, error, flush=True)
    queue.save(ROOT / 'runs/mx8/ABLATION_QUEUE.json', dict(arms=results, errors=errors,
        reference='Frozen main experiment, not a new AB4b arm', timing='3 fresh processes, warm1 + timed3 for N1/N1024'))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
