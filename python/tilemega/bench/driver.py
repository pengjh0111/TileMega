"""The bench configuration entry point; the scheduler owns all timing."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .make_queue import prepare


def run(config_path, *, root=None, out=None, prepare_only=False):
    root = Path(root or Path(__file__).resolve().parents[3]).resolve()
    config = json.loads(Path(config_path).read_text())
    out = Path(out or root / config.get('output', 'runs/bench')).resolve()
    try:
        out.relative_to(root / 'runs')
    except ValueError as error:
        raise ValueError('benchmark output must live under ignored runs/') from error
    if (out / 'arms.json').exists():
        raise FileExistsError('use an unused output directory; resume the existing scheduler separately')
    steps = prepare(config, root, out)
    print(f'prepared {len(steps)} guarded measurement steps in {out}', flush=True)
    if prepare_only:
        return
    policy = (root / config.get('guard_policy', 'configs/bench/guard_policy.json')).resolve()
    env = dict(os.environ, PYTHONPATH=str(root / 'python'))
    env['TILEMEGA_GPU_LOCK'] = str(config.get('gpu_lock',
        os.environ.get('TILEMEGA_GPU_LOCK', Path.home() / '.cache/tilemega/gpu.lock')))
    log = (out / 'scheduler.log').open('a')
    process = subprocess.Popen([sys.executable, '-m', 'tilemega.bench.scheduler',
        '--queue-dir', str(out / 'queue'), '--out', str(out / 'scheduler'),
        '--policy', str(policy), '--deadline-hours', str(config.get('deadline_hours', 96))],
        cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
    log.close()
    (out / 'launcher.json').write_text(json.dumps({'pid': process.pid}) + '\n')
    print(f'scheduler PID {process.pid}; progress: {out}/scheduler/state.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run(args.config, out=args.out, prepare_only=args.prepare_only)
