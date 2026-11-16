"""Conservative admission hints for borrowing otherwise idle DNN2 cards."""
from __future__ import annotations
import json
from pathlib import Path

import dnn1_baselines as base

PREFERRED = (5, 7)
BORROWABLE = (0, 1)
TERMINAL = {'done', 'failed', 'cancelled', 'canceled', 'skipped', 'not_run'}
DNN2 = Path('/root/TileMega-dnn2')


def main_reservations(gpu, root=DNN2):
    if gpu not in BORROWABLE:
        return []
    reasons = []
    if base.held(f'/tmp/tilemega_dnn2_scoreboard_gpu{gpu}.lock'):
        reasons.append('DNN2 scoreboard scheduler lock held')
    for filename in (root / 'runs/dnn2').rglob('queue_*.json'):
        try:
            queue = json.loads(filename.read_text())
            if not isinstance(queue, list):
                continue
            state_path = filename.parent.parent / 'scheduler/state.json'
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            for job in queue:
                if not job.get('gpu'):
                    continue
                env = job.get('env', {})
                card = env.get('TILEMEGA_DEVICE_INDEX', env.get('CUDA_VISIBLE_DEVICES'))
                if str(card) == str(gpu) and state.get(job['name'], {}).get('status') not in TERMINAL:
                    reasons.append('DNN2 queued/running guard: ' + str(filename))
        except (OSError, ValueError, KeyError, TypeError):
            reasons.append('DNN2 queue/state not yet readable: ' + str(filename))
    return reasons
