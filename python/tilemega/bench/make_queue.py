"""Freeze all arm definitions and expand a measurement configuration."""
import json
import re
import sys
from pathlib import Path

from .arms import register


def prepare(config, root, out, *, python=None):
    if config.get('schema') != 'tilemega.bench.v1':
        raise ValueError('expected tilemega.bench.v1 configuration')
    root, out = Path(root).resolve(), Path(out).resolve()
    python = python or sys.executable
    data = dict(config)
    data['measurement_policy'] = str((root / config['measurement_policy']).resolve())
    data['cells'] = {}
    steps = []
    for name, cell in config['cells'].items():
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
            raise ValueError('unsafe cell name')
        workload = cell['workload']
        if any(int(workload[k]) <= 0 for k in ('isl', 'osl', 'batch')) or int(workload['osl']) < 2:
            raise ValueError('workload must have positive ISL/batch and OSL >= 2')
        prompt = (root / cell['prompt_ids']).resolve()
        ids = json.loads(prompt.read_text())
        if len(ids) < workload['batch'] or any(len(row) != workload['isl'] for row in ids):
            raise ValueError('prompt table does not match workload')
        arms, labels = [], set()
        for source in cell['arms']:
            arm = dict(source)
            label = arm['label']
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', label) or label in labels:
                raise ValueError('duplicate or unsafe arm label')
            labels.add(label)
            arm.setdefault('root', str(root))
            arm['root'] = str((root / arm['root']).resolve())
            arm.setdefault('python', python)
            if arm['kind'] == 'tm':
                for phase in ('prefill', 'decode'):
                    arm[phase] = str((root / arm[phase]).resolve())
                if arm.get('mode', 'L1') not in ('L1', 'L2', 'auto') or arm.get('decode_loop', 0) not in (0, 1, '0', '1', 'auto'):
                    raise ValueError('invalid execution mode or loop')
            arms.append(register(arm))
        if not arms:
            raise ValueError('empty comparison')
        data['cells'][name] = dict(cell, arms=arms, prompt_ids=str(prompt),
                                   model_path=str((root / cell['model_path']).resolve()))
        for rnd in range(int(config.get('rounds', 3))):
            steps.append(dict(name=f'{name}_r{rnd}', command=[python, '-m', 'tilemega.bench.anchor',
                '--arms', str(out / 'arms.json'), '--cell', name, '--round', str(rnd),
                '--out', '{attempt_dir}'], cwd=str(root), env={'PYTHONPATH': str(root / 'python')},
                gpu=True, timing=True, priority=10, after=[], timeout_s=config.get('timeout_s', 7200),
                needs_free_mib=cell.get('needs_free_mib', 12288)))
    out.mkdir(parents=True, exist_ok=True)
    queue = out / 'queue'
    queue.mkdir(exist_ok=True)
    (out / 'arms.json').write_text(json.dumps(data, indent=2) + '\n')
    (queue / 'queue_measure.json').write_text(json.dumps(steps, indent=2) + '\n')
    return steps
